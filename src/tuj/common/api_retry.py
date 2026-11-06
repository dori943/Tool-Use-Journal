# -*- coding: utf-8 -*-
"""일시적 API 장애(과부하·레이트리밋·타임아웃·연결 끊김)를 자동 재시도하고, 끝내 실패하면
`API_UNAVAILABLE` 로 분류되는 예외를 올린다.

왜: Gemini/OpenAI 가 503 "high demand" 를 돌려주면 지금까지는 M5 계획 실패
(failure_stage=planning)로 기록돼, Ours/ablation 의 성공률에 API 사정이 섞였다.
이 모듈은 (1) 같은 요청을 지수 백오프로 다시 보내고, (2) 그래도 안 되면 메시지에
`API_UNAVAILABLE` 을 담아 failure_taxonomy 가 `infrastructure` 로 분류하게 한다.

재시도 대상: HTTP 408/429/500/502/503/504, SDK 의 연결·타임아웃·레이트리밋·서버 오류,
메시지에 high demand / overloaded / UNAVAILABLE / RESOURCE_EXHAUSTED 등이 있는 경우.
재시도하지 않음: 400(잘못된 요청)·401/403(키)·404(모델 이름) 등 다시 보내도 같은 결과인 오류.

설정 (환경변수)
  TUJ_API_RETRIES     재시도 횟수 (기본 6 → 최대 7회 호출)
  TUJ_API_BACKOFF_S   첫 대기 초 (기본 10, 매번 2배, 지터 ±25%)
  TUJ_API_BACKOFF_MAX 최대 대기 초 (기본 120)
같은 요청을 다시 보내므로 응답 내용(=실험 조건)은 바뀌지 않는다. 호출 횟수·대기 시간은
api_retry_log 로 남는다.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Callable

TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_TYPES = frozenset({
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
    "ServiceUnavailableError", "Timeout", "ReadTimeout", "ConnectTimeout",
    "ConnectionError", "RemoteDisconnected", "TimeoutError",
})
_TRANSIENT_TEXT = (
    "high demand", "overloaded", "unavailable", "resource_exhausted", "rate limit",
    "rate_limit", "too many requests", "timed out", "timeout", "temporarily",
    "try again later", "connection reset", "connection aborted", "bad gateway",
    "gateway timeout", "internal error",
)
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 422})

# 프로세스 전체 재시도 기록 (run_result 에 첨부하기 위함)
api_retry_log: list[dict[str, Any]] = []


class TransientAPIError(RuntimeError):
    """재시도를 다 썼는데도 일시 장애가 계속됨. 메시지는 항상 API_UNAVAILABLE 로 시작."""

    def __init__(self, message: str, *, status_code=None, attempts: int = 0):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.attempts = attempts


def _status(error: BaseException):
    for attr in ("status_code", "http_status", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_transient(error: BaseException) -> bool:
    if isinstance(error, TransientAPIError):
        return True
    status = _status(error)
    if status in _PERMANENT_STATUS:
        return False
    if status in TRANSIENT_STATUS:
        return True
    if type(error).__name__ in _TRANSIENT_TYPES:
        return True
    text = f"{getattr(error, 'message', '')} {error}".lower()
    return any(token in text for token in _TRANSIENT_TEXT)


def _settings():
    retries = max(0, int(os.environ.get("TUJ_API_RETRIES", "6")))
    base = max(0.0, float(os.environ.get("TUJ_API_BACKOFF_S", "10")))
    cap = max(base, float(os.environ.get("TUJ_API_BACKOFF_MAX", "120")))
    return retries, base, cap


def retry_call(fn: Callable[[], Any], *, label: str = "api") -> Any:
    """fn() 을 호출. 일시 장애면 백오프 후 재호출, 영구 오류는 즉시 그대로 올린다."""
    retries, base, cap = _settings()
    last: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            result = fn()
            if attempt:
                api_retry_log.append({"label": label, "recovered_after": attempt})
                print(f"[api-retry] {label}: {attempt}회 재시도 후 성공", flush=True)
            return result
        except BaseException as error:  # noqa: BLE001 - SDK 예외 계층이 다양함
            if isinstance(error, (KeyboardInterrupt, SystemExit)) or not is_transient(error):
                raise
            last = error
            if attempt == retries:
                break
            wait = min(cap, base * (2 ** attempt)) * random.uniform(0.75, 1.25)
            status = _status(error)
            print(f"[api-retry] {label}: 일시 장애 ({status or type(error).__name__}) "
                  f"→ {wait:.0f}s 후 재시도 {attempt + 1}/{retries}", flush=True)
            time.sleep(wait)
    status = _status(last) if last is not None else None
    detail = str(getattr(last, "message", "") or last)[:600]
    api_retry_log.append({"label": label, "exhausted": True, "status": status, "detail": detail[:200]})
    raise TransientAPIError(
        f"API_UNAVAILABLE: {label}: {retries + 1}회 호출 모두 일시 장애 "
        f"(HTTP {status or type(last).__name__}): {detail}",
        status_code=status, attempts=retries + 1,
    ) from last


class _Wrapped:
    """client.chat.completions.create / client.responses.parse|create 만 가로챈다."""

    _INTERCEPT = {("chat", "completions", "create"), ("responses", "parse"), ("responses", "create")}

    def __init__(self, target, path=(), label="llm"):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_label", label)

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        path = self._path + (name,)
        if path in self._INTERCEPT and callable(attr):
            label = f"{self._label}.{'.'.join(path)}"
            return lambda *a, **k: retry_call(lambda: attr(*a, **k), label=label)
        if any(p[: len(path)] == path for p in self._INTERCEPT):
            return _Wrapped(attr, path, self._label)
        return attr

    def __setattr__(self, name, value):
        setattr(self._target, name, value)


def with_api_retry(client, label: str = "llm"):
    """OpenAI 호환 클라이언트를 재시도 프록시로 감싼다 (이미 감싸져 있으면 그대로)."""
    if client is None or isinstance(client, _Wrapped):
        return client
    return _Wrapped(client, (), label)
