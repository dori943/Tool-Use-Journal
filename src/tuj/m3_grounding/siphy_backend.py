"""M3-(a) SiPhy 백엔드 — 공개 코드(DominoAI-Lab/SiPhy-ECCV-2026)의 질량 레시피 이식.

원 파이프라인(ABO500, 5단계·2 venv·SAM2·BLIP-2)을 우리 스케일(단일 뷰 depth+seg,
객체당 VLM 1회)로 번역:

  SiPhy 원본                               →  우리 이식
  ───────────────────────────────────────────────────────────────
  captioning.py (BLIP-2 캡션)              ┐
  material_proposal.py (GPT 재료+밀도/두께) ├→ VLM 1콜 JSON (gpt4v_candidate_materials
  mask_material_proposal.py (BLIP-2 0-10   │   방식 + PRED_THICKNESS/YOUNG 프롬프트 병합,
    confidence → 정규화 분포)              ┘   confidence 0-10 → 정규화 분포)
  predict_property_integral:
    dense_pts = 마스크 표면점              →  robosuite depth 점군 (M1 _points)
    mat_cell_volumes = s²×thickness        →  동일 (s = 최근접 점 간격 중앙값)
    mass = Σ probs@(density×cellvol)       →  동일 (one-mask: 객체당 분포 1개)
    carving bound로 부피 상한              →  bbox 부피 상한 (단일 뷰라 카빙 불가)
    × correction_factor 0.6                →  동일 (mass.json 기본값)

top-1 확정 게이트 (우리 추가): gap = probs[top1] − probs[top2] > TOP1_GAP 이면
  재료 분포를 top-1에 원-핫 확정 → density·mass·μ 상류가 top-1 재료 값으로 일관.
  gap 작으면 SiPhy 원 철학(재료 확정 안 함, 분포 가중평균) 유지.

API 키: OPENAI_API_KEY(우선) 또는 GEMINI_API_KEY/GOOGLE_API_KEY 환경변수, 혹은 레포 루트
  my_api_key.py (SiPhy 관례). Gemini 는 OpenAI 호환 엔드포인트로 자동 폴백.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re

import numpy as np

from .intrinsic import PropertyBackend

# SiPhy configs/abo500/mass.json 기본값
CORRECTION_FACTOR = 0.60
K_MATERIALS = 5
MAX_TRIES = 3        # gpt_wrapper의 재시도 (원본 10 → 우리 예산에 맞게 축소)
TOP1_GAP = 0.15      # top-1과 top-2 확률차가 이 이상이면 top-1 확정 (분포 → 원-핫)

# 원본 프롬프트 3종(PRED_CAND_MATS_DENSITY_SYS_MSG_4V + PRED_THICKNESS_SYS_MSG +
# PRED_CAND_MATS_YOUNG_MODULUS_SYS_MSG + BLIP-2 0-10 confidence)을 1콜 JSON으로 병합.
SYS_MSG = """You will be given an image of an object (background masked to black). Based on the image, give me a short (5-10 words) description of what the object is, and also %d materials that the object might be made of. For each material give: its mass density (in kg/m^3), the thickness (in cm) of that material in the object, its Young's modulus (in GPa), and, on a scale from 0 to 10, how likely it is that this object is made of that material. You may provide a range low-high of values instead of a single value for density, thickness and Young's modulus. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    "description": description,
    "materials": [
        {"name": material1, "density_kgm3": "low-high", "thickness_cm": "low-high", "youngs_gpa": "low-high", "confidence_0_10": number},
        ...
    ]
}
Do not include any other text in your answer. Do not include unnecessary words besides the material in the material name.
""" % K_MATERIALS


def _parse_range(v) -> tuple[float, float]:
    """SiPhy parse_material_list의 'low-high' 규약 (단일값 허용, 콤마 제거)."""
    if isinstance(v, (int, float)):
        return float(v), float(v)
    s = str(v).replace(",", "").strip()
    nums = re.findall(r"\d+\.?\d*(?:[eE][+-]?\d+)?", s)   # 'low-high'의 -는 구분자 (음수 없음)
    if not nums:
        raise ValueError(f"range 파싱 실패: {v!r}")
    lo = float(nums[0])
    hi = float(nums[1]) if len(nums) > 1 else lo
    return min(lo, hi), max(lo, hi)


def _parse_response(text: str) -> dict:
    """```json 펜스 제거 후 JSON 파싱 (mask_material_proposal.py의 parse_fn 방식)."""
    raw = json.loads(text.replace("```json", "").replace("```", "").strip())
    mats = []
    for m in raw["materials"]:
        mats.append({
            "name": str(m["name"]).lower(),
            "density": _parse_range(m["density_kgm3"]),
            "thickness_cm": _parse_range(m["thickness_cm"]),
            "youngs_gpa": _parse_range(m.get("youngs_gpa", 0)),
            "confidence": max(float(m.get("confidence_0_10", 0)), 0.0),
        })
    if not mats:
        raise ValueError("materials 비어 있음")
    return {"caption": raw.get("description", ""), "materials": mats}


def _cell_size_m(pts_m: np.ndarray, max_pts: int = 2000) -> float:
    """표면셀 크기 s = 최근접 점 간격 중앙값 (SiPhy surface_cell_size의 점군 버전)."""
    if len(pts_m) > max_pts:
        pts_m = pts_m[np.random.default_rng(0).choice(len(pts_m), max_pts, replace=False)]
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts_m).query(pts_m, k=2)
        return float(np.median(d[:, 1]))
    except ImportError:                                # scipy 없으면 밀도 근사
        span = np.ptp(pts_m[:, :2], axis=0)
        return float(np.sqrt(max(span[0] * span[1], 1e-12) / max(len(pts_m), 1)))


def shell_mass_integral(points_mm, probs, dens, thick_cm, correction=CORRECTION_FACTOR):
    """SiPhy predict_physical_property_integral의 one-mask 단순형.

    points_mm (N,3) 표면점 / probs (K,) 재료 분포 / dens·thick (K,2) low-high.
    mass = N · s² · probs @ (density ⊙ thickness), 부피는 bbox로 상한.
    → {"mass_kg": mid, "mass_range_kg": [lo,hi], "volume_bounded": bool}
    """
    pts = np.asarray(points_mm, dtype=np.float64) / 1000.0
    probs = np.asarray(probs, dtype=np.float64)
    dens = np.asarray(dens, dtype=np.float64)                    # (K,2) kg/m³
    thick = np.asarray(thick_cm, dtype=np.float64) / 100.0       # (K,2) m
    s = _cell_size_m(pts)
    n = len(pts)

    cell_vols = s ** 2 * thick                                   # (K,2) m³/셀
    total_vol = n * (probs @ cell_vols)                          # (2,)
    mass = n * (probs @ (dens * cell_vols))                      # (2,)

    ext = np.maximum(np.ptp(pts, axis=0), s)                     # 평면 축 붕괴 방지 (≥셀 크기)
    bound_vol = float(np.prod(ext))                              # bbox 상한 (카빙 대체)
    bounded = bool(total_vol.max() > bound_vol)
    if bounded:                                                  # 원본: bound/total로 축소
        mass = mass * (bound_vol / total_vol.max())

    mass = np.sort(mass * correction)
    return {"mass_kg": round(float(mass.mean()), 3),
            "mass_range_kg": [round(float(mass[0]), 3), round(float(mass[1]), 3)],
            "cell_size_mm": round(s * 1000.0, 2), "volume_bounded": bounded}


def _effective_probs(probs: np.ndarray, gap_threshold: float = TOP1_GAP):
    """top-1 gap이 임계 이상이면 원-핫 확정, 아니면 원 분포 유지.
    → (probs_used, top_idx, gap, committed_top1)  committed는 M2/로깅용 플래그."""
    order = np.argsort(-probs)
    top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    gap = float(probs[top] - probs[second])
    if gap > gap_threshold:
        p = np.zeros_like(probs); p[top] = 1.0
        return p, top, gap, True
    return probs, top, gap, False


# Gemini 는 OpenAI 호환 엔드포인트를 제공 → OPENAI_API_KEY 없고 GEMINI_API_KEY 만
# 있어도 동일 client 로 돌아가게 한다.
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class SiPhyBackend(PropertyBackend):
    """crop RGB 1장 → material/density/mass/E (+ 재료 후보 분포).

    client 주입 가능 (테스트·타 VLM 교체용): client.chat.completions.create 인터페이스.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 client=None, repo_root=None, seed: int = 100, verbose: bool = False):
        self.model, self.seed, self.verbose = model, seed, verbose
        if client is not None:
            self.client = client
        else:                                           # 키 출처에 따라 model 도 조정될 수 있음
            self.client, self.model = self._make_client(api_key, repo_root, model)
        # Gemini OpenAI 호환 엔드포인트는 seed 파라미터를 지원하지 않는다(400).
        burl = str(getattr(self.client, "base_url", "") or "")
        self._is_gemini = ("generativelanguage" in burl
                           or str(self.model).startswith("gemini"))
        self._supports_seed = not self._is_gemini
        # Gemini 2.5 계열은 thinking 토큰이 max_tokens 를 잠식 → 500 이면 JSON 이 잘린다.
        self._max_tokens = 4096 if self._is_gemini else 500

    @staticmethod
    def _make_client(api_key, repo_root, model="gpt-4o-mini"):
        """OpenAI 우선, 없으면 Gemini(OpenAI 호환 엔드포인트)로 폴백.

        키 탐색: 인자 → 환경변수 → 레포 루트 my_api_key.py.
        OPENAI_API_KEY 있으면 OpenAI, 없고 GEMINI_API_KEY(또는 GOOGLE_API_KEY) 있으면
        Gemini 를 쓴다. model 이 gpt* 기본값이면 Gemini 기본 모델로 바꾼다.
        반환: (client, 실제 사용할 model 이름).
        """
        def _from_repo(var):                            # SiPhy 관례: 루트 my_api_key.py
            if not repo_root:
                return None
            f = os.path.join(str(repo_root), "my_api_key.py")
            if not os.path.exists(f):
                return None
            ns: dict = {}
            exec(open(f, encoding="utf-8").read(), ns)
            return ns.get(var)

        from openai import OpenAI
        okey = api_key or os.environ.get("OPENAI_API_KEY") or _from_repo("OPENAI_API_KEY")
        gkey = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                or _from_repo("GEMINI_API_KEY") or _from_repo("GOOGLE_API_KEY"))
        ms = str(model or "")
        # provider: run.py/M2 와 동일 규약(TUJ_LLM_PROVIDER) → 모델 접두어 → 키 자동감지
        provider = (os.environ.get("TUJ_LLM_PROVIDER")
                    or ("gemini" if ms.startswith("gemini")
                        else "openai" if ms.startswith(("gpt", "o1", "o3", "o4", "chatgpt"))
                        else None))
        if provider == "gemini":
            if not gkey:
                raise RuntimeError("GEMINI_API_KEY 없음 (TUJ_LLM_PROVIDER=gemini) — "
                                   "환경변수 또는 my_api_key.py 필요")
            gmodel = model if ms.startswith("gemini") else _DEFAULT_GEMINI_MODEL
            return OpenAI(api_key=gkey, base_url=_GEMINI_BASE_URL), gmodel
        if provider == "openai":
            if not okey:
                raise RuntimeError("OPENAI_API_KEY 없음 (TUJ_LLM_PROVIDER=openai) — "
                                   "환경변수 또는 my_api_key.py 필요")
            return OpenAI(api_key=okey), model
        # provider 미지정: OpenAI 우선, 없으면 Gemini
        if okey:
            return OpenAI(api_key=okey), model
        if gkey:
            gmodel = model if ms.startswith("gemini") else _DEFAULT_GEMINI_MODEL
            return OpenAI(api_key=gkey, base_url=_GEMINI_BASE_URL), gmodel
        raise RuntimeError("OPENAI_API_KEY / GEMINI_API_KEY 없음 — 환경변수 또는 "
                           "my_api_key.py 필요 (오프라인이면 MockBackend 사용)")

    # ── VLM 1콜 ──────────────────────────────────────
    def _propose(self, crop_rgb) -> dict:
        b64 = _to_b64_png(crop_rgb)
        last_err = None
        for t in range(MAX_TRIES):                      # gpt_wrapper 방식: seed+t 재시도
            try:
                kwargs = dict(
                    model=self.model, max_tokens=self._max_tokens,
                    messages=[
                        {"role": "system", "content": SYS_MSG},
                        {"role": "user", "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
                    ])
                if self._supports_seed:              # Gemini 는 seed 미지원 → 생략
                    kwargs["seed"] = self.seed + t
                r = self.client.chat.completions.create(**kwargs)
                if self.verbose and getattr(r, "usage", None):
                    print(f"  [siphy] tokens: {r.usage.total_tokens}")
                choice = r.choices[0]
                text = choice.message.content or ""
                if getattr(choice, "finish_reason", None) == "length":
                    raise RuntimeError(
                        f"응답이 max_tokens={self._max_tokens} 에서 잘림(finish_reason=length). "
                        f"앞부분: {text[:120]!r}")
                try:
                    return _parse_response(text)
                except (json.JSONDecodeError, KeyError, ValueError) as pe:
                    raise RuntimeError(f"JSON 파싱 실패: {pe}. 응답 앞부분: {text[:200]!r}") from pe
            except Exception as e:                      # noqa: BLE001
                last_err = e
        raise RuntimeError(f"SiPhy VLM 호출/파싱 {MAX_TRIES}회 실패: {last_err}")

    # ── PropertyBackend 인터페이스 ───────────────────
    def estimate(self, crop_rgb, cls_hint: str, points_mm=None) -> dict:
        if crop_rgb is None:
            raise ValueError(f"SiPhyBackend는 crop RGB 필요 (node class={cls_hint})")
        prop = self._propose(crop_rgb)
        mats = prop["materials"]

        conf = np.array([m["confidence"] for m in mats], dtype=np.float64)
        probs_raw = conf / conf.sum() if conf.sum() > 0 else np.full(len(mats), 1 / len(mats))
        # top-1 gap 게이트: gap > 임계면 원-핫 확정, 아니면 분포 유지 (원 SiPhy 철학)
        probs, top, gap, committed = _effective_probs(probs_raw)

        dens = np.array([m["density"] for m in mats])            # (K,2)
        thick = np.array([m["thickness_cm"] for m in mats])      # (K,2)
        youngs = np.array([m["youngs_gpa"] for m in mats])       # (K,2)

        out = {
            "material": mats[top]["name"],
            "density_kgm3": round(float(probs @ dens.mean(axis=1)), 1),   # committed면 top-1 밀도
            "youngs_gpa": round(float(probs @ youngs.mean(axis=1)), 2),   # committed면 top-1 E
            "mass_kg": None,
            "confidence": round(float(probs_raw[top]), 3),
            "material_committed": committed,                     # True면 top-1 확정 사용
            "top1_gap": round(gap, 3),                           # 게이트 판단 근거
            "caption": prop["caption"],
            "materials_topk": [                                  # 원 분포 (원-핫 여부와 무관하게 보존)
                {"name": m["name"], "prob": round(float(p), 3),
                 "density_kgm3": list(m["density"]), "thickness_cm": list(m["thickness_cm"])}
                for m, p in zip(mats, probs_raw)],
        }
        if points_mm is not None and len(points_mm) >= 3:        # SiPhy 부피적분 (동일 probs 사용)
            out.update(shell_mass_integral(points_mm, probs, dens, thick))
        return out


def _to_b64_png(crop_rgb) -> str:
    if isinstance(crop_rgb, (str, os.PathLike)):                 # 파일 경로 허용
        return base64.b64encode(open(crop_rgb, "rb").read()).decode()
    from PIL import Image
    arr = np.asarray(crop_rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
