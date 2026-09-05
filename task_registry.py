# -*- coding: utf-8 -*-
"""태스크 레지스트리 — 단일 출처(single source of truth).

task id(c1_1, c2_1, ...) <-> robosuite 환경 클래스의 매핑을 여기 한 곳에만 둔다.
robosuite 를 import 하지 않는 가벼운 모듈이라 어디서든 부담 없이 읽을 수 있다.
(환경 클래스의 실제 로드/등록은 environments/__init__.py 가 이 표를 보고 수행한다.)

새 태스크 추가:
  1) environments/<module>.py 에 씬 클래스 작성
  2) 아래 TASKS 에 한 줄 등록 (id -> module/class, RoboCasa 의존이면 robocasa=True)
그러면 environments 등록, run_m1(씬 추상화), run_m5(모션), run.py 가 전부
이 표를 참조하므로 다른 파일을 손댈 필요가 없다.

M1 씬 추상화는 표준 이름 규칙(로봇 부속 제외 / rack·zone / 끝 _숫자 제거)을 따르는
env 면 범용 어댑터로 자동 처리한다. 이름 규칙이 깨진 env 만 run_m1._OVERRIDES 에
인식 규칙을 등록한다 — 나머지는 여기 한 줄이면 끝이다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskEnv:
    module: str            # environments 하위 모듈 파일명(stem)
    cls: str               # robosuite 환경 클래스명 (robosuite 는 클래스명으로 등록)
    robocasa: bool = False  # RoboCasa(KitchenBase) 의존 -> soft-import 대상
    instruction: str = ""  # M2 서브골 분해에 넣는 자연어 지시문 (한국어 한 문장)


TASKS = {
    "c1_1": TaskEnv("c1_1_lego_sweep",              "C1_1_LegoSweep",
                    instruction="도구를 골라 흩어진 레고 블록을 수집 구역으로 쓸어 담아라"),
    "c2_1": TaskEnv("c2_1_object_sorting",          "C2_1_ObjectSorting",
                    instruction="테이블 위 물체를 왼쪽부터 순서대로 파란 트레이로 옮겨라"),
    "c1_2": TaskEnv("c1_2_dough_flatten",           "C1_2_DoughFlatten",           robocasa=True,
                    instruction="도마 위의 반죽을 평평하게 펴라"),
    "c2_2": TaskEnv("c2_2_sandwich_assembly",       "C2_2_SandwichAssembly",       robocasa=True,
                    instruction="재료를 순서대로 쌓아 샌드위치를 만들어라"),
    "c4_1": TaskEnv("c4_1_interval_fit_extraction", "C4_1_IntervalFitExtraction",  robocasa=True,
                    instruction="가전 사이 틈에 떨어진 카드를 꺼내라"),
    "c4_2": TaskEnv("c4_2_diagonal_fit_packing",    "C4_2_DiagonalFitPacking",     robocasa=True,
                    instruction="긴 물건들을 상자에 담고 뚜껑을 덮어라"),
}

# 편의 뷰: task id -> 환경 클래스명 (러너들이 참조).
TASK_ENVS = {tid: t.cls for tid, t in TASKS.items()}


def env_name(task_id: str) -> str:
    """task id 에 대응하는 robosuite 환경 클래스명. 미등록이면 KeyError."""
    return TASKS[task_id].cls


def instruction(task_id: str) -> str:
    """task id 에 대응하는 자연어 지시문. 미등록/미기재면 KeyError.

    M2 는 이 문장 하나로 서브골을 분해한다. 없는 채로 조용히 다른 태스크의
    문장을 쓰면 장면에 없는 물체(예: 수거함)를 만들어내 검문에서 실패하므로,
    비어 있으면 즉시 KeyError 로 알린다.
    """
    if task_id not in TASKS:
        raise KeyError(f"'{task_id}' 는 task_registry 에 등록되지 않은 태스크다")
    text = TASKS[task_id].instruction
    if not text:
        raise KeyError(f"'{task_id}' 의 instruction 이 비어 있다")
    return text
