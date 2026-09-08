"""C1-T2 / C2-T2 공통 Kitchen scene base.

RoboCasa ``Kitchen`` 환경을 Layout004 + Style002로 고정한다.
decorative clutter / plant 액세서리는 기본 비활성화한다.
"""

from __future__ import annotations

from copy import deepcopy

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.scene_registry import LayoutType, StyleType


DEFAULT_KITCHEN_LAYOUT_ID = LayoutType.LAYOUT004
DEFAULT_KITCHEN_STYLE_ID = StyleType.STYLE002

# Layout004에서 is_clutter가 아닌 decorative accessory까지 실험 scene에서 제외
DEFAULT_DECOR_DISABLE = {
    "plant": {"enable": False},
    "paper_towel": {"enable": False},
    "knife_block": {"enable": False},
    "utensil_set": {"enable": False},
    "fruit_bowl": {"enable": False},
    "soap_dispenser": {"enable": False},
    "tiered_basket": {"enable": False},
    "jar": {"enable": False},
    "jar_2": {"enable": False},
    "glass_cup": {"enable": False},
    "glass_cup_2": {"enable": False},
    "salt_shaker": {"enable": False},
}


class KitchenBase(Kitchen):
    """RoboCasa Kitchen scene (Layout004, Style002)."""

    def __init__(
        self,
        robots="UR5e",
        layout_ids=None,
        style_ids=None,
        layout_and_style_ids=None,
        clutter_mode=0,
        update_fxtr_cfg_dict=None,
        **kwargs,
    ):
        if layout_and_style_ids is None:
            if layout_ids is None:
                layout_ids = DEFAULT_KITCHEN_LAYOUT_ID
            if style_ids is None:
                style_ids = DEFAULT_KITCHEN_STYLE_ID

        decor_cfg = deepcopy(DEFAULT_DECOR_DISABLE)
        if update_fxtr_cfg_dict:
            decor_cfg.update(update_fxtr_cfg_dict)

        # robosuite 표준 MujocoEnv kwarg 흡수. RoboCasa ``Kitchen.__init__``은 명시
        # 시그니처(**kwargs 없음)라 ``hard_reset``을 받지 못한다. M5 env 팩토리
        # (make_tool_use_journal_env)가 테이블 환경(c1_1 등)과 같은 옵션으로
        # hard_reset=False를 넘기면 TypeError가 나서 Kitchen 기반 태스크
        # (c1_2·c2_2·c4_1·c4_2) 전부가 M5 진입 전에 중단됐다. 리셋 방식은 RoboCasa가
        # 자체 관리하므로 여기서 버린다. (다른 kwarg는 그대로 전달해 오타를 숨기지 않는다.)
        kwargs.pop("hard_reset", None)

        # clutter_mode=0 → is_clutter fixture 비활성 + decor_cfg로 plant 등 추가 비활성
        kwargs.setdefault("use_distractors", False)

        super().__init__(
            robots=robots,
            layout_ids=layout_ids,
            style_ids=style_ids,
            layout_and_style_ids=layout_and_style_ids,
            clutter_mode=clutter_mode,
            update_fxtr_cfg_dict=decor_cfg,
            **kwargs,
        )
