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
