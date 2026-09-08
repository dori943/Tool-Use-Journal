"""[호환 shim] m3_grounding 은 m1_scene 으로 통합됐다 (M1·M3 통합).

새 코드는 tuj.m1_scene 을 import 할 것.
  intrinsic      → m1_scene.grounding
  ee_conditioned → m1_scene.ee_rules
  relational     → m1_scene.relations
  memory         → m1_scene.memory
  siphy_backend  → m1_scene.siphy_backend

`from tuj.m3_grounding import PropertyMemory` 같은 옛 코드를 위해 이름들을 재수출한다.
서브모듈(`tuj.m3_grounding.siphy_backend` 등)을 직접 import 하는 경로에서는 이 파일이
실행되지만, __getattr__ 로 lazy 해석해 순환 import 를 피한다.
(이행 안내: docs/m1m3_merge_m2_guide.md)
"""
import warnings as _w

_w.warn("tuj.m3_grounding 은 tuj.m1_scene 으로 통합됐습니다 — 새 코드는 m1_scene 을 import 하세요.",
        DeprecationWarning, stacklevel=2)

_ALIASES = {
    "FrictionHead": ("tuj.m1_scene.grounding", "FrictionHead"),
    "MockBackend":  ("tuj.m1_scene.grounding", "MockBackend"),
    "PropertyBackend": ("tuj.m1_scene.grounding", "PropertyBackend"),
    "ground_intrinsic": ("tuj.m1_scene.grounding", "ground_intrinsic"),
    "pca_dims": ("tuj.m1_scene.grounding", "pca_dims"),
    "surface_rms": ("tuj.m1_scene.grounding", "surface_rms"),
    "SiPhyBackend": ("tuj.m1_scene.siphy_backend", "SiPhyBackend"),
    "shell_mass_integral": ("tuj.m1_scene.siphy_backend", "shell_mass_integral"),
    "evaluate_ee": ("tuj.m1_scene.ee_rules", "evaluate_ee"),
    "grip_slip_margin_fn": ("tuj.m1_scene.ee_rules", "grip_slip_margin_fn"),
    "reach_check": ("tuj.m1_scene.ee_rules", "reach_check"),
    "PropertyMemory": ("tuj.m1_scene.memory", "PropertyMemory"),
    "relational": ("tuj.m1_scene", "relations"),
}
__all__ = list(_ALIASES)


def __getattr__(name):                                # PEP 562 — lazy 재수출
    if name in _ALIASES:
        import importlib
        mod, attr = _ALIASES[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
