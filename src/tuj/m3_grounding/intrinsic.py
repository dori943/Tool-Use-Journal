"""[호환 shim] tuj.m3_grounding.intrinsic 은 tuj.m1_scene.grounding 으로 이동했다."""
from tuj.m1_scene.grounding import *  # noqa: F401,F403
from tuj.m1_scene.grounding import (FrictionHead, MockBackend, PropertyBackend,  # noqa: F401
                                    GEOMETRY_REQUIRED, apply_memory_hit_to_observation,
                                    geometry_from_node, geometry_is_current, ground_intrinsic,
                                    pca_dims, surface_rms, seal_patch_rms_mm)
