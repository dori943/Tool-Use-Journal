import mujoco
import numpy as np
from scripts.run_m1 import _native_segmentation_ids


def test_geom_and_flex_same_numeric_id_do_not_alias_or_become_background():
    raw = np.array([[[int(mujoco.mjtObj.mjOBJ_GEOM), 0],
                     [int(mujoco.mjtObj.mjOBJ_FLEX), 0], [-1, -1]]])
    np.testing.assert_array_equal(_native_segmentation_ids(raw), [[0, -2, -1]])
