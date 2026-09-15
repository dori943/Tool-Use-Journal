"""Focused dimensional and GT-blind R-panel checks (no model predictions)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/c4_table3")]
from tuj.m1_scene.grounding import FrictionHead  # noqa: E402
from tuj.m1_scene.perception import points_from_frame  # noqa: E402
from tuj.m1_scene.siphy_backend import MAX_TRIES, SiPhyBackend  # noqa: E402
from reselect_friction_windows import select_window  # noqa: E402


class RSemanticsTests(unittest.TestCase):
    def test_probed_coefficient_from_metric_deceleration(self):
        dt = 1 / 30
        t = np.arange(20) * dt
        s_m = 1.0 * t - 0.5 * 0.2 * 9.81 * t ** 2
        track_mm = np.column_stack((s_m * 1000, np.zeros_like(s_m)))
        self.assertAlmostEqual(FrictionHead().probe_mu_from_track(track_mm, dt), 0.2)
        self.assertAlmostEqual(FrictionHead().probe_mu_from_track(track_mm, dt / 2), 0.8)

    def test_positive_acceleration_is_not_friction_deceleration(self):
        dt = 1 / 30
        t = np.arange(20) * dt
        s_m = t + 0.5 * 0.2 * 9.81 * t ** 2
        track_mm = np.column_stack((s_m * 1000, np.zeros_like(s_m)))
        self.assertIsNone(FrictionHead().probe_mu_from_track(track_mm, dt))

    def test_depth_projection_outputs_mm(self):
        depth_m = np.ones((5, 5), dtype=float)
        mask = np.ones((5, 5), dtype=np.uint8)
        K = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], dtype=float)
        objects = points_from_frame(depth_m, mask, K, np.eye(4),
                                    {1: ("object", "object")})
        self.assertEqual(len(objects), 1)
        self.assertTrue(np.any(np.all(np.isclose(objects[0]["points"], [20, 20, 1000]), axis=1)))

    def test_recover_early_window_before_late_tracking_jump(self):
        dt = 1 / 30
        t = np.arange(14) * dt
        early = np.column_stack(((1.5 * t - .5 * .2 * 9.81 * t * t) * 1000,
                                 np.zeros_like(t)))
        late = early[-1] + np.array([500., 0.]) + np.column_stack((np.arange(10) * 4.,
                                                                   np.zeros(10)))
        track = np.vstack((early, late))
        policy = {"max_step_mm": 80, "minimum_intervals": 8,
                  "minimum_net_travel_mm": 50, "stop_step_mm": 3,
                  "stop_consecutive_steps": 3, "dt_s": dt}
        start, end, _ = select_window(track, policy)
        self.assertEqual((start, end), (0, 13))

    def test_failed_vlm_replies_remain_raw_and_gemini_seed_is_omitted(self):
        class FakeCompletions:
            calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                choice = SimpleNamespace(message=SimpleNamespace(content="invalid JSON response"),
                                         finish_reason="stop")
                return SimpleNamespace(choices=[choice], usage=None)

        completions = FakeCompletions()
        fake_client = SimpleNamespace(base_url="https://generativelanguage.googleapis.com/",
                                      chat=SimpleNamespace(completions=completions))
        backend = SiPhyBackend(model="gemini-test", client=fake_client, seed=100,
                               temperature=0)
        with self.assertRaises(RuntimeError):
            backend._propose(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(len(backend.last_vlm_attempts), MAX_TRIES)
        self.assertTrue(all(a["raw_response"] == "invalid JSON response"
                            for a in backend.last_vlm_attempts))
        self.assertTrue(all(a["parsing_status"] == "PARSE_FAILED"
                            for a in backend.last_vlm_attempts))
        self.assertTrue(all("seed" not in call for call in completions.calls))


if __name__ == "__main__":
    unittest.main()
