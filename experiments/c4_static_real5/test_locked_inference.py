"""Preflight and checkpoint contracts only; no provider calls or GT reads."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_locked_inference import (append_record, identity_for, load_resume,  # noqa: E402
                                  prediction_key, preflight, prepare_run,
                                  parse_friction_response)


class LockedInferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = ROOT / "output/c4_table3/20260914T062054Z_static_real5_prep"
        cls.manifest = cls.prep / "selected_static_inputs.csv"
        cls.config = cls.prep / "inference_config.yaml"

    def test_locked_preflight_before_api(self):
        rows, cfg, h = preflight(self.manifest, self.config)
        self.assertEqual(len(rows), 50)
        self.assertEqual(h["mass_warnings"], 13)
        self.assertEqual(h["friction_warnings"], 2)
        self.assertEqual(h["smoke_predictions_imported"], 0)
        with tempfile.TemporaryDirectory() as d:
            changed = Path(d) / "changed.yaml"
            changed.write_bytes(self.config.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "LOCKED_INPUT_MISMATCH"):
                preflight(self.manifest, changed)

    def test_friction_parser_statuses(self):
        self.assertEqual(parse_friction_response('{"combined_friction_coefficient":0}')['value'], 0)
        for raw, expected in [('{"x":1}', 'MISSING_VALUE'),
                              ('{"combined_friction_coefficient":-1}', 'OUT_OF_RANGE'),
                              ('{"combined_friction_coefficient":0.2,"unit":"kg"}', 'UNIT_AMBIGUOUS'),
                              ('{bad', 'INVALID_FORMAT')]:
            with self.assertRaises(Exception) as caught:
                parse_friction_response(raw)
            self.assertEqual(caught.exception.status, expected)

    def test_checkpoint_reuses_identical_key_and_rejects_changed_snapshot(self):
        _, cfg, hashes = preflight(self.manifest, self.config)
        identity = identity_for(cfg, hashes)
        key = prediction_key('siphy_adopted', 'Mass_MnRE', 1, '000000_000010',
                             hashes['manifest_sha256'], cfg['mass_prompt_sha256'], cfg['model'])
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'run'
            prepare_run(run, self.manifest, self.config, identity)
            row = {'prediction_id': hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                   'condition': key[0], 'metric': key[1], 'repeat_id': key[2], 'input_id': key[3],
                   'manifest_hash': key[4], 'prompt_hash': key[5], 'model_version': key[6],
                   'requested_seed': 0, 'provider_seed_supported': False,
                   'provider_seed_sent': False, 'api_request_id': None, 'retry_count': 0,
                   'timestamp_utc': 'test', 'inference_time_s': 0.0,
                   'parsing_status': 'MODEL_FAILURE', 'provider_attempts': [],
                   'failure_reason': 'test-only checkpoint record'}
            append_record(run, row)
            restored = load_resume(run, identity, {key})
            self.assertEqual(set(restored), {key})
            (run / 'config_snapshot.yaml').write_bytes(self.config.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'RESUME_CONFIG_MISMATCH'):
                load_resume(run, identity, {key})


if __name__ == '__main__':
    unittest.main()
