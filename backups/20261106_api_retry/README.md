# Backup — API outage classification + automatic retry (2026-11-06)

New: src/tuj/common/api_retry.py, tests/test_api_retry.py
Changed: m2_subgoal/rough.py, m1_scene/siphy_backend.py, m5_motion/gemini_provider.py,
         m5_motion/vlm_provider.py, m5_motion/failure_taxonomy.py, scripts/run.py,
         scripts/make_table3.py, scripts/run_ablation_batch.py
Undo: copy each file here back to its path (name = path with '/' -> '_'),
      delete src/tuj/common/api_retry.py and tests/test_api_retry.py.
Disable retries without undo: $env:TUJ_API_RETRIES = "0"; run.py --api-reruns 0
