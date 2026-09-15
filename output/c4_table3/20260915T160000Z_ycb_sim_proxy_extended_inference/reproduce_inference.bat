@echo off
setlocal
set PREP=output\c4_table3\20260915T133500Z_ycb_sim_proxy_prep
.venv\Scripts\python.exe experiments\c4_table3\sim_d\run_inference.py --prep %PREP% --output output\c4_table3\20260915T160000Z_ycb_sim_proxy_extended_inference --repeats 3 --conditions name_only affordance_labels siphy_adopted geometric_grounding ours_full --reuse-predictions output\c4_table3\_tmp_ycb\reuse_raw.jsonl
