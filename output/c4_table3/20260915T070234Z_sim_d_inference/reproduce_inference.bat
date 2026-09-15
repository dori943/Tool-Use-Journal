@echo off
setlocal
set PREP=output/c4_table3/20260915T064657Z_sim_d_adapter_audit
.venv\Scripts\python.exe experiments\c4_table3\sim_d\run_inference.py --prep %PREP% --output output\c4_table3\20260915T070234Z_sim_d_inference --repeats 3
