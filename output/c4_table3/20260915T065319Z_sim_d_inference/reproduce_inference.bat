@echo off
setlocal
set PREP=C:\Users\SAMSUNG\Downloads\EE\Tool-Use-Journal\output\c4_table3\20260915T064657Z_sim_d_adapter_audit
.venv\Scripts\python.exe experiments\c4_table3\sim_d\run_inference.py --prep %PREP% --output C:\Users\SAMSUNG\Downloads\EE\Tool-Use-Journal\output\c4_table3\20260915T065319Z_sim_d_inference --repeats 3
