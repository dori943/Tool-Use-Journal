@echo off
cd /d "C:\Users\SAMSUNG\Downloads\EE\Tool-Use-Journal"
if /I "%~1"=="/resume" (
  ".venv\Scripts\python.exe" "experiments\c4_static_real5\run_locked_inference.py" --manifest "output\c4_table3\20260914T062054Z_static_real5_prep\selected_static_inputs.csv" --config "output\c4_table3\20260914T062054Z_static_real5_prep\inference_config.yaml" --resume "output\c4_table3\20260914T064753Z_static_real5_inference"
  if errorlevel 1 exit /b 1
  exit /b 0
)
if not defined GEMINI_API_KEY (echo Set GEMINI_API_KEY in this shell first.& exit /b 2)
".venv\Scripts\python.exe" "experiments\c4_static_real5\run_locked_inference.py" --manifest "output\c4_table3\20260914T062054Z_static_real5_prep\selected_static_inputs.csv" --config "output\c4_table3\20260914T062054Z_static_real5_prep\inference_config.yaml"
