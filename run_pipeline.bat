@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at .venv
  echo Create it first:
  echo   py -3.12 -m venv .venv
  exit /b 1
)

echo [1/7] Download encoder...
call .venv\Scripts\python.exe -m src.data.download_encoder
if errorlevel 1 exit /b 1

echo [2/7] Prepare corpus...
call .venv\Scripts\python.exe -m src.data.prepare_corpus --input data/raw/sample_en_zanx_pairs.tsv
if errorlevel 1 exit /b 1

echo [3/7] Build zanX tokenizer...
call .venv\Scripts\python.exe -m src.data.build_zanx_tokenizer --input data/processed/train.jsonl
if errorlevel 1 exit /b 1

echo [4/7] Train model...
call .venv\Scripts\python.exe -m src.train.train --config configs/dev.yaml
if errorlevel 1 exit /b 1

echo [5/7] Evaluate model...
call .venv\Scripts\python.exe -m src.train.evaluate --config configs/dev.yaml --split test
if errorlevel 1 exit /b 1

echo [6/7] Export ONNX...
call .venv\Scripts\python.exe -m src.export.export_onnx --config configs/dev.yaml
if errorlevel 1 exit /b 1

echo [7/7] Run inference smoke test...
call .venv\Scripts\python.exe -m src.infer.onnx_runtime_infer --bundle artifacts/releases/v2 --text "Hello world"
if errorlevel 1 exit /b 1

echo.
echo [DONE] Pipeline completed successfully.
exit /b 0
