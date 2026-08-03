@echo off
REM Quick deployment script for TORMENTED-BERT v2 (Windows)

setlocal enabledelayedexpansion

echo ==========================================
echo TORMENTED-BERT v2 Deployment Quick Start
echo ==========================================

REM Check if checkpoint path provided
if "%~1"=="" (
    echo.
    echo Usage: %~0 ^<checkpoint_path^> [output_dir]
    echo.
    echo Example:
    echo   %~0 checkpoint.pt deployed_model
    echo.
    echo Or run the example:
    echo   python example.py
    exit /b 1
)

set CHECKPOINT=%~1
set OUTPUT=%~2
if "%OUTPUT%"=="" set OUTPUT=deployed_model

echo.
echo Checkpoint: %CHECKPOINT%
echo Output: %OUTPUT%
echo.

REM Deploy
echo [1/3] Deploying model with quantization...
python deploy.py ^
    --checkpoint "%CHECKPOINT%" ^
    --output "%OUTPUT%" ^
    --format quantized ^
    --validate

if errorlevel 1 (
    echo Deployment failed!
    exit /b 1
)

echo.
echo [2/3] Testing inference...
python inference.py ^
    --model "%OUTPUT%" ^
    --text "Prueba de inferencia" ^
    --device cuda

echo.
echo [3/3] Running benchmark...
python inference.py ^
    --model "%OUTPUT%" ^
    --benchmark

echo.
echo ==========================================
echo * Deployment complete!
echo ==========================================
echo.
echo Your model is ready at: %OUTPUT%
echo.
echo To use in interactive mode:
echo   python inference.py --model %OUTPUT%
echo.
