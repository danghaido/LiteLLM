@echo off
chcp 65001 >nul 2>&1
title LiteLLM — Embed PDF to Chroma

:: ───────────────────────────────────────────────────────────────────────────
::  LiteLLM Embed PDF  (song song, không ảnh hưởng luồng chính)
::
::  Cách dùng:
::      embed_pdf.bat paper.pdf
::      embed_pdf.bat paper.pdf --chunk-size 512 --overlap 64
::
::  Tuỳ chọn:
::      %1            — Đường dẫn file PDF (bắt buộc)
::      %2 %3 %4 %5   — Các flag tùy chọn truyền thêm cho embed_pdf.py
:: ───────────────────────────────────────────────────────────────────────────

setlocal

set "PROJECT_ROOT=%~dp0..\"
set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

:: Kiểm tra Python trong venv
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
) else (
    :: Fallback: python từ PATH
    set "PYTHON=python"
)

:: Kiểm tra đối số
if "%~1"=="" (
    echo [ERROR] Chua truyen duong dan file PDF.
    echo.
    echo  Cu phap:
    echo    embed_pdf.bat ^<path_to_pdf^> [--chunk-size N] [--overlap N]
    echo.
    echo  Vi du:
    echo    embed_pdf.bat paper.pdf
    echo    embed_pdf.bat paper.pdf --chunk-size 512 --overlap 64
    echo.
    pause
    exit /b 1
)

:: Kiểm tra file tồn tại
if not exist "%~f1" (
    echo [ERROR] File khong ton tai: %~f1
    pause
    exit /b 1
)

:: Chạy script Python, chuyển toàn bộ đối số qua
"%PYTHON%" "%PROJECT_ROOT%embed_pdf.py" %*

:: Giữ cửa sổ nếu có lỗi
if errorlevel 1 (
    echo.
    echo ========================================
    echo  Da xay ra loi. Kiem tra log phia tren.
    echo ========================================
    pause
)

endlocal
