@echo off
REM Script để chạy LiteLLM auto_query trên Windows

set FROM_AUTO_QUERY=true

python -m LiteLLM.scripts.auto_query %*
IF %ERRORLEVEL% EQU 0 (
    python -m LiteLLM.scripts.evaluation
)
