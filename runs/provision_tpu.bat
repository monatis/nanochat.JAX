@echo off
REM =============================================================================
REM Provision a TPU VM via flex-start and run the nanochat smoke test (Windows)
REM
REM This script runs from YOUR WINDOWS MACHINE (not the TPU VM).
REM It provisions a TPU, SSHes in, clones the repo, and kicks off the smoke test.
REM
REM Prerequisites:
REM   1. gcloud CLI installed: winget install Google.CloudSDK
REM   2. Authenticated: gcloud auth login
REM   3. Project set: gcloud config set project YOUR_PROJECT_ID
REM   4. TPU API enabled: gcloud services enable tpu.googleapis.com
REM   5. Alpha components: gcloud components install alpha --quiet
REM
REM Usage:
REM   runs\provision_tpu.bat
REM
REM   With custom settings (set before calling):
REM   set TPU_TYPE=v5e-4
REM   set ZONE=us-west4-a
REM   runs\provision_tpu.bat
REM =============================================================================

setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------------
REM Configuration (override by setting environment variables before running)
REM ---------------------------------------------------------------------------

if not defined ZONE         set "ZONE=us-central1-a"
if not defined TPU_TYPE     set "TPU_TYPE=v6e-1"
if not defined RUNTIME      set "RUNTIME=v2-alpha-tpuv6e"
if not defined DURATION     set "DURATION=2h"
if not defined NODE_NAME    set "NODE_NAME=nanochat-smoke-test"
if not defined QUEUE_NAME   set "QUEUE_NAME=nanochat-queue"
if not defined REPO_URL     set "REPO_URL=https://github.com/monatis/nanochat.JAX"
if not defined SKIP_PROVISION set "SKIP_PROVISION=0"
if not defined SKIP_CLEANUP set "SKIP_CLEANUP=0"

REM Auto-detect project ID if not set
if not defined PROJECT_ID (
    for /f "tokens=*" %%i in ('gcloud config get-value project 2^>nul') do set "PROJECT_ID=%%i"
)

if not defined PROJECT_ID (
    echo ERROR: No project ID found. Run: gcloud config set project YOUR_PROJECT_ID
    exit /b 1
)

REM Auto-detect runtime from TPU type if set to "auto"
if "%RUNTIME%"=="auto" (
    echo %TPU_TYPE% | findstr /i "v6e" >nul && set "RUNTIME=v2-alpha-tpuv6e"
    echo %TPU_TYPE% | findstr /i "v5p" >nul && set "RUNTIME=v2-alpha-tpuv5"
    echo %TPU_TYPE% | findstr /i "v5e" >nul && set "RUNTIME=v2-alpha-tpuv5-lite"
)

echo ============================================================
echo  nanochat TPU Provisioning ^& Smoke Test (Windows)
echo ============================================================
echo.
echo   Project     : %PROJECT_ID%
echo   Zone        : %ZONE%
echo   TPU type    : %TPU_TYPE%
echo   Runtime     : %RUNTIME%
echo   Max duration: %DURATION%
echo   Node        : %NODE_NAME%
echo   Queue       : %QUEUE_NAME%
echo   Repo        : %REPO_URL%
echo.

REM ---------------------------------------------------------------------------
REM Step 1: Provision TPU via flex-start queued resource
REM ---------------------------------------------------------------------------

if "%SKIP_PROVISION%"=="1" (
    echo ^>^>^> Step 1: SKIPPED ^(SKIP_PROVISION=1^)
    goto :step2
)

echo ^>^>^> Step 1: Creating TPU queued resource ^(flex-start^)...
echo.

REM Delete any existing queued resource with the same name (ignore errors)
gcloud alpha compute tpus queued-resources delete "%QUEUE_NAME%" ^
    --project="%PROJECT_ID%" ^
    --zone="%ZONE%" ^
    --quiet 2>nul

gcloud alpha compute tpus queued-resources create "%QUEUE_NAME%" ^
    --project="%PROJECT_ID%" ^
    --zone="%ZONE%" ^
    --accelerator-type="%TPU_TYPE%" ^
    --runtime-version="%RUNTIME%" ^
    --node-id="%NODE_NAME%" ^
    --provisioning-model=flex-start ^
    --max-run-duration="%DURATION%" ^
    --valid-until-duration="%DURATION%"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create queued resource. Check your gcloud configuration.
    exit /b 1
)

echo.
echo ^>^>^> Waiting for TPU to be provisioned...
echo     ^(This may take a few minutes depending on capacity^)
echo.

REM Poll until the TPU is ACTIVE
set "MAX_WAIT=600"
set "ELAPSED=0"
set "POLL_INTERVAL=15"

:poll_loop
if %ELAPSED% geq %MAX_WAIT% goto :poll_timeout

for /f "tokens=*" %%s in ('gcloud alpha compute tpus queued-resources describe "%QUEUE_NAME%" --project="%PROJECT_ID%" --zone="%ZONE%" --format="value(state.state)" 2^>nul') do set "STATUS=%%s"

echo     Status: %STATUS% ^(%ELAPSED%s elapsed^)

if "%STATUS%"=="ACTIVE" (
    echo.
    echo     [OK] TPU is ACTIVE!
    goto :step2
)
if "%STATUS%"=="FAILED" (
    echo.
    echo     [FAIL] TPU provisioning failed.
    echo     Check: gcloud alpha compute tpus queued-resources describe %QUEUE_NAME% --zone=%ZONE%
    exit /b 1
)
if "%STATUS%"=="SUSPENDED" (
    echo.
    echo     [FAIL] TPU was suspended.
    exit /b 1
)

REM Wait before polling again
timeout /t %POLL_INTERVAL% /nobreak >nul
set /a "ELAPSED+=POLL_INTERVAL"
goto :poll_loop

:poll_timeout
echo.
echo     [TIMEOUT] Waited %MAX_WAIT%s for TPU. It may still be pending.
echo     Check manually: gcloud alpha compute tpus queued-resources describe %QUEUE_NAME% --zone=%ZONE%
exit /b 1

REM ---------------------------------------------------------------------------
REM Step 2: SSH into TPU VM and run the smoke test
REM ---------------------------------------------------------------------------

:step2
echo.
echo ^>^>^> Step 2: Running smoke test on TPU VM...
echo.

REM Build the remote command as a single string
REM The TPU VM is Linux, so this runs bash commands via SSH
set "REMOTE_CMD=set -euo pipefail && echo '=== TPU VM: Starting nanochat smoke test ===' && echo 'Hostname:' $(hostname) && REPO_DIR=$HOME/nanochat && if [ -d $REPO_DIR/.git ]; then cd $REPO_DIR && git pull origin main || git pull origin master || true; else git clone %REPO_URL% $REPO_DIR && cd $REPO_DIR; fi && echo '=== Running smoke test ===' && bash runs/tpu_smoke_test.sh && echo '=== Smoke test finished ==='"

gcloud compute tpus tpu-vm ssh "%NODE_NAME%" ^
    --project="%PROJECT_ID%" ^
    --zone="%ZONE%" ^
    --command="%REMOTE_CMD%"

set "SSH_EXIT=%errorlevel%"

echo.

REM ---------------------------------------------------------------------------
REM Step 3: Cleanup (optional)
REM ---------------------------------------------------------------------------

if "%SKIP_CLEANUP%"=="1" (
    echo ^>^>^> Step 3: SKIPPED ^(SKIP_CLEANUP=1^)
    echo     TPU is still running. Don't forget to clean up!
    echo     gcloud alpha compute tpus queued-resources delete %QUEUE_NAME% --zone=%ZONE% --quiet
    goto :done
)

echo ^>^>^> Step 3: Cleaning up TPU resources...
gcloud alpha compute tpus queued-resources delete "%QUEUE_NAME%" ^
    --project="%PROJECT_ID%" ^
    --zone="%ZONE%" ^
    --quiet 2>nul
echo     [OK] Queued resource deleted

:done
echo.
echo ============================================================
if %SSH_EXIT% equ 0 (
    echo  [OK] TPU smoke test completed successfully!
) else (
    echo  [FAIL] TPU smoke test FAILED ^(exit code %SSH_EXIT%^)
)
echo ============================================================

endlocal
exit /b %SSH_EXIT%
