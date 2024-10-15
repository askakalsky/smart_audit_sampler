@echo off
:: Set log directory and file
set LOGDIR=logs
set LOGFILE=%LOGDIR%\update_log.txt

:: Create logs directory if it doesn't exist
if not exist "%LOGDIR%" (
    mkdir "%LOGDIR%"
)

:: Log start time
echo ===== Update started at %date% %time% ===== >> %LOGFILE%

:: Stash changes in specific files
echo Stashing changes in specific files... >> %LOGFILE%
git stash push start.bat setup.bat update.bat >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo Failed to stash changes. >> %LOGFILE%
    echo Failed to stash changes. Please check the stashing process. >> %LOGFILE%
    pause
    exit /b
)
echo Changes successfully stashed. >> %LOGFILE%

:: Pull updates from the repository
echo Pulling updates from the repository... >> %LOGFILE%
git pull >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo Failed to pull updates. >> %LOGFILE%
    echo Failed to pull updates. Please check your network connection and repository settings. >> %LOGFILE%
    pause
    exit /b
)
echo Updates pulled successfully. >> %LOGFILE%

:: Reapply stashed changes
echo Reapplying stashed changes... >> %LOGFILE%
git stash pop >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo Failed to reapply stashed changes. >> %LOGFILE%
    echo Failed to reapply stashed changes. Please resolve any conflicts manually. >> %LOGFILE%
    pause
    exit /b
)
echo Stashed changes reapplied successfully. >> %LOGFILE%

:: Check if requirements.txt has changed
echo Checking if requirements.txt has changed... >> %LOGFILE%
git diff --exit-code HEAD^ HEAD -- requirements.txt >nul
if %errorlevel% neq 0 (
    echo requirements.txt has changed. Updating dependencies... >> %LOGFILE%
    :: Activate virtual environment
    echo Activating virtual environment... >> %LOGFILE%
    call .venv\Scripts\activate
    if %errorlevel% neq 0 (
        echo Failed to activate virtual environment. >> %LOGFILE%
        echo Failed to activate virtual environment. Please check the environment and try again. >> %LOGFILE%
        pause
        exit /b
    )
    echo Virtual environment activated. >> %LOGFILE%

    :: Install dependencies
    echo Installing dependencies... >> %LOGFILE%
    pip install -r requirements.txt >> %LOGFILE% 2>&1
    if %errorlevel% neq 0 (
        echo Failed to install dependencies. >> %LOGFILE%
        echo Failed to install dependencies. Please check the pip output for errors. >> %LOGFILE%
        pause
        exit /b
    )
    echo Dependencies installed successfully. >> %LOGFILE%
) else (
    echo requirements.txt has not changed. No dependency update required. >> %LOGFILE%
)

:: Output success message
echo Project updated successfully, excluding specified files. >> %LOGFILE%
pause
