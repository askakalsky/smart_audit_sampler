@echo off
:: Set log directory and file
set LOGDIR=logs
set LOGFILE=%LOGDIR%\setup_log.txt

:: Create logs directory if it doesn't exist
if not exist "%LOGDIR%" (
    mkdir "%LOGDIR%"
)

:: Log start time
echo ===== Setup started at %date% %time% ===== >> %LOGFILE%

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python from https://www.python.org/.
    echo Python is not installed. >> %LOGFILE%
    pause
    exit /b
)

echo Python is installed. >> %LOGFILE%

:: Create virtual environment if it does not exist
if not exist ".venv" (
    echo Creating virtual environment... >> %LOGFILE%
    python -m venv .venv >> %LOGFILE% 2>&1
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. >> %LOGFILE%
        echo Failed to create virtual environment. Please ensure that Python is correctly installed and try again.
        pause
        exit /b
    )
    echo Virtual environment created. >> %LOGFILE%
) else (
    echo Virtual environment already exists. >> %LOGFILE%
)

:: Activate the virtual environment
echo Activating virtual environment... >> %LOGFILE%
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. >> %LOGFILE%
    echo Failed to activate virtual environment. Please check the environment and try again. >> %LOGFILE%
    pause
    exit /b
)
echo Virtual environment activated. >> %LOGFILE%

:: Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo Installing dependencies... >> %LOGFILE%
    pip install --upgrade pip >> %LOGFILE% 2>&1
    pip install -r requirements.txt >> %LOGFILE% 2>&1
    if %errorlevel% neq 0 (
        echo Failed to install dependencies. >> %LOGFILE%
        echo Failed to install dependencies. See %LOGFILE% for details. >> %LOGFILE%
        pause
        exit /b
    )
    echo Dependencies installed successfully. >> %LOGFILE%
) else (
    echo requirements.txt not found, skipping dependency installation. >> %LOGFILE%
)

:: Deactivate the virtual environment
echo Deactivating virtual environment... >> %LOGFILE%
deactivate
echo Virtual environment deactivated. >> %LOGFILE%

:: Final success message
echo Project executed successfully. >> %LOGFILE%
pause
