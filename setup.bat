@echo off
set LOGFILE=setup_log.txt
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

:: Create virtual environment if not exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
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
echo Activating virtual environment...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. >> %LOGFILE%
    echo Failed to activate virtual environment. Please check the environment and try again. >> %LOGFILE%
    pause
    exit /b
)
echo Virtual environment activated. >> %LOGFILE%

:: Install dependencies
if exist requirements.txt (
    echo Installing dependencies...
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
echo Deactivating virtual environment...
deactivate
echo Virtual environment deactivated. >> %LOGFILE%

echo Project executed successfully. >> %LOGFILE%
pause
