@echo off
:: Set log directory and file
set LOGDIR=logs
set LOGFILE=%LOGDIR%\start_log.txt

:: Create logs directory if it doesn't exist
if not exist "%LOGDIR%" (
    mkdir "%LOGDIR%"
)

:: Log start time
echo ===== Setup started at %date% %time% ===== >> %LOGFILE%

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

:: Run the main script
echo Running the project... >> %LOGFILE%
python main.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo Failed to run the project. >> %LOGFILE%
    echo Failed to run the project. Please check the code for errors. >> %LOGFILE%
    pause
    exit /b
)

:: Log success message
echo Project ran successfully. >> %LOGFILE%
pause
