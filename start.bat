@echo off
set LOGFILE=start_log.txt
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
python main.py
if %errorlevel% neq 0 (
    echo Failed to run the project. >> %LOGFILE%
    echo Failed to run the project. Please check the code for errors. >> %LOGFILE%
    pause
    exit /b
)