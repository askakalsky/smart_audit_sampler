@echo off
:: Create a log file for error and output tracking
set LOGFILE=setup_log.txt
echo ===== Setup started at %date% %time% ===== >> %LOGFILE%

:: Pull the latest changes from the repository
echo Pulling latest changes from the repository...
echo Pulling latest changes from the repository... >> %LOGFILE%
git pull 2>&1 | tee -a %LOGFILE% || (
    echo Failed to pull the latest changes. See %LOGFILE% for details.
    echo Failed to pull the latest changes. See %LOGFILE% for details. >> %LOGFILE%
    pause
    exit /b
)
echo Successfully pulled the latest changes.
echo Successfully pulled the latest changes. >> %LOGFILE%

:: Check if pyenv is installed
echo Checking if pyenv is installed...
echo Checking if pyenv is installed... >> %LOGFILE%
where pyenv 2>&1 | tee -a %LOGFILE% || (
    echo pyenv not found. Installing pyenv...
    echo pyenv not found. Installing pyenv... >> %LOGFILE%
    git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv 2>&1 | tee -a %LOGFILE% || (
        echo Failed to install pyenv. See %LOGFILE% for details.
        echo Failed to install pyenv. See %LOGFILE% for details. >> %LOGFILE%
        pause
        exit /b
    )
    echo pyenv successfully installed.
    echo pyenv successfully installed. >> %LOGFILE%
)
echo pyenv is already installed.
echo pyenv is already installed. >> %LOGFILE%

:: Update environment variables for the current session (not globally with setx)
echo Updating environment variables...
echo Updating environment variables... >> %LOGFILE%
set "PYENV=%USERPROFILE%\.pyenv\pyenv-win\bin"
set "PATH=%PYENV%;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
echo Environment variables updated for pyenv.
echo Environment variables updated for pyenv. >> %LOGFILE%

:: Check if Python 3.10.11 is installed
echo Checking if Python 3.10.11 is installed...
echo Checking if Python 3.10.11 is installed... >> %LOGFILE%
pyenv versions | findstr "3.10.11" 2>&1 | tee -a %LOGFILE% || (
    echo Python 3.10.11 not found. Installing Python 3.10.11...
    echo Python 3.10.11 not found. Installing Python 3.10.11... >> %LOGFILE%
    pyenv install 3.10.11 2>&1 | tee -a %LOGFILE% || (
        echo Failed to install Python 3.10.11. See %LOGFILE% for details.
        echo Failed to install Python 3.10.11. See %LOGFILE% for details. >> %LOGFILE%
        pause
        exit /b
    )
    echo Python 3.10.11 successfully installed.
    echo Python 3.10.11 successfully installed. >> %LOGFILE%
)
echo Python 3.10.11 is already installed.
echo Python 3.10.11 is already installed. >> %LOGFILE%

:: Set Python 3.10.11 as the local version
echo Setting Python 3.10.11 as the local version...
echo Setting Python 3.10.11 as the local version... >> %LOGFILE%
pyenv local 3.10.11 2>&1 | tee -a %LOGFILE% || (
    echo Failed to set Python 3.10.11 as the local version.
    echo Failed to set Python 3.10.11 as the local version. See %LOGFILE% for details. >> %LOGFILE%
    pause
    exit /b
)
echo Python 3.10.11 set as local version.
echo Python 3.10.11 set as local version. >> %LOGFILE%

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    echo Creating virtual environment... >> %LOGFILE%
    python -m venv .venv 2>&1 | tee -a %LOGFILE% || (
        echo Failed to create virtual environment.
        echo Failed to create virtual environment. See %LOGFILE% for details. >> %LOGFILE%
        pause
        exit /b
    )
    echo Virtual environment created.
    echo Virtual environment created. >> %LOGFILE%
) else (
    echo Virtual environment already exists.
    echo Virtual environment already exists. >> %LOGFILE%
)

:: Activate virtual environment
echo Activating virtual environment...
echo Activating virtual environment... >> %LOGFILE%
call .venv\Scripts\activate 2>&1 | tee -a %LOGFILE% || (
    echo Failed to activate virtual environment.
    echo Failed to activate virtual environment. See %LOGFILE% for details. >> %LOGFILE%
    pause
    exit /b
)
echo Virtual environment activated.
echo Virtual environment activated. >> %LOGFILE%

:: Install dependencies from requirements.txt
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    echo Installing dependencies from requirements.txt... >> %LOGFILE%
    pip install -r requirements.txt 2>&1 | tee -a %LOGFILE% || (
        echo Failed to install dependencies.
        echo Failed to install dependencies. See %LOGFILE% for details. >> %LOGFILE%
        pause
        exit /b
    )
    echo Dependencies installed successfully.
    echo Dependencies installed successfully. >> %LOGFILE%
) else (
    echo requirements.txt not found, skipping dependency installation.
    echo requirements.txt not found, skipping dependency installation. >> %LOGFILE%
)

echo Setup is complete!
echo Setup is complete! >> %LOGFILE%
echo ===== Setup completed at %date% %time% ===== >> %LOGFILE%

:: Run the main project script
echo Running the project...
python main.py

pause