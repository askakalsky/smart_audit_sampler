@echo off
:: Pull the latest changes from the repository
echo Pulling latest changes from the repository...
git pull

:: Check if pyenv is installed
where pyenv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo pyenv not found. Installing pyenv...
    git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv
    setx PYENV %USERPROFILE%\.pyenv\pyenv-win\bin
    setx PATH "%PATH%;%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv-win\shims"
)

:: Update environment variables for the current session
set "PYENV=%USERPROFILE%\.pyenv\pyenv-win\bin"
set "PATH=%PATH%;%PYENV%;%USERPROFILE%\.pyenv\pyenv-win\shims"

:: Check if Python 3.10 is installed
pyenv versions | findstr "3.10" >nul
if %ERRORLEVEL% neq 0 (
    echo Python 3.10 not found. Installing Python 3.10...
    pyenv install 3.10.0
)

:: Set Python 3.10 as the local version
echo Setting Python 3.10 as the local version...
pyenv local 3.10.0

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Activate virtual environment
call .venv\Scripts\activate

:: Install dependencies from requirements.txt
if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found, skipping dependency installation...
)

echo Setup is complete!
pause
