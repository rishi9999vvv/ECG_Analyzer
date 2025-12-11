@echo off
REM ECG Analyzer - Windows Setup Script

echo =========================================
echo ECG Analyzer - Setup Script (Windows)
echo =========================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Create directory structure
echo Creating directory structure...
if not exist "data\mit-bih" mkdir "data\mit-bih"
if not exist "data\user_inputs" mkdir "data\user_inputs"
if not exist "models" mkdir "models"
if not exist "reports" mkdir "reports"
if not exist "outputs" mkdir "outputs"
if not exist "static\css" mkdir "static\css"
if not exist "static\js" mkdir "static\js"
if not exist "static\images" mkdir "static\images"
if not exist "templates" mkdir "templates"
if not exist "backend" mkdir "backend"
echo.

REM Create __init__.py files
type nul > backend\__init__.py
echo.

echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. (Optional) Download MIT-BIH database and place in data\mit-bih\
echo    Visit: https://physionet.org/content/mitdb/1.0.0/
echo.
echo 3. Train the model:
echo    python backend\main_train_eval.py
echo.
echo 4. Run the application:
echo    python app.py
echo.
echo 5. Open browser to: http://localhost:5000
echo.
echo =========================================
echo.
pause