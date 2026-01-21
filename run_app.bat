@echo off
echo ========================================
echo 🛡️ Safety Copilot - Starting App
echo ========================================
echo.

REM Check if .env exists
if not exist .env (
    echo ⚠️  Warning: .env file not found
    echo    Create .env file with your API keys
    echo.
)

REM Run Streamlit
echo 🚀 Starting Streamlit app...
echo.
streamlit run streamlit_app.py

pause

