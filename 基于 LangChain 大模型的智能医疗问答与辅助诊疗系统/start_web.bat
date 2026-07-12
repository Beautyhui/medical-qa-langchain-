@echo off
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" -m streamlit run app/web/streamlit_app.py
pause
