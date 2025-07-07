@echo off
echo 🚀 Iniciando Clasificador de Tweets BERT...
echo.
cd /d "%~dp0"
venv\Scripts\python.exe -m streamlit run app\app_classifier.py --server.port 8501 --server.address localhost
pause
