@echo off
echo Activando entorno virtual...
call venv\Scripts\activate
echo.
echo Python version:
python --version
echo.
echo Iniciando aplicacion...
python app.py
pause