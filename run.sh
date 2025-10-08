#!/bin/bash
echo 'Activando entorno virtual...'
source venv/bin/activate
echo
echo 'Python version:'
python --version
echo
echo 'Iniciando aplicación...'
python app.py