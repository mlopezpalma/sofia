#!/usr/bin/env python3
"""
Script de instalación compatible con Python 3.7+
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Colores para la terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def check_python_version():
    """Verificar versión de Python - MODIFICADO PARA 3.7+"""
    print_info("Verificando versión de Python...")
    
    version = sys.version_info
    
    # Cambiar requisito a Python 3.7
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python 3.7+ requerido. Versión actual: {version.major}.{version.minor}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detectado")
    
    # Advertencia para Python 3.7
    if version.major == 3 and version.minor == 7:
        print_warning("Python 3.7 detectado. Recomendado: Python 3.8+ para mejor rendimiento")
        print_info("El proyecto debería funcionar, pero considera actualizar Python si encuentras problemas")
        
        response = input(f"\n{Colors.WARNING}¿Continuar con Python 3.7? (s/n): {Colors.ENDC}")
        if response.lower() != 's':
            return False
    
    return True

def create_virtual_environment():
    """Crear entorno virtual con el Python actual"""
    print_header("CREANDO ENTORNO VIRTUAL")
    
    venv_name = "venv"
    
    print_info(f"Usando Python {sys.version_info.major}.{sys.version_info.minor} para el entorno virtual")
    
    # Verificar si ya existe
    if os.path.exists(venv_name):
        response = input(f"{Colors.WARNING}El entorno virtual '{venv_name}' ya existe. ¿Eliminar y recrear? (s/n): {Colors.ENDC}")
        if response.lower() == 's':
            shutil.rmtree(venv_name)
            print_success("Entorno virtual anterior eliminado")
        else:
            print_info("Usando entorno virtual existente")
            return venv_name
    
    # Crear nuevo entorno virtual
    print_info("Creando entorno virtual...")
    result = subprocess.run([sys.executable, "-m", "venv", venv_name], capture_output=True)
    
    if result.returncode == 0:
        print_success(f"Entorno virtual '{venv_name}' creado con Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # Instrucciones de activación
        if platform.system() == "Windows":
            activate_cmd = f"{venv_name}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_name}/bin/activate"
        
        print_info(f"Para activar: {activate_cmd}")
        return venv_name
    else:
        print_error("No se pudo crear el entorno virtual")
        return None

def get_pip_command(venv_name):
    """Obtener comando pip según el OS"""
    if platform.system() == "Windows":
        return os.path.join(venv_name, "Scripts", "pip")
    else:
        return os.path.join(venv_name, "bin", "pip")

def install_dependencies(venv_name):
    """Instalar dependencias compatibles con Python 3.7"""
    print_header("INSTALANDO DEPENDENCIAS")
    
    pip_cmd = get_pip_command(venv_name)
    
    # Actualizar pip
    print_info("Actualizando pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], capture_output=True)
    
    # Crear requirements.txt compatible con Python 3.7
    if not os.path.exists("requirements.txt"):
        print_info("Creando requirements.txt compatible con Python 3.7...")
        
        # Versiones compatibles con Python 3.7
        requirements_content = """# Compatibles con Python 3.7+
flask==2.2.5
flask-cors==4.0.0
requests==2.28.2
chromadb==0.4.22
sentence-transformers==2.2.2
python-dotenv==0.21.1
werkzeug==2.2.3"""
        
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print_success("requirements.txt creado (versiones compatibles con Python 3.7)")
    
    # Instalar desde requirements.txt
    print_info("Instalando paquetes desde requirements.txt...")
    print_info("Esto puede tomar varios minutos en la primera instalación...")
    
    result = subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print_success("Todas las dependencias instaladas")
    else:
        print_warning("Algunas dependencias pueden necesitar ajustes")
        # Intentar instalar una por una
        print_info("Intentando instalación individual de paquetes...")
        packages = [
            "flask==2.2.5",
            "flask-cors==4.0.0", 
            "requests==2.28.2",
            "python-dotenv==0.21.1",
            "werkzeug==2.2.3"
        ]
        
        for package in packages:
            print(f"  Instalando {package}...")
            subprocess.run([pip_cmd, "install", package], capture_output=True)
        
        # ChromaDB y sentence-transformers por separado (son más pesados)
        print_info("Instalando ChromaDB (puede tardar)...")
        subprocess.run([pip_cmd, "install", "chromadb==0.4.22"], capture_output=True)
        
        print_info("Instalando sentence-transformers (puede tardar varios minutos)...")
        subprocess.run([pip_cmd, "install", "sentence-transformers==2.2.2"], capture_output=True)

def create_config_files():
    """Crear archivos de configuración"""
    print_header("CREANDO ARCHIVOS DE CONFIGURACIÓN")
    
    # Crear .env si no existe
    if not os.path.exists(".env"):
        env_content = """# ============================================
# VALORES SECRETOS (NO COMPARTIR)
# ============================================
OPENROUTER_API_KEY=tu-api-key-aqui

# ============================================
# CONFIGURACIÓN
# ============================================
FLASK_PORT=5000
FLASK_DEBUG=False
SECRET_KEY=dev-secret-key-cambiar-en-produccion

# ============================================
# LÍMITES PERSONALIZADOS
# ============================================
MAX_TOKENS=2000
TEMPERATURE=0.7
MAX_CONTEXT=5"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        print_success(".env creado (configura tu API key)")
    else:
        print_info(".env ya existe")
    
    # Crear .gitignore
    gitignore_content = """# Entorno virtual
venv/
env/

# Configuración local
.env

# Bases de datos
*.db
chroma_db/

# Python
__pycache__/
*.pyc

# Logs
*.log

# IDEs
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Templates generados
templates/index.html"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print_success(".gitignore creado")

def create_run_scripts():
    """Crear scripts de ejecución"""
    print_header("CREANDO SCRIPTS DE EJECUCIÓN")
    
    # Script para Windows
    bat_content = """@echo off
echo Activando entorno virtual...
call venv\\Scripts\\activate
echo.
echo Python version:
python --version
echo.
echo Iniciando aplicacion...
python app.py
pause"""
    
    with open("run.bat", "w") as f:
        f.write(bat_content)
    
    # Script para Linux/Mac
    sh_content = """#!/bin/bash
echo 'Activando entorno virtual...'
source venv/bin/activate
echo
echo 'Python version:'
python --version
echo
echo 'Iniciando aplicación...'
python app.py"""
    
    with open("run.sh", "w") as f:
        f.write(sh_content)
    
    # Hacer ejecutable el script de Linux/Mac
    if platform.system() != "Windows":
        os.chmod("run.sh", 0o755)
    
    print_success("Scripts de ejecución creados (run.bat / run.sh)")

def print_final_instructions():
    """Mostrar instrucciones finales"""
    print_header("🎉 INSTALACIÓN COMPLETA")
    
    print_success(f"Entorno virtual creado con Python {sys.version_info.major}.{sys.version_info.minor}")
    print("")
    
    if platform.system() == "Windows":
        print("📋 PRÓXIMOS PASOS:")
        print("")
        print("1. Configurar API Key:")
        print(f"   Edita .env y agrega tu API key de OpenRouter")
        print("")
        print("2. Ejecutar la aplicación:")
        print(f"   {Colors.OKBLUE}run.bat{Colors.ENDC}")
    else:
        print("📋 PRÓXIMOS PASOS:")
        print("")
        print("1. Configurar API Key:")
        print(f"   Edita .env y agrega tu API key de OpenRouter")
        print("")
        print("2. Ejecutar la aplicación:")
        print(f"   {Colors.OKBLUE}./run.sh{Colors.ENDC}")
    
    print("")
    print("3. Abrir en navegador:")
    print(f"   {Colors.OKBLUE}http://localhost:5000{Colors.ENDC}")
    print("")
    
    if sys.version_info.minor == 7:
        print_warning("Nota: Estás usando Python 3.7")
        print_warning("Si encuentras problemas, considera actualizar a Python 3.8+")

def main():
    """Función principal"""
    print_header("🧠 INSTALADOR - CHAT MULTI-MODELO CON MEMORIA VECTORIAL")
    
    # Verificar Python (ahora acepta 3.7+)
    if not check_python_version():
        print_error("Instalación cancelada")
        print_info("Para actualizar Python, visita: https://www.python.org/downloads/")
        sys.exit(1)
    
    # Crear entorno virtual
    venv_name = create_virtual_environment()
    if not venv_name:
        sys.exit(1)
    
    # Instalar dependencias
    install_dependencies(venv_name)
    
    # Crear archivos de configuración
    create_config_files()
    
    # Crear scripts de ejecución
    create_run_scripts()
    
    # Instrucciones finales
    print_final_instructions()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Instalación cancelada")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        sys.exit(1)