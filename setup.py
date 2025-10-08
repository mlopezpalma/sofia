#!/usr/bin/env python3
"""
Script de instalación automática para Chat Multi-Modelo con Memoria Vectorial
Ejecutar: python setup.py
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
    """Verificar versión de Python"""
    print_info("Verificando versión de Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ requerido. Versión actual: {version.major}.{version.minor}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def create_virtual_environment():
    """Crear entorno virtual"""
    print_header("CREANDO ENTORNO VIRTUAL")
    
    venv_name = "venv"
    
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
        print_success(f"Entorno virtual '{venv_name}' creado")
        
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
    """Instalar dependencias"""
    print_header("INSTALANDO DEPENDENCIAS")
    
    pip_cmd = get_pip_command(venv_name)
    
    # Actualizar pip
    print_info("Actualizando pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], capture_output=True)
    
    # Crear requirements.txt si no existe
    if not os.path.exists("requirements.txt"):
        print_info("Creando requirements.txt...")
        requirements_content = """flask==3.0.0
flask-cors==4.0.0
requests==2.31.0
chromadb==0.4.22
sentence-transformers==2.2.2
python-dotenv==1.0.0"""
        
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print_success("requirements.txt creado")
    
    # Instalar desde requirements.txt
    print_info("Instalando paquetes desde requirements.txt...")
    result = subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print_success("Todas las dependencias instaladas")
    else:
        print_warning("Algunas dependencias pueden no haberse instalado correctamente")
        print(result.stderr)

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
# CONFIGURACIÓN ESPECÍFICA DEL ENTORNO
# ============================================
FLASK_PORT=5000
FLASK_DEBUG=False
SECRET_KEY=dev-secret-key-cambiar-en-produccion

# ============================================
# LÍMITES PERSONALIZADOS (OPCIONAL)
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
ENV/

# Configuración local
.env
config.local.py

# Bases de datos
*.db
*.sqlite
*.sqlite3
chroma_db/

# Python
__pycache__/
*.py[cod]
*$py.class

# Logs
*.log
logs/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporales
*.tmp
temp/

# Cache del modelo
.cache/
models/

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
echo Iniciando aplicacion...
python app.py
pause"""
    
    with open("run.bat", "w") as f:
        f.write(bat_content)
    
    # Script para Linux/Mac
    sh_content = """#!/bin/bash
echo 'Activando entorno virtual...'
source venv/bin/activate
echo 'Iniciando aplicación...'
python app.py"""
    
    with open("run.sh", "w") as f:
        f.write(sh_content)
    
    # Hacer ejecutable el script de Linux/Mac
    if platform.system() != "Windows":
        os.chmod("run.sh", 0o755)
    
    print_success("Scripts de ejecución creados (run.bat / run.sh)")

def configure_api_key():
    """Configurar API key de OpenRouter"""
    print_header("CONFIGURACIÓN DE API KEY")
    
    print_info("Para usar el sistema necesitas una API key de OpenRouter")
    print_info("1. Ve a https://openrouter.ai/")
    print_info("2. Crea una cuenta y obtén tu API key")
    print_info("3. Edita el archivo .env y reemplaza 'tu-api-key-aqui'")
    
    response = input(f"\n{Colors.WARNING}¿Tienes tu API key ahora? (s/n): {Colors.ENDC}")
    
    if response.lower() == 's':
        api_key = input("Ingresa tu API key: ").strip()
        
        if api_key:
            # Leer .env actual
            with open(".env", "r") as f:
                lines = f.readlines()
            
            # Actualizar API key
            with open(".env", "w") as f:
                for line in lines:
                    if line.startswith("OPENROUTER_API_KEY"):
                        f.write(f"OPENROUTER_API_KEY={api_key}\n")
                    else:
                        f.write(line)
            
            print_success("API key configurada")
    else:
        print_warning("Recuerda configurar tu API key en .env antes de ejecutar")

def print_final_instructions():
    """Mostrar instrucciones finales"""
    print_header("🎉 INSTALACIÓN COMPLETA")
    
    print_success("El proyecto está listo para usar")
    print("")
    
    if platform.system() == "Windows":
        print("📋 PRÓXIMOS PASOS:")
        print("")
        print("1. Ejecutar la aplicación:")
        print(f"   {Colors.OKBLUE}run.bat{Colors.ENDC}")
        print("")
        print("   O manualmente:")
        print(f"   {Colors.OKBLUE}venv\\Scripts\\activate{Colors.ENDC}")
        print(f"   {Colors.OKBLUE}python app.py{Colors.ENDC}")
    else:
        print("📋 PRÓXIMOS PASOS:")
        print("")
        print("1. Ejecutar la aplicación:")
        print(f"   {Colors.OKBLUE}./run.sh{Colors.ENDC}")
        print("")
        print("   O manualmente:")
        print(f"   {Colors.OKBLUE}source venv/bin/activate{Colors.ENDC}")
        print(f"   {Colors.OKBLUE}python app.py{Colors.ENDC}")
    
    print("")
    print("2. Abrir en navegador:")
    print(f"   {Colors.OKBLUE}http://localhost:5000{Colors.ENDC}")
    print("")
    print_warning("IMPORTANTE: Configura tu API key en .env antes de ejecutar")
    print("")

def main():
    """Función principal"""
    print_header("🧠 INSTALADOR DE CHAT MULTI-MODELO CON MEMORIA VECTORIAL")
    
    # Verificar Python
    if not check_python_version():
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
    
    # Configurar API key
    configure_api_key()
    
    # Instrucciones finales
    print_final_instructions()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Instalación cancelada")
        sys.exit(0)