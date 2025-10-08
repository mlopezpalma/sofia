"""
config.py - Configuración centralizada del sistema
Lee valores secretos de .env y define valores por defecto
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# ============================================
# CONFIGURACIÓN DE API (desde .env)
# ============================================
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'tu-api-key-aqui')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'

# ============================================
# CONFIGURACIÓN DE FLASK (desde .env con defaults)
# ============================================
FLASK_HOST = '0.0.0.0'
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-cambiar-en-produccion')

# ============================================
# MODELOS DISPONIBLES (fijos en código)
# ============================================
AVAILABLE_MODELS = [
    "google/gemini-2.5-flash-image",
    "openai/gpt-5-pro",
    "z-ai/glm-4.6",
    "anthropic/claude-sonnet-4.5",
    "deepseek/deepseek-v3.2-exp",
    "thedrummer/cydonia-24b-v4.1",
    "relace/relace-apply-3",
    "google/gemini-2.5-flash-preview-09-2025",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "openai/gpt-3.5-turbo",              # Económico y rápido
    "openai/gpt-4",                      # Más potente
    "openai/gpt-4-turbo-preview",        # GPT-4 Turbo
    "anthropic/claude-3-opus-20240229",  # Claude más potente
    "anthropic/claude-3-sonnet-20240229", # Claude balance
    "anthropic/claude-3-haiku-20240307",  # Claude rápido
    "google/gemini-pro",                  # Google Gemini
    "google/gemini-pro-1.5",              # Gemini 1.5
    "meta-llama/llama-3-70b-instruct",    # Llama 3 grande
    "meta-llama/llama-3-8b-instruct",     # Llama 3 pequeño
    "mistralai/mistral-7b-instruct",      # Mistral pequeño
    "mistralai/mixtral-8x7b-instruct-45b", # Mixtral grande
]

# ============================================
# CONFIGURACIÓN DE EMBEDDINGS (mayormente fijos)
# ============================================
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Modelo local gratuito
EMBEDDING_DIMENSION = 384

# ============================================
# CONFIGURACIÓN DE MEMORIA VECTORIAL
# ============================================
CHROMA_DB_PATH = './chroma_db'
MAX_CONTEXT_MESSAGES = int(os.getenv('MAX_CONTEXT', 5))
SIMILARITY_THRESHOLD = 0.7  # Umbral de similitud (0-1)

# ============================================
# CONFIGURACIÓN DE LLM (personalizables desde .env)
# ============================================
MAX_TOKENS_PER_RESPONSE = int(os.getenv('MAX_TOKENS', 2000))
DEFAULT_TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))

# ============================================
# BASES DE DATOS
# ============================================
SQLITE_DB_PATH = './projects.db'

# ============================================
# FUNCIÓN DE VALIDACIÓN
# ============================================
def validate_config():
    """Verificar que la configuración esté correcta"""
    if OPENROUTER_API_KEY == 'tu-api-key-aqui':
        print("\n⚠️  ADVERTENCIA: OpenRouter API key no configurada")
        print("   Edita .env y agrega tu API key real")
        print("   OPENROUTER_API_KEY=tu-key-real-aqui\n")
        return False
    return True

# ============================================
# IMPRIMIR CONFIGURACIÓN (sin secretos)
# ============================================
def print_config():
    """Mostrar configuración actual (sin valores sensibles)"""
    print("\n📋 Configuración Actual:")
    print(f"   Puerto: {FLASK_PORT}")
    print(f"   Debug: {FLASK_DEBUG}")
    print(f"   Modelo Embeddings: {EMBEDDING_MODEL}")
    print(f"   Max Tokens: {MAX_TOKENS_PER_RESPONSE}")
    print(f"   Temperatura: {DEFAULT_TEMPERATURE}")
    print(f"   Contexto Máximo: {MAX_CONTEXT_MESSAGES}")
    print(f"   API Key Configurada: {'✅' if OPENROUTER_API_KEY != 'tu-api-key-aqui' else '❌'}")
    print(f"   Modelos Disponibles: {len(AVAILABLE_MODELS)}")
    print()