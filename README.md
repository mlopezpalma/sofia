# 🧠 Chat Multi-Modelo con Memoria Vectorial Local

Sistema de chat con IA que implementa memoria semántica vectorial usando embeddings locales gratuitos, ChromaDB para almacenamiento vectorial, y acceso a múltiples LLMs através de OpenRouter.

## 🚀 Instalación Rápida (Automática)

```bash
# 1. Clonar el repositorio
git clone https://github.com/mlopezpalma/sofia.git
cd sofia

# 2. Ejecutar instalador automático
python setup.py

# El instalador:
# - Crea entorno virtual
# - Instala dependencias
# - Genera archivos de configuración
# - Crea scripts de ejecución
```

## 🛠️ Instalación Manual

### 1. Crear entorno virtual

```bash
# Crear entorno
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar API Key

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env y agregar tu API key real
OPENROUTER_API_KEY=sk-or-tu-api-key-real-aqui
```

### 4. Ejecutar aplicación

```bash
# Windows:
run.bat

# Linux/Mac:
./run.sh

# O manualmente:
python app.py
```

## ⚙️ Configuración

### Estructura de configuración

```
.env          → Valores secretos (API keys, passwords)
     ↓
config.py     → Lee .env y define configuración
     ↓  
app.py        → Importa todo desde config.py
```

### Variables principales en `.env`

```env
# Requerido
OPENROUTER_API_KEY=tu-api-key-aqui

# Opcional
FLASK_PORT=5000
MAX_TOKENS=2000
TEMPERATURE=0.7
MAX_CONTEXT=5
```

### Configuración en `config.py`

- Lista de modelos disponibles
- Configuración de embeddings
- Límites y umbrales
- Rutas de bases de datos

## 📁 Estructura del Proyecto

```
chat-memoria-vectorial/
├── venv/                 # Entorno virtual (no subir a git)
├── app.py               # Aplicación principal
├── config.py            # Configuración centralizada  
├── setup.py             # Instalador automático
├── .env                 # Variables secretas (no subir a git)
├── .env.example         # Plantilla de variables
├── requirements.txt     # Dependencias
├── run.bat             # Script Windows
├── run.sh              # Script Linux/Mac
├── README.md           # Este archivo
│
├── templates/          # (Auto-generado)
│   └── index.html     # Interfaz web
├── chroma_db/         # (Auto-generado)
│   └── [vectores]     # Base vectorial
└── projects.db        # (Auto-generado)
```

## 🎮 Uso

### Ejecutar con scripts

```bash
# Windows
run.bat

# Linux/Mac  
./run.sh
```

### Ejecutar manualmente

```bash
# 1. Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Ejecutar aplicación
python app.py

# 3. Abrir navegador
http://localhost:5000
```

## 🔧 Personalización

### Cambiar puerto

En `.env`:
```env
FLASK_PORT=8080
```

### Ajustar límites de tokens

En `.env`:
```env
MAX_TOKENS=3000
TEMPERATURE=0.9
```

### Agregar modelos

En `config.py`:
```python
AVAILABLE_MODELS = [
    "openai/gpt-4-turbo-preview",
    "tu-nuevo-modelo/aqui",
    # ...
]
```

## 🐛 Solución de Problemas

### Error: No module named 'flask'

```bash
# Activar entorno virtual
source venv/bin/activate
pip install -r requirements.txt
```

### Error: API key no configurada

```bash
# Editar .env
OPENROUTER_API_KEY=tu-api-key-real
```

### El modelo tarda en descargar

```bash
# Pre-descargar modelo
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## 📊 Características

- ✅ Embeddings locales gratuitos
- ✅ Búsqueda semántica inteligente
- ✅ Multi-modelo (GPT-4, Claude, Gemini, etc.)
- ✅ Comparación de respuestas
- ✅ Gestión de proyectos
- ✅ 100% privacidad local
- ✅ Sin límites de uso
- ✅ Costo $0 en infraestructura

## 💰 Comparación de Costos

| Solución | Costo Mensual | Privacidad |
|----------|---------------|------------|
| Este Proyecto | $0 | 100% Local |
| OpenAI + Pinecone | $150-500 | Cloud |
| Azure Cognitive | $200-800 | Cloud |

## 📝 Licencia

MIT License

## 🤝 Contribuir

1. Fork el proyecto
2. Crear branch (`git checkout -b feature/nueva`)
3. Commit cambios (`git commit -m 'Add: feature'`)
4. Push (`git push origin feature/nueva`)
5. Crear Pull Request

## 🆘 Soporte.-- nodisponible 


---

**Recuerda:** Siempre trabaja dentro del entorno virtual (`venv`) para mantener las dependencias aisladas.
