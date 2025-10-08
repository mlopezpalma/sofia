"""
Chat Multi-Modelo con OpenRouter, ChromaDB y Embeddings Locales
Sistema completo con memoria vectorial gratuita y acceso a LLMs grandes
"""
import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import secrets
import numpy as np

# ============================================
# IMPORTAR CONFIGURACIÓN DESDE config.py
# ============================================
try:
    from config import (
        # API Configuration
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        
        # Flask Configuration  
        FLASK_HOST,
        FLASK_PORT,
        FLASK_DEBUG,
        SECRET_KEY,
        
        # Models
        AVAILABLE_MODELS,
        
        # Embeddings
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        
        # Vector Search
        CHROMA_DB_PATH,
        MAX_CONTEXT_MESSAGES,
        SIMILARITY_THRESHOLD,
        
        # LLM Settings
        MAX_TOKENS_PER_RESPONSE,
        DEFAULT_TEMPERATURE,
        
        # Database
        SQLITE_DB_PATH,
        
        # Functions
        validate_config,
        print_config
    )
    
    print("\n🧠 Chat Multi-Modelo con Memoria Vectorial")
    print("=" * 50)
    
    # Validar y mostrar configuración
    if not validate_config():
        print("⚠️  Por favor configura tu API key en .env")
    
    print_config()
    
except ImportError as e:
    print("\n❌ ERROR: No se encontró config.py")
    print("   Asegúrate de que config.py está en el mismo directorio")
    print(f"   Error: {e}\n")
    exit(1)

# ============================================
# VERIFICAR DEPENDENCIAS
# ============================================
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    print("✅ Sentence-transformers instalado")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  Instala: pip install sentence-transformers")

import chromadb
from chromadb.config import Settings

# ============================================
# INICIALIZAR FLASK
# ============================================
app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

# ============================================
# INICIALIZAR CHROMADB
# ============================================

#version python 3.8+
#chroma_client = chromadb.PersistentClient(
#    path=CHROMA_DB_PATH,
#    settings=Settings(
#        anonymized_telemetry=False,
#        allow_reset=True
#    )
#)

# Compatibilidad con ChromaDB 0.3.x para Python 3.7
# Compatibilidad con ChromaDB 0.3.x para Python 3.7
import chromadb
from chromadb.config import Settings

# Para ChromaDB 0.3.x - Configuración simplificada
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DB_PATH
)
chroma_client = chromadb.Client(chroma_settings)


# ============================================
# SISTEMA DE MEMORIA VECTORIAL LOCAL
# ============================================
class LocalVectorMemory:
    """Sistema de memoria vectorial con embeddings locales gratuitos"""
    
    def __init__(self):
        # Inicializar modelo de embeddings local
        if EMBEDDINGS_AVAILABLE:
            print(f"🚀 Cargando modelo de embeddings local: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            print(f"✅ Modelo cargado - Dimensión: {self.embedding_dimension}")
        else:
            print("⚠️ Usando embeddings aleatorios de respaldo")
            self.embedding_model = None
            self.embedding_dimension = 384
        
        # Crear o cargar colecciones
        self._init_collections()
    
    def _init_collections(self):
        """Inicializar colecciones en ChromaDB"""
        # Función de embedding personalizada
        def embedding_function(texts: List[str]) -> List[List[float]]:
            if self.embedding_model:
                # Embeddings locales GRATUITOS
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            else:
                # Fallback: embeddings aleatorios
                return [np.random.rand(self.embedding_dimension).tolist() for _ in texts]
        
        # Colección para conversaciones
        try:
            self.conversations = chroma_client.get_collection("conversations")
            print("📚 Colección 'conversations' cargada")
        except:
            self.conversations = chroma_client.create_collection(
                name="conversations",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print("📚 Colección 'conversations' creada")
        
        # Colección para comparaciones
        try:
            self.comparisons = chroma_client.get_collection("comparisons")
            print("📚 Colección 'comparisons' cargada")
        except:
            self.comparisons = chroma_client.create_collection(
                name="comparisons",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print("📚 Colección 'comparisons' creada")
    
    def add_message(self, project_id: int, role: str, content: str, 
                   model: str = "", metadata: Dict = None) -> str:
        """Agregar mensaje a la memoria vectorial"""
        # Generar ID único
        doc_id = f"{project_id}_{datetime.now().timestamp()}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        # Preparar metadata
        full_metadata = {
            "project_id": project_id,
            "role": role,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content),
            **(metadata or {})
        }
        
        # Agregar a ChromaDB
        self.conversations.add(
            documents=[content],
            metadatas=[full_metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def get_relevant_context(self, project_id: int, query: str, 
                            max_results: int = MAX_CONTEXT_MESSAGES) -> List[Dict]:
        """Obtener contexto relevante usando búsqueda semántica"""
        try:
            # Buscar en la memoria vectorial
            results = self.conversations.query(
                query_texts=[query],
                n_results=min(max_results * 2, 20),  # Buscar más para filtrar
                where={"project_id": project_id}
            )
            
            if not results['documents'][0]:
                return []
            
            # Procesar y filtrar resultados
            relevant_messages = []
            seen_content = set()
            
            for i in range(len(results['documents'][0])):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                # Calcular similitud (1 - distancia para cosine)
                similarity = 1 - distance
                
                # Filtrar por umbral de similitud
                if similarity >= SIMILARITY_THRESHOLD:
                    # Evitar duplicados
                    content_hash = hashlib.md5(content[:100].encode()).hexdigest()
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        relevant_messages.append({
                            "role": metadata.get('role', 'assistant'),
                            "content": content,
                            "model": metadata.get('model', ''),
                            "similarity": similarity,
                            "timestamp": metadata.get('timestamp', '')
                        })
            
            # Ordenar por similitud y limitar
            relevant_messages.sort(key=lambda x: x['similarity'], reverse=True)
            return relevant_messages[:max_results]
            
        except Exception as e:
            print(f"Error obteniendo contexto: {e}")
            return []
    
    def add_comparison(self, project_id: int, question: str, 
                      responses: Dict[str, str], evaluation: Dict) -> str:
        """Guardar comparación en memoria vectorial"""
        # Crear documento combinado
        doc_content = f"Pregunta: {question}\n\n"
        for model, response in responses.items():
            doc_content += f"{model}:\n{response[:300]}...\n\n"
        
        doc_id = f"comp_{project_id}_{datetime.now().timestamp()}"
        
        metadata = {
            "project_id": project_id,
            "question": question[:200],
            "models": json.dumps(list(responses.keys())),
            "best_model": evaluation.get("best_model", ""),
            "best_score": evaluation.get("best_score", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        self.comparisons.add(
            documents=[doc_content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def search_similar_comparisons(self, project_id: int, question: str, 
                                  max_results: int = 3) -> List[Dict]:
        """Buscar comparaciones similares previas"""
        try:
            results = self.comparisons.query(
                query_texts=[question],
                n_results=max_results,
                where={"project_id": project_id}
            )
            
            similar_comparisons = []
            for i in range(len(results['documents'][0]) if results['documents'][0] else 0):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                similar_comparisons.append({
                    "question": metadata.get('question', ''),
                    "best_model": metadata.get('best_model', ''),
                    "models": json.loads(metadata.get('models', '[]')),
                    "similarity": 1 - distance,
                    "timestamp": metadata.get('timestamp', '')
                })
            
            return similar_comparisons
            
        except Exception as e:
            print(f"Error buscando comparaciones: {e}")
            return []
    
    def get_project_stats(self, project_id: int) -> Dict:
        """Obtener estadísticas del proyecto"""
        try:
            # Obtener todas las conversaciones del proyecto
            all_docs = self.conversations.get(
                where={"project_id": project_id},
                limit=1000
            )
            
            total_messages = len(all_docs['ids']) if all_docs['ids'] else 0
            
            # Contar por rol
            user_messages = sum(1 for m in all_docs['metadatas'] 
                              if m.get('role') == 'user')
            assistant_messages = sum(1 for m in all_docs['metadatas'] 
                                   if m.get('role') == 'assistant')
            
            # Obtener comparaciones
            all_comparisons = self.comparisons.get(
                where={"project_id": project_id},
                limit=100
            )
            
            total_comparisons = len(all_comparisons['ids']) if all_comparisons['ids'] else 0
            
            return {
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "total_comparisons": total_comparisons,
                "vector_db_size_mb": self._get_db_size()
            }
            
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "total_comparisons": 0,
                "vector_db_size_mb": 0
            }
    
    def _get_db_size(self) -> float:
        """Obtener tamaño de la base de datos en MB"""
        try:
            db_path = "./chroma_db"
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0

# ============================================
# GESTOR DE CHAT Y COMPARACIONES
# ============================================
class ChatManager:
    """Gestor principal del sistema de chat"""
    
    def __init__(self):
        self.vector_memory = LocalVectorMemory()
    
    def call_openrouter(self, model: str, messages: List[Dict[str, str]], 
                       temperature: float = 0.7) -> Dict:
        """Llamar a OpenRouter para obtener respuesta del LLM"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Multi-Model Chat with Vector Memory"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "content": data['choices'][0]['message']['content'],
                "tokens_used": data.get('usage', {}).get('total_tokens', 0)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "content": f"Error al llamar a {model}: {str(e)}",
                "tokens_used": 0
            }
    
    def chat_with_context(self, project_id: int, model: str, message: str, 
                         use_context: bool = True) -> Dict:
        """Chat con contexto vectorial inteligente"""
        
        # Construir mensajes con contexto
        messages = []
        context_info = []
        
        if use_context:
            # Obtener contexto relevante de la memoria vectorial
            relevant_context = self.vector_memory.get_relevant_context(
                project_id, message
            )
            
            # Agregar contexto relevante a los mensajes
            if relevant_context:
                # Agregar mensaje de sistema explicando el contexto
                context_summary = f"Contexto relevante de conversaciones anteriores ({len(relevant_context)} mensajes encontrados):"
                messages.append({"role": "system", "content": context_summary})
                
                # Agregar mensajes de contexto
                for ctx in relevant_context:
                    messages.append({
                        "role": ctx['role'],
                        "content": ctx['content']
                    })
                    context_info.append({
                        "preview": ctx['content'][:100] + "...",
                        "similarity": round(ctx['similarity'], 2)
                    })
        
        # Agregar mensaje actual
        messages.append({"role": "user", "content": message})
        
        # Llamar al modelo a través de OpenRouter
        response = self.call_openrouter(model, messages)
        
        if response['success']:
            # Guardar en memoria vectorial
            self.vector_memory.add_message(project_id, "user", message, model)
            self.vector_memory.add_message(project_id, "assistant", response['content'], model)
        
        return {
            "success": response['success'],
            "response": response['content'],
            "model": model,
            "context_used": len(context_info),
            "context_details": context_info,
            "tokens_used": response['tokens_used']
        }
    
    def compare_models(self, project_id: int, question: str, 
                      models: List[str]) -> Dict:
        """Comparar respuestas de múltiples modelos"""
        
        # Buscar comparaciones similares previas
        similar_comparisons = self.vector_memory.search_similar_comparisons(
            project_id, question
        )
        
        # Obtener contexto relevante
        context = self.vector_memory.get_relevant_context(project_id, question, max_results=3)
        
        # Construir mensajes base
        base_messages = []
        if context:
            base_messages.append({
                "role": "system", 
                "content": f"Contexto de conversaciones anteriores relevantes:"
            })
            for ctx in context[:3]:  # Limitar contexto para comparaciones
                base_messages.append({
                    "role": ctx['role'],
                    "content": ctx['content']
                })
        
        # Obtener respuesta de cada modelo
        responses = {}
        tokens_total = 0
        
        for model in models:
            messages = base_messages.copy()
            messages.append({"role": "user", "content": question})
            
            result = self.call_openrouter(model, messages)
            responses[model] = {
                "content": result['content'],
                "success": result['success'],
                "tokens": result['tokens_used']
            }
            tokens_total += result['tokens_used']
        
        # Evaluar respuestas
        evaluation = self._evaluate_responses(question, responses)
        
        # Guardar comparación en memoria
        responses_content = {m: r['content'] for m, r in responses.items()}
        self.vector_memory.add_comparison(project_id, question, responses_content, evaluation)
        
        # Guardar mensajes individuales
        self.vector_memory.add_message(project_id, "user", question, "comparison")
        for model, response in responses.items():
            if response['success']:
                self.vector_memory.add_message(
                    project_id, "assistant", 
                    response['content'], 
                    model,
                    {"comparison": True}
                )
        
        return {
            "question": question,
            "responses": responses,
            "evaluation": evaluation,
            "similar_comparisons": similar_comparisons,
            "context_used": len(context),
            "total_tokens": tokens_total
        }
    
    def _evaluate_responses(self, question: str, responses: Dict) -> Dict:
        """Evaluar y comparar respuestas usando un modelo árbitro"""
        
        # Preparar contenido para evaluación
        valid_responses = {m: r['content'] for m, r in responses.items() 
                          if r['success']}
        
        if len(valid_responses) < 2:
            return {
                "best_model": list(valid_responses.keys())[0] if valid_responses else "",
                "reasoning": "Evaluación no disponible - respuestas insuficientes",
                "scores": {}
            }
        
        evaluation_prompt = f"""
        Evalúa las siguientes respuestas a la pregunta: "{question}"
        
        Respuestas:
        {json.dumps(valid_responses, indent=2, ensure_ascii=False)}
        
        Criterios de evaluación:
        1. Precisión y corrección
        2. Completitud de la respuesta
        3. Claridad y estructura
        4. Utilidad práctica
        
        Proporciona tu evaluación en formato JSON:
        {{
            "scores": {{"modelo": puntuación del 1-10}},
            "best_model": "nombre_del_mejor_modelo",
            "reasoning": "explicación breve de la elección"
        }}
        """
        
        # Usar GPT-4 como árbitro
        messages = [{"role": "user", "content": evaluation_prompt}]
        result = self.call_openrouter("openai/gpt-4-turbo-preview", messages, temperature=0.3)
        
        if result['success']:
            try:
                # Intentar parsear JSON de la respuesta
                import re
                json_match = re.search(r'\{.*\}', result['content'], re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                    evaluation['best_score'] = max(evaluation.get('scores', {}).values()) if evaluation.get('scores') else 0
                    return evaluation
            except:
                pass
        
        # Fallback si falla la evaluación
        return {
            "best_model": list(valid_responses.keys())[0],
            "reasoning": "Evaluación automática no disponible",
            "scores": {m: 5 for m in valid_responses.keys()},
            "best_score": 5
        }

# ============================================
# BASE DE DATOS SQL PARA PROYECTOS
# ============================================
def init_database():
    """Inicializar base de datos SQLite para proyectos"""
    conn = sqlite3.connect('projects.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Crear proyecto por defecto si no existe
    cursor.execute('SELECT COUNT(*) FROM projects')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO projects (name, description)
            VALUES (?, ?)
        ''', ("Proyecto Principal", "Proyecto por defecto"))
    
    conn.commit()
    conn.close()

# ============================================
# INSTANCIA GLOBAL Y RUTAS DE LA API
# ============================================
chat_manager = ChatManager()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/api/projects', methods=['GET', 'POST'])
def handle_projects():
    """Gestionar proyectos"""
    if request.method == 'POST':
        data = request.json
        conn = sqlite3.connect('projects.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO projects (name, description)
            VALUES (?, ?)
        ''', (data['name'], data.get('description', '')))
        project_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return jsonify({'id': project_id, 'name': data['name']})
    
    else:
        conn = sqlite3.connect('projects.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, created_at FROM projects ORDER BY id DESC')
        projects = []
        for row in cursor.fetchall():
            project_id = row[0]
            # Obtener estadísticas del proyecto
            stats = chat_manager.vector_memory.get_project_stats(project_id)
            projects.append({
                'id': project_id,
                'name': row[1],
                'description': row[2],
                'created_at': row[3],
                'stats': stats
            })
        conn.close()
        return jsonify(projects)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint para chat con contexto vectorial"""
    data = request.json
    
    result = chat_manager.chat_with_context(
        project_id=data.get('project_id', 1),
        model=data['model'],
        message=data['message'],
        use_context=data.get('use_context', True)
    )
    
    return jsonify(result)

@app.route('/api/compare', methods=['POST'])
def compare():
    """Endpoint para comparar modelos"""
    data = request.json
    
    result = chat_manager.compare_models(
        project_id=data.get('project_id', 1),
        question=data['question'],
        models=data['models']
    )
    
    return jsonify(result)

@app.route('/api/search/<int:project_id>', methods=['POST'])
def search_memory(project_id):
    """Buscar en la memoria del proyecto"""
    data = request.json
    query = data['query']
    
    # Buscar contexto relevante
    context = chat_manager.vector_memory.get_relevant_context(
        project_id, query, max_results=10
    )
    
    # Buscar comparaciones similares
    comparisons = chat_manager.vector_memory.search_similar_comparisons(
        project_id, query, max_results=5
    )
    
    return jsonify({
        "context": context,
        "comparisons": comparisons
    })

@app.route('/api/stats/<int:project_id>')
def get_project_stats(project_id):
    """Obtener estadísticas del proyecto"""
    stats = chat_manager.vector_memory.get_project_stats(project_id)
    return jsonify(stats)

# ============================================
# TEMPLATE HTML
# ============================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Chat Multi-Modelo con Memoria Vectorial Local</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .status-indicators {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }
        
        .status-badge {
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 280px 1fr 380px;
            height: calc(100vh - 200px);
        }
        
        .sidebar {
            background: #f8f9fa;
            padding: 20px;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
        }
        
        .chat-area {
            display: flex;
            flex-direction: column;
            padding: 20px;
            background: #fff;
        }
        
        .memory-panel {
            background: #f8f9fa;
            padding: 20px;
            border-left: 1px solid #e0e0e0;
            overflow-y: auto;
        }
        
        .section-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .project-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .project-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }
        
        .project-card.active {
            border-color: #667eea;
            background: linear-gradient(to right, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        }
        
        .project-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
            margin-top: 10px;
            font-size: 0.85em;
            color: #666;
        }
        
        .model-selector {
            margin-bottom: 20px;
        }
        
        .model-checkbox {
            display: block;
            margin: 8px 0;
            padding: 10px;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid #e0e0e0;
        }
        
        .model-checkbox:hover {
            background: linear-gradient(to right, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
            transform: translateX(5px);
        }
        
        .model-checkbox input {
            margin-right: 10px;
        }
        
        .model-checkbox.selected {
            background: linear-gradient(to right, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border-color: #667eea;
        }
        
        .context-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }
        
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: 0.4s;
            border-radius: 26px;
        }
        
        .slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.4s;
            border-radius: 50%;
        }
        
        input:checked + .slider {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        input:checked + .slider:before {
            transform: translateX(24px);
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #fafafa;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .message {
            margin: 15px 0;
            padding: 15px 20px;
            border-radius: 15px;
            animation: fadeIn 0.3s;
            max-width: 80%;
        }
        
        @keyframes fadeIn {
            from { 
                opacity: 0; 
                transform: translateY(10px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        
        .assistant-message {
            background: white;
            border: 1px solid #e0e0e0;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }
        
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.85em;
            opacity: 0.8;
        }
        
        .model-badge {
            background: rgba(0,0,0,0.1);
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
        }
        
        .context-indicator {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: #e8f4f8;
            color: #667eea;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            margin-left: 10px;
        }
        
        .comparison-container {
            background: white;
            border: 2px solid #667eea;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .comparison-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .comparison-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #667eea;
        }
        
        .comparison-stats {
            display: flex;
            gap: 15px;
            font-size: 0.9em;
        }
        
        .model-response {
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .model-response-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .model-name {
            font-weight: 600;
            color: #333;
        }
        
        .model-score {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        
        .best-model-badge {
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            margin-left: 10px;
        }
        
        .evaluation-box {
            background: linear-gradient(to right, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            padding: 15px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        .message-input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            resize: none;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        .message-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.95em;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(240, 147, 251, 0.3);
        }
        
        .memory-search {
            position: relative;
            margin-bottom: 15px;
        }
        
        .memory-search input {
            width: 100%;
            padding: 10px 15px 10px 35px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            font-size: 0.95em;
        }
        
        .memory-search::before {
            content: "🔍";
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
        }
        
        .memory-tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }
        
        .memory-tab {
            flex: 1;
            padding: 8px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            text-align: center;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        
        .memory-tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }
        
        .memory-content {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .memory-item {
            background: white;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 3px solid #667eea;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .memory-item:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .memory-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
            font-size: 0.85em;
        }
        
        .similarity-badge {
            background: #e8f4f8;
            color: #667eea;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
        }
        
        .memory-item-content {
            font-size: 0.9em;
            color: #666;
            line-height: 1.4;
        }
        
        .empty-state {
            text-align: center;
            padding: 50px 20px;
            color: #999;
        }
        
        .empty-state h3 {
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        
        .loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            background: #fee;
            color: #c00;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #c00;
        }
        
        .success-message {
            background: #efe;
            color: #080;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #080;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Chat Multi-Modelo con Memoria Vectorial</h1>
            <div class="subtitle">Embeddings Locales + ChromaDB + OpenRouter</div>
            <div class="status-indicators">
                <span class="status-badge">💾 ChromaDB: Local</span>
                <span class="status-badge">🔧 Embeddings: Gratuitos</span>
                <span class="status-badge">🚀 LLMs: OpenRouter</span>
            </div>
        </div>
        
        <div class="main-content">
            <!-- Panel Izquierdo: Proyectos y Modelos -->
            <div class="sidebar">
                <div class="section-title">
                    <span>📁</span> Proyectos
                </div>
                <div id="projectsList"></div>
                <button class="btn btn-primary" style="width: 100%; margin-top: 10px;" onclick="createProject()">
                    + Nuevo Proyecto
                </button>
                
                <div class="section-title" style="margin-top: 25px;">
                    <span>🤖</span> Modelos LLM
                </div>
                <div id="modelsList" class="model-selector"></div>
                
                <div class="context-toggle">
                    <label class="toggle-switch">
                        <input type="checkbox" id="useContext" checked>
                        <span class="slider"></span>
                    </label>
                    <span>Usar Memoria Vectorial</span>
                </div>
            </div>
            
            <!-- Área Central: Chat -->
            <div class="chat-area">
                <div class="chat-messages" id="chatMessages">
                    <div class="empty-state">
                        <h3>💬 Comienza una Conversación</h3>
                        <p>Selecciona uno o más modelos y escribe tu mensaje</p>
                        <p style="margin-top: 10px; font-size: 0.9em;">
                            La memoria vectorial encontrará automáticamente<br>
                            el contexto más relevante para cada pregunta
                        </p>
                    </div>
                </div>
                
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea 
                            id="messageInput" 
                            class="message-input"
                            placeholder="Escribe tu mensaje aquí..."
                            rows="2"
                            onkeypress="if(event.key==='Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }"
                        ></textarea>
                    </div>
                    <div class="button-group">
                        <button class="btn btn-primary" onclick="sendMessage()">
                            Enviar
                        </button>
                        <button class="btn btn-secondary" onclick="compareModels()">
                            Comparar
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Panel Derecho: Memoria -->
            <div class="memory-panel">
                <div class="section-title">
                    <span>🧠</span> Memoria Vectorial
                </div>
                
                <div class="memory-search">
                    <input 
                        type="text" 
                        id="searchInput" 
                        placeholder="Buscar en la memoria..."
                        onkeyup="searchMemory(this.value)"
                    >
                </div>
                
                <div class="memory-tabs">
                    <div class="memory-tab active" onclick="switchTab('stats')">
                        📊 Stats
                    </div>
                    <div class="memory-tab" onclick="switchTab('context')">
                        📚 Contexto
                    </div>
                    <div class="memory-tab" onclick="switchTab('history')">
                        🕐 Historial
                    </div>
                </div>
                
                <div class="memory-content">
                    <div id="statsContent" class="tab-content"></div>
                    <div id="contextContent" class="tab-content" style="display: none;"></div>
                    <div id="historyContent" class="tab-content" style="display: none;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Estado global de la aplicación
        let currentProjectId = 1;
        let selectedModels = [];
        let isLoading = false;
        let searchTimeout = null;
        
        // Inicializar aplicación
        document.addEventListener('DOMContentLoaded', () => {
            loadProjects();
            loadModels();
            updateStats();
        });
        
        // Cargar proyectos
        async function loadProjects() {
            try {
                const response = await fetch('/api/projects');
                const projects = await response.json();
                
                const container = document.getElementById('projectsList');
                container.innerHTML = '';
                
                projects.forEach(project => {
                    const card = document.createElement('div');
                    card.className = `project-card ${project.id === currentProjectId ? 'active' : ''}`;
                    card.onclick = () => selectProject(project.id);
                    
                    const stats = project.stats || {};
                    card.innerHTML = `
                        <div style="font-weight: 600;">${project.name}</div>
                        <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                            ${project.description || 'Sin descripción'}
                        </div>
                        <div class="project-stats">
                            <span>💬 ${stats.total_messages || 0} msgs</span>
                            <span>📊 ${stats.total_comparisons || 0} comp</span>
                        </div>
                    `;
                    
                    container.appendChild(card);
                });
                
                if (projects.length > 0 && !currentProjectId) {
                    currentProjectId = projects[0].id;
                }
            } catch (error) {
                console.error('Error cargando proyectos:', error);
            }
        }
        
        // Cargar modelos disponibles
        function loadModels() {
            const models = {{ models | tojson }};
            const container = document.getElementById('modelsList');
            
            models.forEach(model => {
                const label = document.createElement('label');
                label.className = 'model-checkbox';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.value = model;
                checkbox.onchange = (e) => {
                    if (e.target.checked) {
                        selectedModels.push(model);
                        label.classList.add('selected');
                    } else {
                        selectedModels = selectedModels.filter(m => m !== model);
                        label.classList.remove('selected');
                    }
                };
                
                const modelName = model.split('/').pop();
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(modelName));
                
                container.appendChild(label);
            });
        }
        
        // Seleccionar proyecto
        function selectProject(projectId) {
            currentProjectId = projectId;
            
            // Actualizar UI
            document.querySelectorAll('.project-card').forEach(card => {
                card.classList.remove('active');
            });
            event.currentTarget.classList.add('active');
            
            // Limpiar chat
            document.getElementById('chatMessages').innerHTML = `
                <div class="empty-state">
                    <h3>📂 Proyecto Cargado</h3>
                    <p>La memoria vectorial está lista</p>
                </div>
            `;
            
            // Actualizar estadísticas
            updateStats();
        }
        
        // Crear nuevo proyecto
        async function createProject() {
            const name = prompt('Nombre del proyecto:');
            if (!name) return;
            
            const description = prompt('Descripción (opcional):');
            
            try {
                const response = await fetch('/api/projects', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, description })
                });
                
                const project = await response.json();
                currentProjectId = project.id;
                await loadProjects();
                
                showMessage('success', `Proyecto "${name}" creado exitosamente`);
            } catch (error) {
                showMessage('error', 'Error creando proyecto');
            }
        }
        
        // Enviar mensaje
        async function sendMessage() {
            if (isLoading) return;
            
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            if (selectedModels.length === 0) {
                showMessage('error', 'Por favor selecciona al menos un modelo');
                return;
            }
            
            if (selectedModels.length > 1) {
                // Si hay múltiples modelos, hacer comparación
                await compareModels();
                return;
            }
            
            isLoading = true;
            input.value = '';
            
            // Mostrar mensaje del usuario
            addMessage('user', message);
            
            // Mostrar indicador de carga
            showLoading();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: currentProjectId,
                        model: selectedModels[0],
                        message: message,
                        use_context: document.getElementById('useContext').checked
                    })
                });
                
                const result = await response.json();
                hideLoading();
                
                if (result.success) {
                    // Mostrar respuesta
                    addMessage('assistant', result.response, {
                        model: selectedModels[0],
                        contextUsed: result.context_used,
                        tokens: result.tokens_used
                    });
                    
                    // Actualizar panel de contexto si se usó
                    if (result.context_details && result.context_details.length > 0) {
                        updateContextPanel(result.context_details);
                    }
                } else {
                    showMessage('error', result.response);
                }
                
                updateStats();
                
            } catch (error) {
                hideLoading();
                showMessage('error', 'Error al enviar mensaje');
            }
            
            isLoading = false;
        }
        
        // Comparar modelos
        async function compareModels() {
            if (isLoading) return;
            
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) {
                // Buscar último mensaje del usuario
                const lastUserMsg = [...document.querySelectorAll('.user-message')].pop();
                if (!lastUserMsg) {
                    showMessage('error', 'Por favor escribe una pregunta');
                    return;
                }
            }
            
            if (selectedModels.length < 2) {
                showMessage('error', 'Selecciona al menos 2 modelos para comparar');
                return;
            }
            
            isLoading = true;
            const question = message || [...document.querySelectorAll('.user-message')].pop()?.textContent;
            input.value = '';
            
            // Mostrar pregunta si es nueva
            if (message) {
                addMessage('user', message);
            }
            
            showLoading();
            
            try {
                const response = await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: currentProjectId,
                        question: question,
                        models: selectedModels
                    })
                });
                
                const result = await response.json();
                hideLoading();
                
                displayComparison(result);
                updateStats();
                
            } catch (error) {
                hideLoading();
                showMessage('error', 'Error al comparar modelos');
            }
            
            isLoading = false;
        }
        
        // Mostrar comparación
        function displayComparison(data) {
            const container = document.createElement('div');
            container.className = 'comparison-container';
            
            // Header
            const header = document.createElement('div');
            header.className = 'comparison-header';
            header.innerHTML = `
                <div class="comparison-title">📊 Comparación de Modelos</div>
                <div class="comparison-stats">
                    <span class="context-indicator">
                        📚 ${data.context_used || 0} contextos
                    </span>
                    <span class="context-indicator">
                        ⚡ ${data.total_tokens || 0} tokens
                    </span>
                </div>
            `;
            container.appendChild(header);
            
            // Mostrar si hay comparaciones similares previas
            if (data.similar_comparisons && data.similar_comparisons.length > 0) {
                const similarDiv = document.createElement('div');
                similarDiv.style.cssText = 'background: #e8f4f8; padding: 10px; border-radius: 8px; margin-bottom: 15px;';
                similarDiv.innerHTML = `
                    <strong>💡 Comparaciones similares encontradas:</strong>
                    ${data.similar_comparisons.map(c => 
                        `<div style="margin-top: 5px;">• ${c.question.substring(0, 50)}... (Mejor: ${c.best_model})</div>`
                    ).join('')}
                `;
                container.appendChild(similarDiv);
            }
            
            // Respuestas de modelos
            const evaluation = data.evaluation || {};
            const bestModel = evaluation.best_model;
            
            Object.entries(data.responses).forEach(([model, response]) => {
                const responseDiv = document.createElement('div');
                responseDiv.className = 'model-response';
                
                const score = evaluation.scores?.[model] || 0;
                const isBest = model === bestModel;
                
                responseDiv.innerHTML = `
                    <div class="model-response-header">
                        <div>
                            <span class="model-name">${model.split('/').pop()}</span>
                            ${isBest ? '<span class="best-model-badge">✨ Mejor Respuesta</span>' : ''}
                        </div>
                        <span class="model-score">${score}/10</span>
                    </div>
                    <div style="margin-top: 10px; line-height: 1.5;">
                        ${response.content || response}
                    </div>
                `;
                
                container.appendChild(responseDiv);
            });
            
            // Evaluación
            if (evaluation.reasoning) {
                const evalDiv = document.createElement('div');
                evalDiv.className = 'evaluation-box';
                evalDiv.innerHTML = `
                    <strong>🎯 Evaluación del Árbitro:</strong>
                    <div style="margin-top: 10px;">${evaluation.reasoning}</div>
                `;
                container.appendChild(evalDiv);
            }
            
            document.getElementById('chatMessages').appendChild(container);
            scrollToBottom();
        }
        
        // Agregar mensaje al chat
        function addMessage(type, content, metadata = {}) {
            const messagesContainer = document.getElementById('chatMessages');
            
            // Limpiar estado vacío si existe
            const emptyState = messagesContainer.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            if (type === 'assistant' && metadata.model) {
                const modelName = metadata.model.split('/').pop();
                messageDiv.innerHTML = `
                    <div class="message-header">
                        <span class="model-badge">${modelName}</span>
                        ${metadata.contextUsed ? `
                            <span class="context-indicator">
                                📚 ${metadata.contextUsed} contextos
                            </span>
                        ` : ''}
                    </div>
                    <div>${content}</div>
                `;
            } else {
                messageDiv.textContent = content;
            }
            
            messagesContainer.appendChild(messageDiv);
            scrollToBottom();
        }
        
        // Buscar en memoria
        async function searchMemory(query) {
            if (searchTimeout) clearTimeout(searchTimeout);
            
            if (query.length < 2) {
                document.getElementById('historyContent').innerHTML = '';
                return;
            }
            
            searchTimeout = setTimeout(async () => {
                try {
                    const response = await fetch(`/api/search/${currentProjectId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query })
                    });
                    
                    const results = await response.json();
                    displaySearchResults(results);
                } catch (error) {
                    console.error('Error buscando:', error);
                }
            }, 300);
        }
        
        // Mostrar resultados de búsqueda
        function displaySearchResults(results) {
            const container = document.getElementById('historyContent');
            container.innerHTML = '';
            
            if (!results.context || results.context.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #999;">No se encontraron resultados</p>';
                return;
            }
            
            results.context.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'memory-item';
                itemDiv.innerHTML = `
                    <div class="memory-item-header">
                        <span>${item.role === 'user' ? '👤' : '🤖'} ${item.model || ''}</span>
                        <span class="similarity-badge">${Math.round(item.similarity * 100)}%</span>
                    </div>
                    <div class="memory-item-content">
                        ${item.content.substring(0, 150)}...
                    </div>
                `;
                container.appendChild(itemDiv);
            });
            
            // Cambiar a tab de historial
            switchTab('history');
        }
        
        // Actualizar panel de contexto
        function updateContextPanel(contexts) {
            const container = document.getElementById('contextContent');
            container.innerHTML = '<h4 style="margin-bottom: 10px;">Contextos Utilizados:</h4>';
            
            contexts.forEach((ctx, index) => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'memory-item';
                itemDiv.innerHTML = `
                    <div class="memory-item-header">
                        <span>Contexto ${index + 1}</span>
                        <span class="similarity-badge">${Math.round(ctx.similarity * 100)}%</span>
                    </div>
                    <div class="memory-item-content">${ctx.preview}</div>
                `;
                container.appendChild(itemDiv);
            });
            
            switchTab('context');
        }
        
        // Actualizar estadísticas
        async function updateStats() {
            if (!currentProjectId) return;
            
            try {
                const response = await fetch(`/api/stats/${currentProjectId}`);
                const stats = await response.json();
                
                const container = document.getElementById('statsContent');
                container.innerHTML = `
                    <div style="display: grid; gap: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>Total Mensajes:</span>
                            <strong>${stats.total_messages || 0}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Mensajes Usuario:</span>
                            <strong>${stats.user_messages || 0}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Respuestas IA:</span>
                            <strong>${stats.assistant_messages || 0}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Comparaciones:</span>
                            <strong>${stats.total_comparisons || 0}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Tamaño DB:</span>
                            <strong>${stats.vector_db_size_mb || 0} MB</strong>
                        </div>
                    </div>
                    <div style="margin-top: 20px; padding: 10px; background: #f0f8ff; border-radius: 8px;">
                        <strong>💡 Tip:</strong><br>
                        La memoria vectorial encuentra automáticamente el contexto más relevante para cada pregunta.
                    </div>
                `;
            } catch (error) {
                console.error('Error cargando estadísticas:', error);
            }
        }
        
        // Cambiar tab
        function switchTab(tabName) {
            // Actualizar tabs
            document.querySelectorAll('.memory-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            event.currentTarget?.classList.add('active');
            
            // Actualizar contenido
            document.querySelectorAll('.tab-content').forEach(content => {
                content.style.display = 'none';
            });
            
            document.getElementById(`${tabName}Content`).style.display = 'block';
            
            if (tabName === 'stats') {
                updateStats();
            }
        }
        
        // Mostrar indicador de carga
        function showLoading() {
            const loadingDiv = document.createElement('div');
            loadingDiv.id = 'loadingIndicator';
            loadingDiv.className = 'loading';
            loadingDiv.innerHTML = `
                <div class="spinner"></div>
                <div>Procesando con memoria vectorial...</div>
            `;
            document.getElementById('chatMessages').appendChild(loadingDiv);
            scrollToBottom();
        }
        
        // Ocultar indicador de carga
        function hideLoading() {
            const loading = document.getElementById('loadingIndicator');
            if (loading) loading.remove();
        }
        
        // Mostrar mensaje de estado
        function showMessage(type, message) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `${type}-message`;
            messageDiv.textContent = message;
            document.getElementById('chatMessages').appendChild(messageDiv);
            scrollToBottom();
            
            // Auto-eliminar después de 5 segundos
            setTimeout(() => messageDiv.remove(), 5000);
        }
        
        // Scroll al final del chat
        function scrollToBottom() {
            const messages = document.getElementById('chatMessages');
            messages.scrollTop = messages.scrollHeight;
        }
    </script>
</body>
</html>
"""

# ============================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 CHAT MULTI-MODELO CON MEMORIA VECTORIAL LOCAL")
    print("=" * 60)
    print()
    
    # Verificar dependencias
    print("📋 VERIFICANDO DEPENDENCIAS:")
    print("-" * 40)
    
    dependencies_ok = True
    
    # Verificar sentence-transformers
    if EMBEDDINGS_AVAILABLE:
        print("✅ sentence-transformers instalado")
    else:
        print("❌ sentence-transformers NO instalado")
        print("   Ejecuta: pip install sentence-transformers")
        dependencies_ok = False
    
    # Verificar ChromaDB
    try:
        import chromadb
        print("✅ ChromaDB instalado")
    except ImportError:
        print("❌ ChromaDB NO instalado")
        print("   Ejecuta: pip install chromadb")
        dependencies_ok = False
    
    if not dependencies_ok:
        print()
        print("⚠️ INSTALACIÓN REQUERIDA:")
        print("pip install flask flask-cors requests chromadb sentence-transformers")
        print()
    
    print()
    print("🔑 CONFIGURACIÓN:")
    print("-" * 40)
    print("1. Obtén tu API Key de OpenRouter:")
    print("   https://openrouter.ai/")
    print("   Reemplaza 'tu-api-key-aqui' en la línea 35")
    print()
    
    print("🚀 ARQUITECTURA DEL SISTEMA:")
    print("-" * 40)
    print("• Embeddings: LOCALES y GRATUITOS (all-MiniLM-L6-v2)")
    print("• Base Vectorial: ChromaDB LOCAL (./chroma_db)")
    print("• LLMs: Via OpenRouter API")
    print("• Coste adicional: $0 (solo pagas por tokens de chat)")
    print()
    
    # Crear directorio templates si no existe
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Directorio 'templates' creado")
    
    # Guardar template HTML
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
        print("📄 Template HTML guardado")
    
    # Inicializar base de datos
    init_database()
    print("💾 Base de datos inicializada")
    
    print()
    print("🌐 SERVIDOR INICIANDO...")
    print("-" * 40)
    print("Abre tu navegador en: http://localhost:5000")
    print()
    print("Presiona Ctrl+C para detener el servidor")
    print("=" * 60)
    
    # Ejecutar servidor
    app.run(debug=True, port=5000, host='0.0.0.0')