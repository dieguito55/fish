from sentence_transformers import SentenceTransformer, util
from sqlalchemy import func
from backend import db, models
import torch

# --- CONFIGURACIÓN ---
# Modelo multilingüe pequeño y rápido para embeddings
print("🧠 Cargando modelo de embeddings (Sentence-BERT)...")
model_emb = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Definimos las "Intenciones" (Qué sabe hacer nuestro sistema)
# Cada clave tiene varias formas de preguntar para que el modelo aprenda ejemplos
INTENTS = {
    "total_count": [
        "¿Cuántos peces se han detectado hoy?",
        "Dime el total de detecciones",
        "recuento de peces",
        "cantidad total",
        "¿hubo mucha actividad?"
    ],
    "performance": [
        "¿Cómo va el rendimiento?",
        "dime los FPS promedio",
        "latencia del sistema",
        "¿está lento el sistema?",
        "velocidad de procesamiento"
    ],
    "last_seen": [
        "¿A qué hora fue el último pez?",
        "última detección",
        "hora actual de monitoreo",
        "¿cuándo viste algo?"
    ]
}

# Pre-calculamos los vectores de nuestras intenciones (Base de Conocimiento)
intent_embeddings = {}
intent_keys = []
corpus_embeddings = []

for key, phrases in INTENTS.items():
    for phrase in phrases:
        intent_keys.append(key) # Guardamos a qué intención pertenece cada frase
    
    # Codificamos todas las frases de esta intención
    embeddings = model_emb.encode(phrases, convert_to_tensor=True)
    corpus_embeddings.append(embeddings)

# Aplanamos para búsqueda rápida
corpus_embeddings = torch.cat(corpus_embeddings, dim=0)


def answer_question(user_question):
    """
    Pipeline NLP:
    1. Embed de la pregunta usuario
    2. Búsqueda semántica (Cosine Similarity)
    3. Ejecución de SQL según la intención ganadora
    """
    session = db.SessionLocal()
    try:
        # 1. Entender la pregunta (Embedding)
        question_embedding = model_emb.encode(user_question, convert_to_tensor=True)
        
        # 2. Buscar similitud
        cos_scores = util.cos_sim(question_embedding, corpus_embeddings)[0]
        best_score_idx = torch.argmax(cos_scores)
        best_score = cos_scores[best_score_idx]
        
        # Detectar la intención ganadora
        detected_intent = intent_keys[best_score_idx]
        
        print(f"DEBUG: Intención detectada: '{detected_intent}' (Score: {best_score:.4f})")

        # Umbral de duda (Si no se parece a nada)
        if best_score < 0.4:
            return "Lo siento, soy un experto en peces, pero no entendí esa pregunta específica."

        # 3. Generar respuesta basada en datos (SQL)
        if detected_intent == "total_count":
            count = session.query(func.sum(models.DetectionEvent.num_fish)).scalar() or 0
            return f"Hasta ahora, he contabilizado un total de **{count} peces** en el stream."
            
        elif detected_intent == "performance":
            fps = session.query(func.avg(models.DetectionEvent.fps)).scalar() or 0
            lat = session.query(func.avg(models.DetectionEvent.latency)).scalar() or 0
            return f"El sistema corre a **{fps:.1f} FPS** con una latencia media de {lat:.1f}ms."
            
        elif detected_intent == "last_seen":
            last = session.query(models.DetectionEvent).order_by(models.DetectionEvent.timestamp.desc()).first()
            if last:
                time_str = last.timestamp.strftime("%H:%M:%S")
                return f"La última actividad fue registrada a las **{time_str}**."
            else:
                return "Aún no he detectado nada hoy."
                
    except Exception as e:
        return f"Error procesando tu pregunta: {str(e)}"
    finally:
        session.close()

if __name__ == "__main__":
    # Prueba interactiva en terminal
    print("\n🤖 Chatbot FishWatch (Escribe 'salir' para terminar)")
    while True:
        q = input("Tú: ")
        if q.lower() == "salir": break
        ans = answer_question(q)
        print(f"Bot: {ans}\n")