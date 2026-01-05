import torch
from transformers import pipeline
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend import db, models
from datetime import datetime

# --- CONFIGURACIÓN DEL MODELO ---
# Usamos un modelo GPT-2 ligero en español para dar "toque humano"
print("🧠 Cargando modelo de generación de texto (puede tardar la primera vez)...")
generator = pipeline('text-generation', model='DeepESP/gpt2-spanish', max_length=100)

def get_daily_stats(db_session: Session):
    """Extrae las métricas duras de la base de datos"""
    today = datetime.now().date()
    
    # Consultas SQL optimizadas
    total_fish = db_session.query(func.sum(models.DetectionEvent.num_fish)).scalar() or 0
    avg_fps = db_session.query(func.avg(models.DetectionEvent.fps)).scalar() or 0
    avg_conf = db_session.query(func.avg(models.DetectionEvent.avg_confidence)).scalar() or 0
    
    # Calcular Hora Pico (un poco más complejo en SQL, simplificado aquí)
    # En producción harías un GROUP BY hour, aquí simulamos con el último evento
    last_event = db_session.query(models.DetectionEvent).order_by(models.DetectionEvent.timestamp.desc()).first()
    # El timestamp está en hora local del sistema
    last_seen = last_event.timestamp.strftime("%H:%M") if last_event else "N/A"

    return {
        "date": today.strftime("%d/%m/%Y"),
        "total": total_fish,
        "fps": round(avg_fps, 1),
        "conf": round(avg_conf, 2),
        "last_seen": last_seen
    }

def generate_report():
    """Genera el reporte final usando IA + Datos"""
    session = db.SessionLocal()
    try:
        stats = get_daily_stats(session)
        
        # 1. Usar Transformer para generar una "apertura" creativa
        # Le damos un pie forzado para que empiece hablando de monitoreo
        prompt = "El sistema de monitoreo ambiental ha registrado hoy actividad importante. En resumen,"
        # Generamos texto (ajustamos randomness para que no alucine demasiado)
        intro_generated = generator(prompt, num_return_sequences=1, do_sample=True, temperature=0.7)[0]['generated_text']
        
        # Cortamos la generación en el primer punto para que sea una frase limpia
        intro_clean = intro_generated.split('.')[0] + "."

        # 2. Fusión: IA Creativa + Datos Reales (Técnica: Slot Filling)
        # Esto asegura que los números sean 100% reales (la IA a veces inventa números)
        report_body = (
            f" {intro_clean} "
            f"Hasta el momento ({stats['last_seen']}), se han detectado un total de **{stats['total']} especímenes**. "
            f"El rendimiento del sistema se mantiene estable con un promedio de {stats['fps']} FPS "
            f"y una confianza de detección del {int(stats['conf']*100)}%. "
        )
        
        # Clasificación simple de estado basada en reglas
        status = "🟢 Óptimo" if stats['fps'] > 15 else "🔴 Sobrecarga"
        
        final_report = f"""
        📋 **REPORTE AUTOMÁTICO - FISHWATCH**
        Fecha: {stats['date']}
        -------------------------------------
        {report_body}
        
        Estado del Sistema: {status}
        """
        return final_report

    finally:
        session.close()

if __name__ == "__main__":
    print(generate_report())