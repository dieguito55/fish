import pytest
from fastapi.testclient import TestClient
from backend.app import app
from nlp.qa import answer_question
from pathlib import Path

# Cliente de pruebas para FastAPI
client = TestClient(app)

# 1. TEST DE BACKEND (API)
def test_read_stats():
    """Verifica que el endpoint /stats/summary responda JSON correcto"""
    response = client.get("/stats/summary")
    assert response.status_code == 200
    json_data = response.json()
    assert "average_fps" in json_data
    assert "total_fish_seen_accumulated" in json_data
    # Verifica tipos de datos
    assert isinstance(json_data["total_fish_seen_accumulated"], int)

def test_export_csv():
    """Verifica que la descarga de CSV funcione"""
    response = client.get("/export/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]

# 2. TEST DE NLP (QA)
def test_qa_total_count():
    """Verifica que el chatbot entienda una pregunta básica de conteo"""
    # Pregunta difícil (typo o informal)
    question = "cuantos peses hay en total" 
    answer = answer_question(question)
    
    # La respuesta debe contener la palabra clave o el formato esperado
    # Como la respuesta depende de la DB, buscamos palabras clave de la plantilla
    assert "total" in answer.lower() or "peces" in answer.lower()

def test_qa_unknown():
    """Verifica que el chatbot admita cuando no sabe"""
    question = "quien ganó el mundial de futbol"
    answer = answer_question(question)
    assert "no entendí" in answer.lower() or "siento" in answer.lower()

# 3. TEST DE INTEGRIDAD DE DATOS (Estructura de carpetas)
def test_directory_structure():
    """Verifica que las carpetas críticas existan (Rúbrica: Reproducibilidad)"""
    assert Path("data/splits/train/images").exists()
    assert Path("reports/runs").exists()
    assert Path("vision/fish.yaml").exists()