# 🐟 FishWatch: Sistema Inteligente de Monitoreo Acuícola

![Status](https://img.shields.io/badge/Status-Terminado-green) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![YOLO](https://img.shields.io/badge/YOLO-v11-orange)

Sistema integral de visión por computadora y procesamiento de lenguaje natural para la detección, conteo y análisis de peces en tiempo real. Desarrollado como Proyecto Final de NLP.

## 🌟 Características Principales
1. **Visión por Computadora (SOTA):**
   - Detección en tiempo real usando **YOLO11s** (Small).
   - Optimización con **ONNX** (~98 FPS en RTX 5060).
   - Precisión **mAP@50: 74.4%** y QA de dataset automático.

2. **Backend & Persistencia:**
   - API RESTful con **FastAPI**.
   - Base de datos SQL para registro histórico de eventos.

3. **Dashboard Interactivo:**
   - Visualización en tiempo real con **Streamlit**.
   - Métricas de rendimiento (FPS/Latencia) y conteo histórico.

4. **Inteligencia NLP (Transformers):**
   - **Reportes Automáticos:** Generación de texto natural con GPT-2.
   - **Chatbot QA:** Responde preguntas sobre los datos usando Embeddings (Sentence-BERT).

## 🏗️ Arquitectura del Sistema
El sistema sigue un diseño modular desacoplado:
- `vision/`: Módulo de inferencia y streaming.
- `backend/`: API y gestión de base de datos.
- `dashboard/`: Interfaz de usuario.
- `nlp/`: Modelos de lenguaje para interacción humano-máquina.

## 🚀 Instalación y Uso

### 1. Configuración Inicial
```bash
# Clonar y crear entorno
git clone [https://github.com/tu-usuario/fishwatch.git](https://github.com/tu-usuario/fishwatch.git)
cd fishwatch
python -m venv venv
source venv/bin/activate  # o .\venv\Scripts\activate en Windows
pip install -r requirements.txt