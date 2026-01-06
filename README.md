🐟 FishWatch
Sistema Inteligente de Monitoreo Acuícola con Visión Computacional y NLP

FishWatch es un sistema full-stack que integra Visión Computacional (YOLO) y Procesamiento de Lenguaje Natural (NLP) para la detección, conteo y análisis de peces en tiempo real, permitiendo la interacción con los datos mediante lenguaje natural.

👥 Equipo de Desarrollo
Integrante	Rol Técnico
Cristian Ticona Márquez	Product Manager & System Architect
Vanessa Castro Callo	UX/UI Engineer & Visualization Specialist
Jorge Olarte Quispe	Data Engineer & Dataset Curator
Jhon Marco Aracayo Mamani	NLP Engineer & Intelligent Systems
Juan Diego Canaza Paucara	Computer Vision Engineer & ML Deployment Lead
📋 Gestión del Proyecto

🔗 Tablero Trello (gestión y evidencias del proyecto):
👉 https://trello.com/invite/b/695cdbdbb31b19be8675d7f7/ATTI49cd47f841957f1d26851861ca3cfb91C333854D/fish-nlp

🎯 Objetivo del Proyecto

Desarrollar un sistema inteligente capaz de:

Detectar peces en tiempo real.

Almacenar métricas históricas.

Permitir consultas en lenguaje natural.

Generar reportes automáticos comprensibles para usuarios no técnicos.

🏗️ Arquitectura General
fishwatch/
├── data/           # Datasets (raw, processed, splits)
├── vision/         # Entrenamiento, evaluación y exportación YOLO
├── backend/        # API FastAPI y lógica de negocio
├── nlp/            # NLP: chatbot QA y reportes automáticos
├── scripts/        # QA, splits y benchmarks
├── static/         # Frontend (Dashboard)
└── reports/        # Resultados y métricas

🧹 Pipeline de Datos

Consolidación de datasets heterogéneos.

Validación automática de etiquetas YOLO.

Splits reproducibles (70% train / 20% val / 10% test, seed=42).

Validación visual de anotaciones.

🧠 Visión Computacional

Modelos YOLOv11 (Nano y Small).

Evaluación con métricas estándar:

mAP@0.5

mAP@0.5:0.95

Optimización para inferencia:

PyTorch → ONNX → TensorRT

Benchmark de FPS vs precisión para selección del modelo final.

💬 NLP y Chatbot Inteligente

Sentence-Transformers (Sentence-BERT) para detección de intención.

Arquitectura híbrida tipo RAG:

El NLP interpreta la pregunta.

Los datos reales se obtienen vía SQL.

Evita alucinaciones al no generar valores numéricos.

📊 Reportes Automáticos

Uso de LLM (GPT-2 en español) solo para redacción.

Técnica de Slot Filling:

Texto generado por IA.

Métricas insertadas directamente desde la base de datos.

Reportes legibles para toma de decisiones.

🌐 Aplicación Web

Backend: FastAPI + SQLAlchemy.

Frontend: Dashboard interactivo (HTML, CSS, JavaScript).

Video en tiempo real con detecciones.

KPIs y filtros históricos.

Chatbot integrado.

🛠️ Tecnologías Utilizadas

Visión: Ultralytics YOLO, OpenCV

NLP: Sentence-Transformers, HuggingFace Transformers

Backend: FastAPI, SQLAlchemy

Optimización: ONNX, TensorRT

Base de Datos: SQLite / PostgreSQL

✅ Estado del Proyecto

✔ Desarrollo completo
✔ Evaluación técnica realizada
✔ Optimización y despliegue funcional
✔ Documentación y gestión en Trello