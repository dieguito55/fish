<div align="center">

# 🐟 FishWatch  
### Sistema Inteligente de Monitoreo Acuícola  
**Visión Computacional · NLP · Full-Stack**

</div>

---

## 📌 Descripción General

**FishWatch** es un sistema *full-stack* que integra **Visión Computacional (YOLO)** y **Procesamiento de Lenguaje Natural (NLP)** para la **detección, conteo y análisis de peces en tiempo real**, permitiendo a los usuarios interactuar con los datos mediante **lenguaje natural**.

El proyecto está diseñado bajo principios de **reproducibilidad, eficiencia y precisión**, alineado a estándares académicos y profesionales.

---

## 👥 Equipo de Desarrollo

| Integrante | Rol Técnico |
|---------|-----------|
| **Cristian Ticona Márquez** | *Product Manager & System Architect* |
| **Vanessa Castro Callo** | *UX/UI Engineer & Visualization Specialist* |
| **Jorge Olarte Quispe** | *Data Engineer & Dataset Curator* |
| **Jhon Marco Aracayo Mamani** | *NLP Engineer & Intelligent Systems* |
| **Juan Diego Canaza Paucara** | *Computer Vision Engineer & ML Deployment Lead* |

---

## 🗂️ Gestión del Proyecto

🔗 **Tablero Trello (planificación, evidencias y backlog):**  
👉 [FishWatch – Trello Board](https://trello.com/invite/b/695cdbdbb31b19be8675d7f7/ATTI49cd47f841957f1d26851861ca3cfb91C333854D/fish-nlp)

---

## 🎯 Objetivo del Proyecto

Desarrollar un sistema inteligente capaz de:

- 🟢 Detectar peces en tiempo real  
- 🟢 Almacenar métricas históricas confiables  
- 🟢 Permitir consultas en lenguaje natural  
- 🟢 Generar reportes automáticos interpretables  

---

## 🏗️ Arquitectura del Proyecto

```text
fishwatch/
├── data/           # Datasets (raw, processed, splits)
├── vision/         # Entrenamiento, evaluación y exportación YOLO
├── backend/        # API FastAPI y lógica de negocio
├── nlp/            # Chatbot QA y reportes inteligentes
├── scripts/        # QA, splits y benchmarks
├── static/         # Frontend (Dashboard)
└── reports/        # Métricas, gráficos y resultados
