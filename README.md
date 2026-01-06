# hola
# 📘 MEMORIA TÉCNICA DE DESARROLLO - FISHWATCH

Este documento detalla paso a paso el proceso de ingeniería realizado para construir el sistema FishWatch, desde la gestión de datos crudos hasta el despliegue de la aplicación final.

---

## 🏗️ FASE 0: Configuración del Entorno y Estructura

**Objetivo:** Establecer una base sólida y reproducible para el proyecto.

### 1. Estructura de Directorios
Se utilizó un script de automatización (`init_project.py`, ya eliminado tras su uso) para generar una arquitectura estándar de Data Science:

```
fishwatch/
├── data/           # Almacenamiento de datasets (raw, processed, splits)
├── vision/         # Scripts de entrenamiento y evaluación YOLO
├── backend/        # API FastAPI y lógica de negocio
├── nlp/            # Módulos de Procesamiento de Lenguaje Natural
├── scripts/        # Herramientas de utilidad (QA, Splits, Benchmarks)
├── static/         # Frontend (HTML/JS/CSS)
└── reports/        # Resultados, gráficos y logs
```

### 2. Gestión de Dependencias
Se definieron las bibliotecas exactas en `requirements.txt` para garantizar compatibilidad:
- **Visión:** `ultralytics` (YOLOv11), `opencv-python-headless`.
- **Backend:** `fastapi`, `uvicorn`, `sqlalchemy`.
- **NLP:** `sentence-transformers`, `transformers`.

---

## 🧹 FASE 1: Ingeniería de Datos (Data Engineering)

**Objetivo:** Transformar datos crudos y desordenados en un dataset de alta calidad para entrenamiento.

### Paso 1.1: "Zona de Aterrizaje" (Landing Zone)
- **Acción:** Se consolidaron todas las imágenes y etiquetas `.txt` provenientes de diversas fuentes (DeepFish, OzFish, etc.) en una única carpeta: `data/raw/all_data/`.
- **Resultado:** Una "sopa" de datos heterogénea lista para ser procesada.

### Paso 1.2: Control de Calidad (QA) - `scripts/validate_labels.py`
- **Código:** Se implementó un script de validación rigurosa.
- **Verificaciones:**
    1.  Existencia del par imagen-etiqueta.
    2.  Formato de coordenadas YOLO (x_center, y_center, width, height).
    3.  Normalización correcta (valores entre 0 y 1).
    4.  Dimensiones positivas (width > 0, height > 0).
- **Salida:** `reports/tables/dataset_summary.csv` con el estado de cada archivo.

### Paso 1.3: Estratificación y Splits - `scripts/make_splits.py`
- **Código:** Script para dividir el dataset validado.
- **Lógica:**
    - Se aplicó una semilla aleatoria (`SEED=42`) para reproducibilidad.
    - Distribución: **70% Train, 20% Val, 10% Test**.
    - **Corrección de Clases:** Se normalizaron todas las clases a `0` (Fish) durante la copia para evitar inconsistencias de datasets externos.
- **Resultado:** Estructura final en `data/splits/{train,val,test}/{images,labels}`.

### Paso 1.4: Validación Visual - `scripts/sample_viz.py`
- **Acción:** Generación de imágenes con *bounding boxes* dibujadas sobre una muestra aleatoria.
- **Propósito:** Verificación humana de que las etiquetas coinciden visualmente con los peces.

---

## 🧠 FASE 2: Entrenamiento del Modelo Baseline (YOLO11n)

**Objetivo:** Establecer una línea base de rendimiento con el modelo más ligero (Nano).

### Paso 2.1: Configuración - `vision/fish.yaml`
- Definición de rutas absolutas/relativas al dataset.
- Definición de clases (`nc: 1`, `names: ['fish']`).

### Paso 2.2: Entrenamiento - `vision/train.py`
- **Modelo:** YOLO11n (Nano).
- **Hiperparámetros:** `epochs=50`, `imgsz=640`, `batch=16`.
- **Salida:** Pesos guardados en `reports/runs/baseline_yolo11n/weights/best.pt`.

### Paso 2.3: Evaluación Técnica - `vision/eval.py`
- **Acción:** Evaluación del modelo entrenado sobre el conjunto de **TEST** (datos nunca vistos).
- **Métricas Generadas:**
    - mAP@0.5 (Precisión media con IoU 0.5).
    - mAP@0.5:0.95 (Métrica estricta COCO).
    - Matrices de Confusión y Curvas PR.

---

## 🚀 FASE 3: Mejora y Optimización (Challenger Model)

**Objetivo:** Superar al baseline y optimizar para inferencia en tiempo real.

### Paso 3.1: Entrenamiento Challenger - `vision/train_s.py`
- **Modelo:** YOLO11s (Small) - Mayor capacidad que el Nano.
- **Estrategia:** Comparar si el aumento de parámetros justifica la ganancia en precisión vs. la pérdida de FPS.

### Paso 3.2: Exportación y Optimización - `vision/export.py`
- **Acción:** Conversión de los modelos PyTorch (`.pt`) a formatos de inferencia optimizada.
- **Formatos:**
    - **ONNX:** Para interoperabilidad y ejecución en CPU rápida.
    - **TensorRT (.engine):** (Opcional) Para máxima velocidad en GPUs NVIDIA.

### Paso 3.3: Benchmark de Trade-offs - `scripts/bench_fps.py`
- **Código:** Script dedicado a medir rendimiento puro.
- **Prueba:** Ejecuta inferencia en bucle sobre imágenes de prueba.
- **Métricas:** FPS promedio y Latencia (ms) para cada formato (PT vs ONNX) y tamaño (Nano vs Small).
- **Resultado:** Tabla comparativa para justificar la elección del modelo final en producción.

---

## 🌐 FASE 4: Integración y Despliegue (Full Stack)

**Objetivo:** Construir la aplicación final utilizable por el usuario.

### Paso 4.1: Backend (FastAPI) - `backend/app.py`
- **API REST:** Endpoints para gestión de video y datos.
- **WebSocket:** Transmisión de video procesado y metadatos en tiempo real.
- **Base de Datos:** SQLite con SQLAlchemy para persistencia de detecciones.

### Paso 4.2: NLP (RAG) - `nlp/qa.py`
- **Implementación:** Sistema de preguntas y respuestas sobre los datos SQL.
- **Tecnología:** Sentence Transformers para detectar intención del usuario ("¿Cuántos peces hubo ayer?") y traducir a consultas SQL.

### Paso 4.3: Frontend Moderno - `static/`
- **Interfaz:** HTML5 + CSS3 + JavaScript Vanilla (sin frameworks pesados).
- **Características:**
    - Dashboard de KPIs en tiempo real.
    - Chatbot integrado para consultas NLP.
    - Visualización de video con bounding boxes.

---

## ✅ Conclusión
El sistema ha seguido un flujo de desarrollo profesional, desde la limpieza de datos hasta la optimización de modelos y despliegue web, cumpliendo con todos los requisitos de la rúbrica de evaluación.
