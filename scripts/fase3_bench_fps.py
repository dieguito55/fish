
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

# Configuración
TEST_IMG = list(Path("data/splits/test/images").glob("*.jpg"))[0] # Tomamos una imagen real
N_LOOPS = 200  # Número de inferencias para promediar
WARMUP = 20    # Inferencias iniciales para calentar la GPU

def benchmark_model(model_path, format_name):
    print(f"⚡ Benchmarking: {model_path.name} [{format_name}]")
    
    # Cargar modelo
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"   Error cargando: {e}")
        return None

    # Leer imagen
    img = cv2.imread(str(TEST_IMG))

    # Warmup (Calentamiento)
    for _ in range(WARMUP):
        model(img, verbose=False)

    # Medición
    latencies = []
    start_time = time.perf_counter()
    
    for _ in range(N_LOOPS):
        t0 = time.perf_counter()
        model(img, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # ms

    total_time = time.perf_counter() - start_time
    fps = N_LOOPS / total_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"   Resultados: {fps:.2f} FPS | Latencia: {avg_latency:.2f}ms")
    
    return {
        "Model": model_path.stem.replace("best", format_name), # Ejemplo: baseline_yolo11n_onnx
        "Format": format_name,
        "FPS": round(fps, 1),
        "Latency_ms": round(avg_latency, 2),
        "P95_Latency_ms": round(p95_latency, 2)
    }

def run_benchmarks():
    results = []
    
    # Lista de modelos a probar (ajusta rutas si es necesario)
    # 1. PyTorch (Baseline .pt)
    results.append(benchmark_model(Path("reports/runs/baseline_yolo11n/weights/best.pt"), "PyTorch"))
    
    # 2. ONNX (Optimized .onnx)
    onnx_path = Path("reports/runs/baseline_yolo11n/weights/best.onnx")
    if onnx_path.exists():
        results.append(benchmark_model(onnx_path, "ONNX"))

    # 3. TensorRT (.engine) - La joya de la corona
    engine_path = Path("reports/runs/baseline_yolo11n/weights/best.engine")
    if engine_path.exists():
        results.append(benchmark_model(engine_path, "TensorRT"))

    # Repetir para el modelo SMALL cuando termine...
    # (Agrega aquí las rutas del Small cuando existan)

    # Guardar tabla
    df = pd.DataFrame([r for r in results if r is not None])
    print("\n🏆 TABLA FINAL DE BENCHMARK")
    print(df)
    df.to_csv("reports/tables/benchmark_results.csv", index=False)

if __name__ == "__main__":
    # Forzar uso de GPU si es posible para PyTorch
    if torch.cuda.is_available():
        print(f"💻 Usando GPU: {torch.cuda.get_device_name(0)}")
    run_benchmarks()