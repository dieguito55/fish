from ultralytics import YOLO
import shutil
import os
from pathlib import Path

def evaluate_model():
    # 1. Cargar el MEJOR modelo del entrenamiento anterior
    model_path = Path("reports/runs/baseline_yolo11n/weights/best.pt")
    
    if not model_path.exists():
        print("❌ No se encuentra el modelo entrenado. Ejecuta train.py primero.")
        return

    print(f"⚖️ Evaluando modelo: {model_path}")
    model = YOLO(model_path)

    # 2. Validar sobre el set de TEST (Datos nunca vistos)
    # split='test' es clave para la rúbrica
    metrics = model.val(
        data=os.path.abspath("vision/fish.yaml"),
        split='test',
        project='reports/runs',
        name='eval_test',
        exist_ok=True
    )

    # 3. Extraer métricas clave para el reporte
    print("\n📊 MÉTRICAS FINALES (TEST SET):")
    print("-" * 30)
    print(f"map50 (Precisión general): {metrics.box.map50:.4f}")
    print(f"map50-95 (Precisión estricta): {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("-" * 30)

    # 4. Mover gráficos importantes a reports/figures para fácil acceso
    source_dir = Path("reports/runs/eval_test")
    dest_dir = Path("reports/figures")
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "val_batch0_labels.jpg", # Muestra de ground truth
        "val_batch0_pred.jpg"    # Muestra de predicciones
    ]

    print("\n📂 Copiando gráficos a reports/figures/...")
    for f in files_to_copy:
        src = source_dir / f
        if src.exists():
            shutil.copy(src, dest_dir / f"test_{f}")
            print(f"   ✅ Guardado: test_{f}")

if __name__ == '__main__':
    evaluate_model()