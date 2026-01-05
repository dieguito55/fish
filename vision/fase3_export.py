from ultralytics import YOLO
from pathlib import Path

def export_models():
    # Definir rutas de los pesos entrenados
    # Nota: El path del 's' existirá cuando termine el entrenamiento del Paso 1
    models_to_export = [
        ("Nano", Path("reports/runs/baseline_yolo11n/weights/best.pt")),
        ("Small", Path("reports/runs/challenger_yolo11s/weights/best.pt")) # Este fallará si no ha terminado train_s.py
    ]

    for name, path in models_to_export:
        if not path.exists():
            print(f"⚠️ Saltando {name}: No se encuentra {path}")
            continue
            
        print(f"\n📦 Exportando {name} ({path})...")
        model = YOLO(path)
        
        # 1. Exportar a ONNX (Interoperabilidad - CPU/GPU genérico)
        # opset=12 es muy compatible
        model.export(format='onnx', opset=12, dynamic=False)
        print(f"   ✅ {name} exportado a ONNX")

        # 2. Exportar a TensorRT (.engine) -> ESTO ES ORO PARA LA RÚBRICA
        # Solo funciona si tienes GPU NVIDIA configurada correctamente
        try:
            model.export(format='engine', device=0, half=True) # half=True usa FP16 (más rápido)
            print(f"   🚀 {name} exportado a TensorRT (Engine)")
        except Exception as e:
            print(f"   ❌ No se pudo exportar a Engine (¿Falta TensorRT?): {e}")

if __name__ == "__main__":
    export_models()