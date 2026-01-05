from ultralytics import YOLO
import os

def train_model():
    # 1. Cargar modelo base
    model = YOLO('yolo11n.pt') 

    # 2. Rutas absolutas
    yaml_path = os.path.abspath("vision/fish.yaml")

    print(f"🚀 Iniciando entrenamiento con: {yaml_path}")

    # 3. Entrenar
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,               # ⬅️ Bajamos a 8 por seguridad (tu GPU aguanta 16, pero priorizamos estabilidad)
        project='reports/runs',
        name='baseline_yolo11n',
        patience=10,
        exist_ok=True,
        device=0,
        workers=1,             # ⬅️ CLAVE: En Windows, pon esto en 1 o 0. (8 crashea el sistema)
        verbose=True
    )
    
    print("✅ Entrenamiento finalizado.")
    print(f"📂 Resultados guardados en: reports/runs/baseline_yolo11n")

if __name__ == '__main__':
    # Esto es obligatorio en Windows para evitar bucles de procesos
    import multiprocessing
    multiprocessing.freeze_support()
    
    train_model()