from ultralytics import YOLO
import os
import multiprocessing

def train_model_s():
    # CAMBIO 1: Usamos 'yolo11s.pt' (Small) en lugar de 'n'
    model = YOLO('yolo11s.pt') 

    yaml_path = os.path.abspath("vision/fish.yaml")
    print(f"🚀 Iniciando entrenamiento MODELO SMALL con: {yaml_path}")

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,               # Mantenemos 8 para no saturar VRAM con el modelo S
        project='reports/runs',
        name='challenger_yolo11s', # Nombre diferente para comparar
        patience=10,
        exist_ok=True,
        device=0,
        workers=1,
        verbose=True
    )
    print("✅ Entrenamiento Small finalizado.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model_s()