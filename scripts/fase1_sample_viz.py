import cv2
import random
from pathlib import Path
import matplotlib.pyplot as plt

# Usamos el set de validación ya procesado
DATA_DIR = Path("data/splits/val")
OUTPUT_DIR = Path("reports/figures/qa_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def visualize_samples(num_samples=9):
    img_dir = DATA_DIR / 'images'
    label_dir = DATA_DIR / 'labels'
    
    images = list(img_dir.glob("*.jpg"))
    if not images:
        print("⚠️ No se encontraron imágenes en data/splits/val")
        return

    samples = random.sample(images, min(num_samples, len(images)))
    
    print(f"🎨 Generando {len(samples)} muestras visuales...")
    
    for i, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        txt_path = label_dir / img_path.with_suffix('.txt').name
        
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cls, cx, cy, bw, bh = map(float, line.split())
                    
                    # Des-normalizar coordenadas YOLO a pixeles
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    # Dibujar caja
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "Fish", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Guardar resultado
        out_path = OUTPUT_DIR / f"sample_{i}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"   Guardado: {out_path}")

if __name__ == "__main__":
    visualize_samples()