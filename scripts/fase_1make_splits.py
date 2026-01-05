import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Configuración
RAW_DIR = Path("data/raw/all_data")
SPLIT_DIR = Path("data/splits")
SEED = 42  # Para reproducibilidad (Rúbrica)

# Proporciones
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def create_splits():
    random.seed(SEED)
    
    # Limpiar splits anteriores si existen
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    
    # Crear estructura YOLO
    for split in ['train', 'val', 'test']:
        (SPLIT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Listar pares válidos (asumimos que ya corriste validate_labels)
    images = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png"))
    valid_pairs = []
    
    print("📦 Identificando pares válidos para split...")
    for img in images:
        txt = img.with_suffix('.txt')
        if txt.exists():
            valid_pairs.append((img, txt))
    
    # Mezclar
    random.shuffle(valid_pairs)
    
    # Calcular cortes
    total = len(valid_pairs)
    train_idx = int(total * TRAIN_RATIO)
    val_idx = train_idx + int(total * VAL_RATIO)
    
    datasets = {
        'train': valid_pairs[:train_idx],
        'val': valid_pairs[train_idx:val_idx],
        'test': valid_pairs[val_idx:]
    }
    
    print(f"📊 Distribución: Train={len(datasets['train'])}, Val={len(datasets['val'])}, Test={len(datasets['test'])}")
    
    # Copiar archivos
    for split, pairs in datasets.items():
        print(f"🚀 Generando split: {split}...")
        for img_src, txt_src in tqdm(pairs):
            # Copiar imagen
            shutil.copy(img_src, SPLIT_DIR / split / 'images' / img_src.name)
            
            # Copiar y CORREGIR label (Forzar clase 0 si es necesario)
            # Esto es vital: YOLO-Fish a veces tiene clases raras, aquí unificamos a "Pez = 0"
            dest_txt = SPLIT_DIR / split / 'labels' / txt_src.name
            with open(txt_src, 'r') as f_in, open(dest_txt, 'w') as f_out:
                lines = f_in.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # Reescribimos forzando la clase 0 (Pez)
                        # Si quieres distinguir especies, borra la linea de abajo y usa parts[0]
                        parts[0] = "0" 
                        f_out.write(" ".join(parts) + "\n")
                        
    print("\n✅ Splits generados exitosamente en 'data/splits/'")

if __name__ == "__main__":
    create_splits()