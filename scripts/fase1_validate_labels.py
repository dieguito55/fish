import os
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Configuración
RAW_DIR = Path("data/raw/all_data")
REPORT_DIR = Path("reports/tables")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def validate_dataset():
    print(f"🔍 Iniciando validación en: {RAW_DIR}")
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(RAW_DIR.glob(ext)))
    
    print(f"📸 Imágenes encontradas: {len(image_files)}")
    
    stats = []
    valid_pairs = 0
    
    for img_path in tqdm(image_files, desc="Validando"):
        txt_path = img_path.with_suffix('.txt')
        status = "OK"
        details = ""
        
        # 1. Verificar si existe label
        if not txt_path.exists():
            status = "MISSING_LABEL"
            details = "No existe archivo .txt asociado"
            stats.append({"file": img_path.name, "status": status, "details": details})
            continue
            
        # 2. Verificar contenido del label
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                status = "EMPTY_LABEL" # Puede ser válido si es background (sin peces)
                details = "Archivo .txt vacío"
            
            # Verificar coordenadas
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    status = "FORMAT_ERROR"
                    details = f"Línea malformada: {line}"
                    break
                
                cls, x, y, w, h = map(float, parts)
                
                # Check normalización [0,1]
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    status = "COORD_ERROR"
                    details = f"Coordenadas fuera de [0,1]: {line}"
                    break
                
                # Check dimensiones positivas
                if w <= 0 or h <= 0:
                    status = "DIM_ERROR"
                    details = f"Ancho/Alto inválido: {line}"
                    break

        except Exception as e:
            status = "READ_ERROR"
            details = str(e)

        if status == "OK":
            valid_pairs += 1
            
        stats.append({"file": img_path.name, "status": status, "details": details})

    # Guardar reporte
    df = pd.DataFrame(stats)
    csv_path = REPORT_DIR / "dataset_summary.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n📊 RESUMEN DE QA")
    print("-" * 30)
    print(df['status'].value_counts())
    print("-" * 30)
    print(f"✅ Pares válidos listos para usar: {valid_pairs}")
    print(f"📄 Reporte guardado en: {csv_path}")

if __name__ == "__main__":
    validate_dataset()