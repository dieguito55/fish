#!/usr/bin/env python3
"""
Script para limpiar la base de datos y empezar de nuevo
"""
from backend import db, models
from pathlib import Path

def reset_database():
    """Elimina todos los registros de la base de datos"""
    session = db.SessionLocal()
    try:
        # Contar registros antes
        count_before = session.query(models.DetectionEvent).count()
        print(f"📊 Registros actuales en la BD: {count_before}")
        
        if count_before == 0:
            print("✅ La base de datos ya está vacía")
            return
        
        # Confirmar
        print("\n⚠️  ADVERTENCIA: Esto eliminará todos los registros de detección.")
        response = input("¿Deseas continuar? (si/no): ").strip().lower()
        
        if response not in ['si', 's', 'yes', 'y']:
            print("❌ Operación cancelada")
            return
        
        # Eliminar todos los registros
        session.query(models.DetectionEvent).delete()
        session.commit()
        
        print(f"✅ Se eliminaron {count_before} registros")
        print("✅ Base de datos limpia")
        print("\n📝 Ahora procesa un nuevo video para llenar la BD con timestamps correctos")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        session.rollback()
    finally:
        session.close()

def show_database_info():
    """Muestra información de la base de datos"""
    db_path = Path("fishwatch.db")
    
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"\n💾 Base de datos: {db_path}")
        print(f"   Tamaño: {size_mb:.2f} MB")
    else:
        print(f"\n💾 No existe la base de datos en: {db_path}")

if __name__ == "__main__":
    print("="*60)
    print("🗑️  LIMPIAR BASE DE DATOS - FISHWATCH")
    print("="*60)
    
    show_database_info()
    print()
    reset_database()
    
    print("\n" + "="*60)
    print("Para empezar de nuevo:")
    print("1. cd fishwatch")
    print("2. Abre http://localhost:8000/new")
    print("3. Sube y procesa un nuevo video")
    print("4. Los timestamps ahora estarán correctos")
    print("="*60)
