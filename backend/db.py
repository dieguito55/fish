# fishwatch/backend/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# OPCIÓN A: PostgreSQL (Si usaste Docker)
# SQLALCHEMY_DATABASE_URL = "postgresql://admin:password123@localhost/fishwatch_db"

# OPCIÓN B: SQLite (Si Docker falla, descomenta esta y comenta la de arriba)
SQLALCHEMY_DATABASE_URL = "sqlite:///./fishwatch.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependencia para obtener la DB en cada request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()