# fishwatch/backend/models.py
from sqlalchemy import Column, Integer, Float, DateTime, String
from backend.db import Base

class DetectionEvent(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    num_fish = Column(Integer)
    avg_confidence = Column(Float)
    fps = Column(Float)
    latency = Column(Float)
    status = Column(String) # "active", "idle", etc.