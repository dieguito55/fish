# fishwatch/backend/schemas.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

# Lo que recibimos del YOLO
class DetectionCreate(BaseModel):
    timestamp: datetime
    num_fish: int
    avg_confidence: float
    fps: float
    latency: float
    status: str = "ok"

# Lo que devolvemos al usuario
class DetectionResponse(DetectionCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)