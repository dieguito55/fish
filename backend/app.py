# fishwatch/backend/app.py
from __future__ import annotations

from fastapi import (
    FastAPI, Depends, HTTPException,
    UploadFile, File, WebSocket, WebSocketDisconnect, Form, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import io
import time
import asyncio
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

from backend import models, schemas, db


# =========================
# Paths robustos (NO relativos)
# =========================
BACKEND_DIR = Path(__file__).resolve().parent          # .../fishwatch/backend
PROJECT_DIR = BACKEND_DIR.parent                       # .../fishwatch
STATIC_DIR = PROJECT_DIR / "static"
UPLOAD_DIR = PROJECT_DIR / "uploads"

# Crear uploads si no existe
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 1) Crear Tablas al inicio (para endpoints legacy /events/)
models.Base.metadata.create_all(bind=db.engine)

app = FastAPI(
    title="FishWatch Live API",
    description="Backend para monitoreo de peces en tiempo real con YOLO",
    version="2.1.0"
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # en prod: pon tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Static
# =========================
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/new")
async def serve_new_interface():
    """Servir interfaz premium en /new"""
    return FileResponse(STATIC_DIR / "index_new.html")


# =========================
# Helpers
# =========================
def _safe_iso_to_dt(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None
    return None


def _parse_window(window: str) -> timedelta:
    """
    window ejemplo: '15m', '1h', '6h', '1d'
    """
    w = (window or "1h").strip().lower()
    try:
        if w.endswith("m"):
            return timedelta(minutes=int(w[:-1]))
        if w.endswith("h"):
            return timedelta(hours=int(w[:-1]))
        if w.endswith("d"):
            return timedelta(days=int(w[:-1]))
    except Exception:
        pass
    return timedelta(hours=1)


def _filter_events_since(events: List[dict], since_dt: datetime) -> List[dict]:
    out = []
    for e in events:
        dt = _safe_iso_to_dt(e.get("timestamp"))
        if dt and dt >= since_dt:
            out.append(e)
    return out


async def _broadcast(payload: dict) -> None:
    """
    Envía a todos los websockets conectados, limpiando los muertos.
    """
    dead: List[WebSocket] = []
    for ws in active_websockets:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            active_websockets.remove(ws)
        except ValueError:
            pass


# =========================
# YOLO Manager con Tracking IoU
# =========================
class YOLOManager:
    def __init__(self) -> None:
        self.model: Optional[YOLO] = None

        # Ruta robusta: si tu best.pt está en otro sitio, ajusta aquí
        # (Mantengo tu ruta, pero relativa al proyecto si existiera)
        candidate_1 = PROJECT_DIR / "reports" / "runs" / "baseline_yolo11n" / "weights" / "best.pt"
        candidate_2 = PROJECT_DIR / "reports" / "runs" / "challenger_yolo11s" / "weights" / "best.pt"

        if candidate_1.exists():
            self.model_path = str(candidate_1)
        elif candidate_2.exists():
            self.model_path = str(candidate_2)
        else:
            # fallback: lo que tú tenías
            self.model_path = "reports/runs/baseline_yolo11n/weights/best.pt"

        self.next_id = 0
        
        # Sistema de tracking
        self.active_tracks: Dict[int, Dict[str, Any]] = {}  # {track_id: {"bbox": [x1,y1,x2,y2], "frames_lost": 0}}
        self.max_frames_lost = 3  # Si un pez no se detecta en 3 frames, se elimina
        self.iou_threshold = 0.5  # Umbral de IoU aumentado para ser más estricto (50% overlap)
        
        self.load_model()

    def reset_tracking(self):
        """Resetea el sistema de tracking para un nuevo video"""
        self.active_tracks.clear()
        self.next_id = 0
        print("🔄 Tracking reseteado")

    def load_model(self) -> None:
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Modelo cargado: {self.model_path}")
            else:
                self.model = None
                print(f"⚠️ Modelo no encontrado: {self.model_path}")
        except Exception as e:
            self.model = None
            print(f"❌ Error cargando modelo: {e}")

    def get_zone(self, cx: float, cy: float, width: int, height: int) -> str:
        zones = [
            ['A', 'B', 'C'],
            ['D', 'E', 'F'],
            ['G', 'H', 'I']
        ]
        col = min(int(cx / max(width, 1) * 3), 2)
        row = min(int(cy / max(height, 1) * 3), 2)
        return zones[row][col]

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calcula IoU entre dos bboxes [x1, y1, x2, y2]
        """
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        inter_width = max(0, x2_min - x1_max)
        inter_height = max(0, y2_min - y1_max)
        inter_area = inter_width * inter_height
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    def match_detections_to_tracks(self, current_detections: List[Dict]) -> List[Dict]:
        """
        Asigna track_ids a las detecciones actuales basándose en IoU con tracks activos.
        """
        matched_tracks = set()
        
        # Para cada detección, buscar el track más cercano (mayor IoU)
        for det in current_detections:
            x, y, w, h = det["bbox"]
            det_box = [x, y, x + w, y + h]  # [x1, y1, x2, y2]
            
            best_iou = 0.0
            best_track_id = None
            
            # Comparar con todos los tracks activos
            for track_id, track_info in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue  # Ya asignado
                
                track_box = track_info["bbox"]
                iou = self.calculate_iou(det_box, track_box)
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            # Si encontramos un match, reutilizar el ID
            if best_track_id is not None:
                det["track_id"] = best_track_id
                self.active_tracks[best_track_id]["bbox"] = det_box
                self.active_tracks[best_track_id]["frames_lost"] = 0
                matched_tracks.add(best_track_id)
            else:
                # Nuevo pez detectado, asignar nuevo ID
                det["track_id"] = self.next_id
                self.active_tracks[self.next_id] = {
                    "bbox": det_box,
                    "frames_lost": 0
                }
                self.next_id += 1
        
        # Incrementar frames_lost para tracks no emparejados
        tracks_to_remove = []
        for track_id in self.active_tracks:
            if track_id not in matched_tracks:
                self.active_tracks[track_id]["frames_lost"] += 1
                if self.active_tracks[track_id]["frames_lost"] > self.max_frames_lost:
                    tracks_to_remove.append(track_id)
        
        # Eliminar tracks perdidos
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
        
        return current_detections

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, List[dict]]:
        if self.model is None:
            return frame, []

        results = self.model.predict(frame, conf=0.4, verbose=False)
        result = results[0]
        detections: List[dict] = []

        if result.boxes is None:
            return frame, detections

        h, w = frame.shape[:2]
        
        # Primera pasada: recolectar todas las detecciones
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            detections.append({
                "bbox": bbox,
                "confidence": conf,
                "class": cls,
                "track_id": -1,  # Temporal, se asignará después
                "zone": self.get_zone(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, w, h),
            })
        
        # Aplicar tracking: asignar IDs basados en IoU
        detections = self.match_detections_to_tracks(detections)
        
        # Dibujar con IDs persistentes
        colors = np.array([
            [0, 255, 255],   # Amarillo
            [255, 0, 255],   # Magenta
            [0, 255, 0],     # Verde
            [255, 255, 0],   # Cyan
            [255, 128, 0],   # Naranja
            [128, 0, 255],   # Púrpura
            [0, 128, 255],   # Azul claro
            [255, 0, 128]    # Rosa
        ])
        
        for det in detections:
            x, y, w_box, h_box = det["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w_box), int(y + h_box)
            track_id = det["track_id"]
            conf = det["confidence"]
            
            # Color consistente basado en track_id
            color = tuple(map(int, colors[track_id % len(colors)]))
            
            # Dibujar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Etiqueta con ID y confianza
            label_text = f"ID:{track_id} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame, detections


yolo_manager = YOLOManager()

# =========================
# In-memory store
# =========================
event_queue: deque = deque(maxlen=5000)   # cada evento = 1 pez detectado (según tu lógica actual)
active_websockets: List[WebSocket] = []


# =========================
# Frontend
# =========================
@app.get("/")
async def root():
    index_new = STATIC_DIR / "index_new.html"
    index = STATIC_DIR / "index.html"
    if index_new.exists():
        return FileResponse(str(index_new))
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(status_code=404, detail="No se encontró static/index_new.html ni static/index.html")


@app.get("/new")
async def new_interface():
    index_new = STATIC_DIR / "index_new.html"
    if index_new.exists():
        return FileResponse(str(index_new))
    raise HTTPException(status_code=404, detail="No se encontró static/index_new.html")


# =========================
# Health
# =========================
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "version": "2.1.0",
        "model_loaded": yolo_manager.model is not None,
        "timestamp": datetime.now().isoformat()
    }


# =========================
# Detections (manual push)
# =========================
@app.post("/api/detections")
async def create_detection(detection: dict):
    detection["id"] = f"{int(time.time() * 1000)}"
    detection["timestamp"] = datetime.now().isoformat()
    event_queue.append(detection)
    await _broadcast(detection)
    return {"success": True, "id": detection["id"]}


@app.get("/api/detections/latest")
def get_latest_detections(limit: int = 50):
    return list(event_queue)[-max(1, min(limit, 5000)) :]


@app.delete("/api/detections/clear")
def clear_detections():
    event_queue.clear()
    return {"success": True, "message": "Detecciones eliminadas"}


@app.delete("/api/db/clear")
def clear_database():
    """Elimina todos los registros de la base de datos"""
    try:
        db_session = db.SessionLocal()
        deleted_count = db_session.query(models.DetectionEvent).delete()
        db_session.commit()
        db_session.close()
        print(f"🗑️ Base de datos limpiada: {deleted_count} registros eliminados")
        return {"success": True, "message": f"Se eliminaron {deleted_count} registros", "deleted": deleted_count}
    except Exception as e:
        print(f"❌ Error limpiando base de datos: {e}")
        return {"success": False, "message": str(e)}


# =========================
# Metrics
# =========================
@app.get("/api/metrics/summary")
def get_metrics_summary(window: str = Query("1h", description="Ej: 15m, 1h, 6h, 1d")):
    events_list = list(event_queue)
    if not events_list:
        return {"totalToday": 0, "perMinute": 0, "peakHour": "--:--", "avgConfidence": 0, "avgFps": 0, "avgLatency": 0}

    now = datetime.now()

    # total hoy
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_events = _filter_events_since(events_list, today_start)

    # ventana para promedios + perMinute
    delta = _parse_window(window)
    since = now - delta
    recent_events = _filter_events_since(events_list, since)

    minutes = max(delta.total_seconds() / 60.0, 1.0)
    per_minute = len(recent_events) / minutes if recent_events else 0

    # hora pico (del día)
    hour_counts: Dict[int, int] = {}
    for e in today_events:
        dt = _safe_iso_to_dt(e.get("timestamp"))
        if not dt:
            continue
        hour_counts[dt.hour] = hour_counts.get(dt.hour, 0) + 1
    peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0

    # promedios (ventana)
    def _avg(key: str) -> float:
        vals = [float(e.get(key, 0) or 0) for e in recent_events]
        return sum(vals) / len(vals) if vals else 0.0

    avg_conf = _avg("confidence") * 100.0
    avg_fps = _avg("fps")
    avg_lat = _avg("latency")

    return {
        "totalToday": len(today_events),
        "perMinute": round(per_minute, 2),
        "peakHour": f"{peak_hour:02d}:00",
        "avgConfidence": round(avg_conf, 1),
        "avgFps": round(avg_fps, 1),
        "avgLatency": round(avg_lat, 1),
    }


@app.get("/api/metrics/zones")
def get_zone_metrics(window: str = Query("1h", description="Ej: 15m, 1h, 6h, 1d")):
    events_list = list(event_queue)
    if not events_list:
        return {}

    now = datetime.now()
    since = now - _parse_window(window)
    recent_events = _filter_events_since(events_list, since)

    zone_counts: Dict[str, int] = {}
    for e in recent_events:
        zone = e.get("zone", "A")
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    return zone_counts


# =========================
# Frame detection (webcam / frame-by-frame)
# =========================
@app.post("/api/detect/frame")
async def detect_frame(
    frame: UploadFile = File(...),
    frame_number: int = Form(0),
    return_frame: bool = Query(False, description="Si true, devuelve jpeg base64 del frame anotado"),
    auto_save: bool = Form(False)
):
    start_time = time.time()
    
    # Log para debugging
    if frame_number % 100 == 0:
        print(f"📊 Frame {frame_number}: auto_save={auto_save}")

    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo leer el frame")

    annotated_frame, detections = yolo_manager.detect(img)

    process_time = time.time() - start_time
    fps = (1.0 / process_time) if process_time > 0 else 0.0
    latency_ms = process_time * 1000.0

    detections_with_frame: List[dict] = []
    unique_tracks = set()
    for det in detections:
        track_id = det.get("track_id", 0)
        unique_tracks.add(track_id)
        
        detection_event = {
            "id": f"{int(time.time() * 1000)}_{frame_number}_{track_id}",
            "timestamp": datetime.now().isoformat(),
            "zone": det["zone"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "track_id": track_id,
            "source": "video" if frame_number > 0 else "webcam",
            "fps": fps,
            "latency": latency_ms,
            "frame": frame_number,
        }
        event_queue.append(detection_event)
        detections_with_frame.append(detection_event)

        await _broadcast(detection_event)

    # Auto-guardar cada 100 frames si auto_save=True
    if auto_save and frame_number > 0 and frame_number % 100 == 0 and len(event_queue) > 0:
        print(f"🔍 Frame {frame_number}: Intentando auto-guardado... (auto_save={auto_save}, queue_size={len(event_queue)})")
        try:
            db_session = db.SessionLocal()
            
            # Contar peces únicos
            unique_all = set()
            total_conf = 0.0
            total_fps = 0.0
            total_lat = 0.0
            
            for event in event_queue:
                tid = event.get("track_id")
                if tid is not None:
                    unique_all.add(tid)
                total_conf += event.get("confidence", 0)
                total_fps += event.get("fps", 0)
                total_lat += event.get("latency", 0)
            
            count = len(event_queue)
            avg_conf = total_conf / count if count > 0 else 0
            avg_fps = total_fps / count if count > 0 else 0
            avg_lat = total_lat / count if count > 0 else 0
            
            # Guardar en BD
            local_time = datetime.now()
            db_event = models.DetectionEvent(
                timestamp=local_time,
                num_fish=len(unique_all),
                avg_confidence=avg_conf,
                fps=avg_fps,
                latency=avg_lat,
                status="processing"
            )
            db_session.add(db_event)
            db_session.commit()
            db_session.close()
            print(f"💾 Auto-guardado en frame {frame_number}: {len(unique_all)} peces únicos")
        except Exception as e:
            print(f"⚠️ Error en auto-guardado: {e}")

    response: Dict[str, Any] = {
        "success": True,
        "count": len(detections_with_frame),
        "unique_fish": len(unique_tracks),
        "fps": fps,
        "latency": latency_ms,
        "detections": detections_with_frame,
    }

    if return_frame:
        ok, jpg = cv2.imencode(".jpg", annotated_frame)
        if ok:
            import base64
            response["frame_b64"] = base64.b64encode(jpg.tobytes()).decode("utf-8")

    return response


# Nuevo endpoint para guardar sesión en BD
@app.post("/api/session/save")
async def save_session(db_sess: Session = Depends(db.get_db)):
    """Guarda la sesión actual en la base de datos"""
    try:
        if len(event_queue) == 0:
            return {"success": False, "message": "No hay eventos para guardar"}
        
        # Contar peces únicos
        unique_tracks = set()
        total_conf = 0.0
        total_fps = 0.0
        total_lat = 0.0
        
        for event in event_queue:
            track_id = event.get("track_id")
            if track_id is not None:
                unique_tracks.add(track_id)
            total_conf += event.get("confidence", 0)
            total_fps += event.get("fps", 0)
            total_lat += event.get("latency", 0)
        
        count = len(event_queue)
        avg_conf = total_conf / count if count > 0 else 0
        avg_fps = total_fps / count if count > 0 else 0
        avg_lat = total_lat / count if count > 0 else 0
        
        # Crear registro en BD
        local_time = datetime.now()
        db_event = models.DetectionEvent(
            timestamp=local_time,
            num_fish=len(unique_tracks),
            avg_confidence=avg_conf,
            fps=avg_fps,
            latency=avg_lat,
            status="completed"
        )
        db_sess.add(db_event)
        db_sess.commit()
        db_sess.refresh(db_event)
        
        print(f"✅ Sesión guardada en BD: {len(unique_tracks)} peces únicos (ID: {db_event.id})")
        print(f"   Timestamp: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            "success": True,
            "message": f"Sesión guardada: {len(unique_tracks)} peces únicos",
            "event_id": db_event.id,
            "unique_fish": len(unique_tracks),
            "total_detections": count
        }
    except Exception as e:
        print(f"⚠️ Error guardando sesión: {e}")
        db_sess.rollback()
        return {"success": False, "error": str(e)}


# =========================
# Video upload & processing
# =========================
@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename vacío")

    # Guardar en uploads/
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # RESET tracking para nuevo video
    yolo_manager.reset_tracking()

    # IMPORTANT: devolvemos 'video_path' (lo que el front debe mandar a /api/video/process)
    return {
        "success": True,
        "filename": file.filename,
        "video_path": str(file_path),
        "path": str(file_path),  # compat con tu respuesta anterior
        "message": "Video subido correctamente"
    }


class VideoProcessRequest(BaseModel):
    video_path: str
    fps_limit: int = 30


@app.post("/api/video/process")
async def process_video(request: VideoProcessRequest):
    video_path = request.video_path
    fps_limit = request.fps_limit

    # compat: si te mandan "uploads/xxx.mp4" relativo, lo resolvemos al PROJECT_DIR
    vp = Path(video_path)
    if not vp.is_absolute():
        vp = (PROJECT_DIR / vp).resolve()

    if not vp.exists():
        raise HTTPException(status_code=404, detail=f"Video no encontrado: {vp}")

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="No se pudo abrir el video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    # RESET tracking antes de procesar
    yolo_manager.reset_tracking()
    print(f"🎬 Iniciando procesamiento de video con {total_frames} frames")

    detections_summary: List[dict] = []
    frame_count = 0
    tracked_fish: Dict[int, int] = {}
    db_session = db.SessionLocal()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        start_time = time.time()
        _, detections = yolo_manager.detect(frame)
        process_time = time.time() - start_time

        if detections:
            for det in detections:
                track_id = int(det.get("track_id", 0))
                tracked_fish[track_id] = tracked_fish.get(track_id, 0) + 1

                detection_event = {
                    "id": f"{int(time.time() * 1000)}_{frame_count}",
                    "timestamp": datetime.now().isoformat(),
                    "zone": det["zone"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "track_id": track_id,
                    "source": str(vp),
                    "fps": (1.0 / process_time) if process_time > 0 else 0.0,
                    "latency": process_time * 1000.0,
                    "frame": frame_count,
                }
                event_queue.append(detection_event)
                detections_summary.append(detection_event)
                await _broadcast(detection_event)

        # (opcional) limitar fps de procesamiento
        if fps_limit and fps_limit > 0:
            target = 1.0 / fps_limit
            if process_time < target:
                await asyncio.sleep(target - process_time)

    cap.release()

    zone_counts: Dict[str, int] = {}
    for det in detections_summary:
        z = det.get("zone", "A")
        zone_counts[z] = zone_counts.get(z, 0) + 1

    # GUARDAR EN BASE DE DATOS
    try:
        avg_conf = sum(d["confidence"] for d in detections_summary) / len(detections_summary) if detections_summary else 0
        avg_fps = sum(d["fps"] for d in detections_summary) / len(detections_summary) if detections_summary else 0
        avg_lat = sum(d["latency"] for d in detections_summary) / len(detections_summary) if detections_summary else 0

        # Usar hora local del sistema
        local_time = datetime.now()
        db_event = models.DetectionEvent(
            timestamp=local_time,
            num_fish=len(tracked_fish),  # Peces únicos
            avg_confidence=avg_conf,
            fps=avg_fps,
            latency=avg_lat,
            status="completed"
        )
        db_session.add(db_event)
        db_session.commit()
        db_session.refresh(db_event)
        print(f"✅ Guardado en BD: {len(tracked_fish)} peces únicos (ID evento: {db_event.id})")
        print(f"   Timestamp: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"⚠️ Error guardando en BD: {e}")
        db_session.rollback()
    finally:
        db_session.close()

    return {
        "success": True,
        "total_frames": total_frames,
        "processed_frames": frame_count,
        "total_detections": len(detections_summary),
        "unique_fish": len(tracked_fish),
        "video_fps": video_fps,
        "zone_distribution": zone_counts,
        "detections": detections_summary[-100:],
    }


# =========================
# WebSocket realtime
# =========================
@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        # No obligamos al cliente a enviar textos.
        # Enviamos pings periódicos para mantener viva la conexión.
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_json({"type": "ping", "ts": datetime.now().isoformat()})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            active_websockets.remove(websocket)
        except ValueError:
            pass


# =========================
# Export
# =========================
@app.get("/api/export/csv")
def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "zone", "confidence", "bbox", "source", "id", "fps", "latency", "track_id", "frame"])

    for e in event_queue:
        writer.writerow([
            e.get("timestamp"),
            e.get("zone"),
            e.get("confidence"),
            str(e.get("bbox")),
            e.get("source"),
            e.get("id"),
            e.get("fps"),
            e.get("latency"),
            e.get("track_id"),
            e.get("frame"),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fishwatch_detections.csv"}
    )


@app.get("/api/export/json")
def export_json():
    return JSONResponse(content=list(event_queue))


# =========================
# NLP
# =========================
class NLPQuestionRequest(BaseModel):
    question: str
    context_window: str = "today"


@app.post("/api/nlp/ask")
async def nlp_ask(request: NLPQuestionRequest):
    try:
        from nlp import qa
        answer = qa.answer_question(request.question)
        return {"answer": answer, "success": True}
    except Exception as e:
        print(f"⚠️ Error en NLP QA: {e}")
        # Fallback: respuestas basadas en event_queue
        q = (request.question or "").lower()

        if "total" in q or "cuánto" in q or "cuantos" in q:
            # Contar peces únicos por track_id
            unique_tracks = set()
            for e in event_queue:
                track_id = e.get("track_id")
                if track_id is not None:
                    unique_tracks.add(track_id)
            count = len(unique_tracks) if unique_tracks else len(event_queue)
            return {"answer": f"Se han detectado {count} peces únicos en el video.", "success": True}

        if "zona" in q:
            zone_counts: Dict[str, int] = {}
            for e in event_queue:
                z = e.get("zone", "A")
                zone_counts[z] = zone_counts.get(z, 0) + 1
            if zone_counts:
                top_zone = max(zone_counts.items(), key=lambda x: x[1])
                return {"answer": f"La zona más activa es {top_zone[0]} con {top_zone[1]} detecciones.", "success": True}

        return {"answer": "No puedo responder esa pregunta en este momento.", "success": False}


@app.post("/api/nlp/summary")
async def nlp_summary():
    # Intentar generar reporte desde BD primero
    try:
        from nlp import report
        generated_report = report.generate_report()
        return {"summary": generated_report, "success": True, "source": "database"}
    except Exception as e:
        print(f"⚠️ Error generando reporte desde BD: {e}")
        # Fallback: usar event_queue
        total = len(event_queue)
        if total == 0:
            return {"summary": "No hay detecciones registradas.", "success": True, "source": "memory"}

        # Contar peces únicos
        unique_tracks = set()
        zone_counts: Dict[str, int] = {}
        total_conf = 0.0
        for e in event_queue:
            track_id = e.get("track_id")
            if track_id is not None:
                unique_tracks.add(track_id)
            
            z = e.get("zone", "A")
            zone_counts[z] = zone_counts.get(z, 0) + 1
            total_conf += float(e.get("confidence", 0) or 0)

        unique_fish = len(unique_tracks) if unique_tracks else 0
        top_zone = max(zone_counts.items(), key=lambda x: x[1]) if zone_counts else ("A", 0)
        avg_conf = (total_conf / total) * 100.0 if total > 0 else 0.0

        summary = (
            "📊 Resumen de Detecciones:\n\n"
            f"• Peces únicos detectados: {unique_fish}\n"
            f"• Total de detecciones: {total}\n"
            f"• Zona más activa: {top_zone[0]} ({top_zone[1]} detecciones)\n"
            f"• Confianza promedio: {avg_conf:.1f}%\n"
            "• Estado del sistema: Operativo ✅\n"
        )
        return {"summary": summary, "success": True, "source": "memory"}


# =========================
# Database Stats
# =========================
@app.get("/api/db/stats")
def get_database_stats(db_sess: Session = Depends(db.get_db)):
    """Obtiene estadísticas de la base de datos"""
    try:
        total_events = db_sess.query(func.count(models.DetectionEvent.id)).scalar() or 0
        total_fish = db_sess.query(func.sum(models.DetectionEvent.num_fish)).scalar() or 0
        avg_fps = db_sess.query(func.avg(models.DetectionEvent.fps)).scalar() or 0
        avg_conf = db_sess.query(func.avg(models.DetectionEvent.avg_confidence)).scalar() or 0
        
        latest = db_sess.query(models.DetectionEvent).order_by(models.DetectionEvent.timestamp.desc()).first()
        
        # Agregar información de zona horaria local
        import time
        if time.daylight:
            tz_offset = -time.altzone / 3600
        else:
            tz_offset = -time.timezone / 3600
        
        # Formatear timestamp con nota de hora local
        latest_ts = None
        latest_ts_formatted = None
        if latest:
            latest_ts = latest.timestamp.isoformat()
            latest_ts_formatted = latest.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "success": True,
            "total_events": total_events,
            "total_fish": int(total_fish),
            "avg_fps": round(float(avg_fps), 2),
            "avg_confidence": round(float(avg_conf), 2),
            "latest_timestamp": latest_ts,
            "latest_timestamp_formatted": latest_ts_formatted,
            "timezone_offset": f"UTC{tz_offset:+.1f}",
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "has_data": total_events > 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "has_data": False
        }


@app.get("/api/db/detections")
def get_all_detections(
    limit: int = Query(1000, description="Número máximo de registros a retornar"),
    offset: int = Query(0, description="Offset para paginación"),
    db_sess: Session = Depends(db.get_db)
):
    """Obtiene todas las detecciones de la base de datos con paginación"""
    try:
        # Query para obtener eventos de detección
        total_count = db_sess.query(func.count(models.DetectionEvent.id)).scalar() or 0
        
        events_query = db_sess.query(models.DetectionEvent)\
            .order_by(models.DetectionEvent.timestamp.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()
        
        # Convertir a formato compatible con el frontend
        # Como no tenemos coordenadas individuales, usamos datos agregados
        detections = []
        for i, event in enumerate(events_query):
            # Generar múltiples "detecciones" por cada evento según num_fish
            for fish_idx in range(event.num_fish):
                detections.append({
                    "id": event.id * 100 + fish_idx,  # ID único por pez
                    "timestamp": event.timestamp.isoformat(),
                    "zone": chr(65 + (fish_idx % 3)),  # Rotar entre A, B, C
                    "confidence": event.avg_confidence,
                    "x1": 100 + (fish_idx * 50) % 500,  # Coordenadas simuladas
                    "y1": 100 + (fish_idx * 30) % 400,
                    "x2": 150 + (fish_idx * 50) % 500,
                    "y2": 150 + (fish_idx * 30) % 400,
                    "frame_number": event.id
                })
        
        return {
            "success": True,
            "total_count": total_count,
            "returned_count": len(detections),
            "limit": limit,
            "offset": offset,
            "detections": detections
        }
    except Exception as e:
        print(f"❌ Error obteniendo detecciones: {e}")
        return {
            "success": False,
            "error": str(e),
            "detections": []
        }


# =========================
# LEGACY (DB)
# =========================
@app.post("/events/", response_model=schemas.DetectionResponse)
def create_event(event: schemas.DetectionCreate, db_sess: Session = Depends(db.get_db)):
    db_event = models.DetectionEvent(**event.model_dump())
    db_sess.add(db_event)
    db_sess.commit()
    db_sess.refresh(db_event)
    return db_event


@app.get("/runtime/latest")
def get_latest_runtime(db_sess: Session = Depends(db.get_db)):
    latest = db_sess.query(models.DetectionEvent).order_by(models.DetectionEvent.id.desc()).first()
    if not latest:
        return {"fps": 0, "latency": 0, "status": "offline"}
    return {"fps": latest.fps, "latency": latest.latency, "timestamp": latest.timestamp}


@app.get("/stats/summary")
def get_stats(db_sess: Session = Depends(db.get_db)):
    avg_fps = db_sess.query(func.avg(models.DetectionEvent.fps)).scalar()
    total_detections = db_sess.query(func.sum(models.DetectionEvent.num_fish)).scalar()
    max_fish = db_sess.query(func.max(models.DetectionEvent.num_fish)).scalar()

    return {
        "average_fps": round(float(avg_fps), 2) if avg_fps else 0,
        "total_fish_seen_accumulated": int(total_detections) if total_detections else 0,
        "max_fish_in_frame": int(max_fish) if max_fish else 0
    }
