// ============================================
// FISHWATCH LIVE - MAIN APPLICATION (FIXED)
// Compatible con index_new.html + backend /api/*
// ============================================

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
  API_BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000',

  MODE: 'real',
  AUTO_RECONNECT: true,
  RECONNECT_DELAY: 3000,

  MAX_EVENTS: 5000,
  CHART_UPDATE_INTERVAL: 800, // ms

  ZONES: ['A','B','C','D','E','F','G','H','I'],
};

const State = {
  currentView: 'camera',
  isStreaming: false,
  apiConnected: false,
  websocket: null,
  currentMode: 'video', // video por defecto

  events: [], // eventos llegados por websocket (o por endpoints)
  detectionsByFrame: new Map(), // frame -> array detections
  currentVideoDetections: [],
  uploadedFile: null,
  uploadedVideoPath: null,
  videoFps: 30,

  videoElement: null,
  overlayCanvas: null,
  overlayCtx: null,
  detectionInterval: null,

  pipCanvas: null,
  pipCtx: null,
  pipInterval: null,

  charts: {
    timeseries: null,
    zones: null,
    confidence: null
  },

  heatmapCounts: {
    A:0,B:0,C:0,D:0,E:0,F:0,G:0,H:0,I:0
  },

  threshold: 0.4,
  showBoxes: true
};

// ============================================
// INIT
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  console.log('🐟 FishWatch Live - Fixed Frontend');

  // cargar settings guardados
  const savedApiUrl = localStorage.getItem('fishwatch_api_url');
  if (savedApiUrl) {
    CONFIG.API_BASE_URL = savedApiUrl;
    // WS_URL default: mismo host/puerto (cambia si lo necesitas)
    CONFIG.WS_URL = savedApiUrl.replace('http://', 'ws://').replace('https://', 'wss://');
  }

  // refs
  State.videoElement = document.getElementById('videoElement');
  State.overlayCanvas = document.getElementById('overlayCanvas');
  State.overlayCtx = State.overlayCanvas.getContext('2d');

  State.pipCanvas = document.getElementById('pipCanvas');
  State.pipCtx = State.pipCanvas ? State.pipCanvas.getContext('2d') : null;

  const apiUrlInput = document.getElementById('apiUrlInput');
  if (apiUrlInput) apiUrlInput.value = CONFIG.API_BASE_URL;

  // listeners de video
  wireVideoListeners();

  // init UI
  initializeCharts();
  initializeHeatmapGrid();
  checkAPIConnection();

  // mostrar view inicial
  switchView('camera');

  console.log('✅ Front inicializado');
});

function wireVideoListeners() {
  const video = State.videoElement;
  if (!video) return;

  // Actualiza timeline y tiempos
  video.addEventListener('timeupdate', () => {
    updateTimelineUI();
    // si estás scrubeando/pausado, igual dibuja
    processVideoFrame(false);
    drawPiP();
  });

  video.addEventListener('loadedmetadata', () => {
    // Ajustar overlay size
    syncOverlaySizeToVideo();
    updateTimelineUI(true);

    // Mostrar total duration
    const totalTime = document.getElementById('totalTime');
    if (totalTime) totalTime.textContent = formatTime(video.duration || 0);

    // ocultar placeholder + mostrar controles
    const placeholder = document.getElementById('videoPlaceholder');
    if (placeholder) placeholder.style.display = 'none';

    const controls = document.getElementById('videoPlaybackControls');
    if (controls) controls.style.display = 'flex';
  });

  video.addEventListener('ended', () => {
    stopDetectionLoop();
    const btn = document.getElementById('playPauseBtn');
    if (btn) btn.innerHTML = '<i class="fas fa-play"></i>';
    
    // Auto-guardar cuando el video termine
    if (State.events.length > 0) {
      console.log('Video terminado, guardando sesión automáticamente...');
      setTimeout(() => saveSessionToDB(), 1000);
    }
  });

  // Si cambias de tamaño (fullscreen, etc.)
  window.addEventListener('resize', () => {
    syncOverlaySizeToVideo();
    drawPiP();
  });
}

function syncOverlaySizeToVideo() {
  const video = State.videoElement;
  if (!video || !State.overlayCanvas) return;
  if (video.videoWidth === 0 || video.videoHeight === 0) return;

  State.overlayCanvas.width = video.videoWidth;
  State.overlayCanvas.height = video.videoHeight;
}

// ============================================
// API
// ============================================
const API = {
  async healthCheck() {
    try {
      const r = await fetch(`${CONFIG.API_BASE_URL}/health`);
      return r.ok;
    } catch {
      return false;
    }
  },

  async uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);

    const r = await fetch(`${CONFIG.API_BASE_URL}/api/video/upload`, {
      method: 'POST',
      body: formData
    });
    if (!r.ok) throw new Error('Upload failed');
    return await r.json();
  },

  async askNLP(question) {
    try {
      const r = await fetch(`${CONFIG.API_BASE_URL}/api/nlp/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      return await r.json();
    } catch {
      return { answer: 'Error al procesar la pregunta' };
    }
  },

  async exportJSON() {
    const r = await fetch(`${CONFIG.API_BASE_URL}/api/export/json`);
    if (!r.ok) throw new Error('Export JSON failed');
    const blob = await r.blob();
    downloadBlob(blob, `fishwatch_export_${Date.now()}.json`);
  },

  async exportCSV() {
    const r = await fetch(`${CONFIG.API_BASE_URL}/api/export/csv`);
    if (!r.ok) throw new Error('Export CSV failed');
    const blob = await r.blob();
    downloadBlob(blob, `fishwatch_export_${Date.now()}.csv`);
  }
};

// ============================================
// WS
// ============================================
function initializeWebSocket() {
  // ✅ Si ya está abierto, no hagas nada (evita open/close/open/close)
  if (State.websocket && State.websocket.readyState === WebSocket.OPEN) {
    return;
  }

  // ✅ Si está conectando, tampoco lo mates
  if (State.websocket && State.websocket.readyState === WebSocket.CONNECTING) {
    return;
  }

  // ✅ Si estaba CLOSING/CLOSED, recién cerramos (por limpieza) y creamos uno nuevo
  if (State.websocket) {
    try { State.websocket.close(); } catch {}
    State.websocket = null;
  }

  console.log('🔌 Conectando WS...');
  State.websocket = new WebSocket(`${CONFIG.WS_URL}/ws/detections`);

  State.websocket.onopen = () => {
    console.log('✅ WS conectado');
    updateConnectionStatus(true);
  };

  State.websocket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      // ignore pings
      if (msg && msg.type === 'ping') return;

      handleNewDetection(msg);

      // cache por frame si existe
      if (msg && msg.frame !== undefined) {
        const frame = msg.frame;
        if (!State.detectionsByFrame.has(frame)) State.detectionsByFrame.set(frame, []);
        State.detectionsByFrame.get(frame).push(msg);
      }
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  State.websocket.onerror = (err) => {
    console.warn('⚠️ WebSocket no disponible (opcional)', err);
    updateConnectionStatus(true); // Mantener API como conectada
    // No es crítico, el sistema funciona sin WS
  };

  State.websocket.onclose = (evt) => {
    console.log('🔌 WS cerrado', evt?.code, evt?.reason || '');
    
    // No intentar reconectar si es error 404 (endpoint no existe)
    if (evt?.code === 1006 || evt?.code === 404) {
      console.log('💡 WebSocket no disponible, continuando sin él');
      State.websocket = null;
      return;
    }

    // ✅ Reconectar solo si el usuario está "streaming"
    // y si realmente está cerrado (readyState=CLOSED)
    if (CONFIG.AUTO_RECONNECT && State.isStreaming) {
      setTimeout(() => {
        // Evita reconectar si ya conectó mientras tanto
        if (!State.websocket || State.websocket.readyState === WebSocket.CLOSED) {
          initializeWebSocket();
        }
      }, CONFIG.RECONNECT_DELAY);
    }
  };
}


// ============================================
// CONNECTION STATUS UI
// ============================================
async function checkAPIConnection() {
  const connected = await API.healthCheck();
  State.apiConnected = connected;
  updateConnectionStatus(connected);

  if (!connected) {
    showToast('Backend no disponible', 'error');
  }
}

function updateConnectionStatus(connected) {
  // sidebar
  const apiStatus = document.getElementById('apiStatus');
  if (apiStatus) {
    const dot = apiStatus.querySelector('.status-dot');
    const span = apiStatus.querySelector('span');
    if (dot) dot.className = `status-dot ${connected ? 'online' : 'offline'}`;
    if (span) span.textContent = connected ? 'API Online' : 'API Offline';
  }

  // topbar
  const liveStatus = document.getElementById('liveStatus');
  if (liveStatus) {
    const statusSpan = liveStatus.querySelector('span');
    if (statusSpan) statusSpan.textContent = connected ? 'ONLINE' : 'OFFLINE';
    liveStatus.className = `status-badge ${connected ? 'online' : 'offline'}`;
  }
}

// ============================================
// VIEW SWITCH
// ============================================
function switchView(view) {
  State.currentView = view;

  // nav active por data-view
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.querySelector(`.nav-item[data-view="${view}"]`);
  if (activeBtn) activeBtn.classList.add('active');

  // views: clase .view
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  const section = document.getElementById(`${view}View`);
  if (section) section.classList.add('active');

  // title
  const viewTitle = document.getElementById('viewTitle');
  if (viewTitle) {
    viewTitle.textContent =
      view === 'camera' ? 'Cámara en Vivo' :
      view === 'dashboard' ? 'Dashboard de Acuicultura' :
      'Histórico';
  }
  
  // Refresh dashboard when switching to it
  if (view === 'dashboard') {
    refreshDashboard();
  }

  // PiP - Mostrar cuando NO estás en cámara pero hay video reproduciéndose
  const pip = document.getElementById('pipPreview');
  const video = State.videoElement;
  if (pip && video) {
    if (view !== 'camera' && State.currentMode === 'video' && !video.paused) {
      pip.style.display = 'block';
      startPipLoop();
      console.log('📺 PiP activado - Video continúa en background');
    } else {
      pip.style.display = 'none';
      stopPipLoop();
    }
  }
  
  // La detección continúa automáticamente si el video está reproduciéndose
  // No detenemos startDetectionLoop() al cambiar de vista

  // refrescar
  updateMetricsUI();
  renderRecentDetections();
  renderTopZones();
  renderEventsTable();
  updateCharts();
  updateHeatmapGrid();
}

// ============================================
// VIDEO UPLOAD
// ============================================
function handleFileUpload(event) {
  const fileInput = event.target;
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  State.uploadedFile = file;
  State.uploadedVideoPath = null;

  // limpiar cache/estado
  State.detectionsByFrame.clear();
  State.currentVideoDetections = [];
  State.events = [];
  resetHeatmapCounts();

  // UI: nombre archivo
  const infoBox = document.getElementById('uploadedVideoInfo');
  const fileName = document.getElementById('uploadedFileName');
  if (infoBox) infoBox.style.display = 'block';
  if (fileName) fileName.textContent = `Archivo: ${file.name}`;

  // cargar local en <video> (preview inmediato)
  const url = URL.createObjectURL(file);
  State.videoElement.src = url;

  // subir al servidor
  uploadVideo(file);
}

async function uploadVideo(file) {
  if (State.uploading) return;        // ✅ evita doble
  State.uploading = true;

  try {
    showToast('Subiendo video al servidor...', 'info');
    const result = await API.uploadVideo(file);

    State.uploadedVideoPath = result.video_path || result.path || null;

    if (!State.uploadedVideoPath) {
      showToast('Upload OK, pero no llegó video_path/path', 'error');
      return;
    }

    showToast('Video subido. Presiona Play para detectar en vivo.', 'success');

    if (!State.websocket || State.websocket.readyState !== WebSocket.OPEN) {
      initializeWebSocket();
    }
  } catch (e) {
    console.error(e);
    showToast('Error al subir el video', 'error');
  } finally {
    State.uploading = false;          // ✅ libera lock
  }
}


// ============================================
// PLAYBACK / DETECTION LOOP
// ============================================
function togglePlayPause() {
  const video = State.videoElement;
  const btn = document.getElementById('playPauseBtn');

  if (State.currentMode === 'video' && !State.uploadedVideoPath) {
    showToast('Primero selecciona y sube un video', 'error');
    return;
  }

  if (video.paused) {
    video.play();
    if (btn) btn.innerHTML = '<i class="fas fa-pause"></i>';

    State.isStreaming = true;

    // asegurar WS
    if (!State.websocket || State.websocket.readyState !== WebSocket.OPEN) {
      initializeWebSocket();
    }

    // loop detección (cada 120ms)
    startDetectionLoop();
  } else {
    video.pause();
    if (btn) btn.innerHTML = '<i class="fas fa-play"></i>';
    stopDetectionLoop();
  }
}

function startDetectionLoop() {
  if (State.detectionInterval) return;

  State.detectionInterval = setInterval(() => {
    // detectar solo si reproduce
    const v = State.videoElement;
    if (!v || v.paused || v.ended) return;
    detectCurrentFrame();
    processVideoFrame(false);
    drawPiP();
  }, 120);
}

function stopDetectionLoop() {
  if (State.detectionInterval) {
    clearInterval(State.detectionInterval);
    State.detectionInterval = null;
  }
  State.isStreaming = false;
}

async function detectCurrentFrame() {
  const video = State.videoElement;
  if (!video || video.paused || video.ended) return;
  if (video.videoWidth === 0 || video.videoHeight === 0) return;

  const currentFrame = Math.floor(video.currentTime * State.videoFps);

  // si ya está cacheado, no vuelvas a pedir
  if (State.detectionsByFrame.has(currentFrame)) return;

  // capturar frame
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(async (blob) => {
    if (!blob) return;

    const formData = new FormData();
    formData.append('frame', blob, 'frame.jpg');
    formData.append('frame_number', String(currentFrame));
    formData.append('auto_save', '1'); // Habilitar auto-guardado (1=true)

    try {
      const r = await fetch(`${CONFIG.API_BASE_URL}/api/detect/frame`, {
        method: 'POST',
        body: formData
      });

      if (!r.ok) return;

      const data = await r.json();

      // backend devuelve { success, count, fps, latency, detections: [...] }
      const dets = Array.isArray(data) ? data : (data.detections || []);

      // cachea por frame
      State.detectionsByFrame.set(currentFrame, dets);

      // si vinieron dets, dibuja inmediato
      if (dets.length > 0) {
        processVideoFrame(true);
      }

      // HUD de performance (si viene)
      updatePerfHUD(data.fps, data.latency);

    } catch (e) {
      console.error('Error detectCurrentFrame:', e);
    }
  }, 'image/jpeg', 0.85);
}

function processVideoFrame(force = false) {
  const video = State.videoElement;
  if (!video || video.videoWidth === 0) return;

  // permitir dibujar incluso pausado (scrub), pero no si no hay metadata
  if (video.ended) return;

  syncOverlaySizeToVideo();

  // determinar frame
  const currentFrame = Math.floor(video.currentTime * State.videoFps);

  // usar detecciones de frames cercanos para estabilidad visual
  const detections = [];
  const seen = new Set();

  for (let offset = -2; offset <= 2; offset++) {
    const f = currentFrame + offset;
    if (State.detectionsByFrame.has(f)) {
      const arr = State.detectionsByFrame.get(f) || [];
      for (const det of arr) {
        const key = det.id || det.track_id || JSON.stringify(det.bbox);
        if (!seen.has(key)) {
          detections.push(det);
          seen.add(key);
        }
      }
    }
  }

  State.currentVideoDetections = detections;

  // dibujar cajas
  drawBoundingBoxes(detections);

  // HUD conteo/conf
  updateVideoHUD(detections);
}

function drawBoundingBoxes(detections) {
  if (!State.overlayCanvas || !State.overlayCtx) return;

  const ctx = State.overlayCtx;
  ctx.clearRect(0, 0, State.overlayCanvas.width, State.overlayCanvas.height);

  if (!State.showBoxes) return;
  if (!detections || detections.length === 0) return;

  // Colores consistentes para cada ID
  const colors = [
    '#00ffff',  // Amarillo/Cyan
    '#ff00ff',  // Magenta
    '#00ff00',  // Verde
    '#ffff00',  // Amarillo brillante
    '#ff8000',  // Naranja
    '#8000ff',  // Púrpura
    '#0080ff',  // Azul claro
    '#ff0080'   // Rosa
  ];

  detections.forEach((det, idx) => {
    if (!det.bbox || det.bbox.length < 4) return;
    const [x, y, w, h] = det.bbox;

    // Usar track_id para color consistente
    const trackId = det.track_id !== undefined ? det.track_id : idx;
    const color = colors[trackId % colors.length];

    // Dibujar bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    const confPct = ((det.confidence || 0) * 100).toFixed(0);
    const label = `ID:${trackId} ${confPct}%`;

    ctx.font = 'bold 16px Inter';
    const tw = ctx.measureText(label).width;

    // Fondo del label
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 26), tw + 12, 24);

    // Texto del label
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 6, Math.max(16, y - 8));
  });
}

// ============================================
// VIDEO CONTROLS (timeline etc.)
// ============================================
function updateTimelineUI(force = false) {
  const video = State.videoElement;
  if (!video || !isFinite(video.duration) || video.duration <= 0) return;

  const timeline = document.getElementById('videoTimeline');
  const currentTime = document.getElementById('currentTime');
  const totalTime = document.getElementById('totalTime');

  if (currentTime) currentTime.textContent = formatTime(video.currentTime);
  if (totalTime) totalTime.textContent = formatTime(video.duration);

  if (timeline) {
    const percent = (video.currentTime / video.duration) * 100;
    // no “pisar” si el usuario está arrastrando? (simple)
    timeline.value = String(Math.max(0, Math.min(100, percent)));
  }
}

function seekVideo(value) {
  const video = State.videoElement;
  if (!video || !isFinite(video.duration) || video.duration <= 0) return;

  const v = Number(value);
  const percent = isFinite(v) ? Math.max(0, Math.min(100, v)) : 0;
  video.currentTime = (percent / 100) * video.duration;

  // dibuja aunque esté pausado
  processVideoFrame(true);
  drawPiP();
  updateTimelineUI(true);
}

function rewindVideo() {
  const video = State.videoElement;
  if (!video) return;
  video.currentTime = Math.max(0, video.currentTime - 10);
  processVideoFrame(true);
  drawPiP();
}

function forwardVideo() {
  const video = State.videoElement;
  if (!video) return;
  video.currentTime = Math.min(video.duration || 0, video.currentTime + 10);
  processVideoFrame(true);
  drawPiP();
}

function previousFrame() {
  const video = State.videoElement;
  if (!video) return;
  const dt = 1 / (State.videoFps || 30);
  video.currentTime = Math.max(0, video.currentTime - dt);
  processVideoFrame(true);
  drawPiP();
}

function nextFrame() {
  const video = State.videoElement;
  if (!video) return;
  const dt = 1 / (State.videoFps || 30);
  video.currentTime = Math.min(video.duration || 0, video.currentTime + dt);
  processVideoFrame(true);
  drawPiP();
}

// ============================================
// TOGGLES / HUD
// ============================================
function updateThreshold(value) {
  const v = Number(value);
  State.threshold = isFinite(v) ? v : State.threshold;
  const el = document.getElementById('thresholdValue');
  if (el) el.textContent = String(State.threshold.toFixed(2));
}

function setPlaybackSpeed(speed) {
  const video = document.getElementById('videoElement');
  if (video && video.src) {
    video.playbackRate = speed;
    document.getElementById('currentSpeed').textContent = speed + 'x';
    
    // Update active button
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.classList.remove('active');
      if (btn.textContent === speed + 'x') {
        btn.classList.add('active');
      }
    });
    
    console.log(`⏱️ Velocidad ajustada a ${speed}x`);
  }
}

async function clearDatabase() {
  if (!confirm('¿Estás seguro de que deseas eliminar TODOS los registros de la base de datos? Esta acción no se puede deshacer.')) {
    return;
  }
  
  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/db/clear`, {
      method: 'DELETE'
    });
    
    const data = await response.json();
    
    if (data.success) {
      alert(`✅ Base de datos limpiada: ${data.deleted} registros eliminados`);
      console.log('🗑️', data.message);
    } else {
      alert('❌ Error al limpiar la base de datos: ' + data.message);
    }
  } catch (error) {
    console.error('❌ Error:', error);
    alert('❌ Error de conexión al limpiar la base de datos');
  }
}

function toggleBoxes() {
  const cb = document.getElementById('showBoxes');
  State.showBoxes = !!(cb && cb.checked);
  processVideoFrame(true);
}

function updateVideoHUD(detections) {
  const count = detections.length;
  const avgConf = count > 0
    ? (detections.reduce((s, d) => s + (d.confidence || 0), 0) / count) * 100
    : 0;

  const hudCount = document.getElementById('hudCount');
  const hudConfidence = document.getElementById('hudConfidence');
  const hudFps = document.getElementById('hudFps');
  const pipCount = document.getElementById('pipCount');

  if (hudCount) hudCount.textContent = String(count);
  if (hudConfidence) hudConfidence.textContent = `${avgConf.toFixed(0)}%`;
  if (pipCount) pipCount.textContent = String(count);

  // fps mostrado en hudFps se alimenta con updatePerfHUD si viene del backend
  if (hudFps && !hudFps.textContent) hudFps.textContent = '0';
  
  // Update aquaculture monitoring sidebar
  updateAquaMonitoring(detections);
}

function updateAquaMonitoring(detections) {
  const count = detections.length;
  const avgConf = count > 0
    ? (detections.reduce((s, d) => s + (d.confidence || 0), 0) / count) * 100
    : 0;
  
  // Live count
  const liveCount = document.getElementById('liveCount');
  if (liveCount) liveCount.textContent = String(count);
  
  // Live confidence
  const liveConfidence = document.getElementById('liveConfidence');
  if (liveConfidence) liveConfidence.textContent = `${avgConf.toFixed(1)}%`;
  
  // Count by zones
  const zoneCounts = { A: 0, B: 0, C: 0 };
  detections.forEach(det => {
    const zone = det.zone || 'A';
    if (zoneCounts[zone] !== undefined) {
      zoneCounts[zone]++;
    }
  });
  
  // Find top zone
  let topZone = 'A';
  let maxCount = 0;
  for (const zone in zoneCounts) {
    if (zoneCounts[zone] > maxCount) {
      maxCount = zoneCounts[zone];
      topZone = zone;
    }
  }
  
  const liveTopZone = document.getElementById('liveTopZone');
  if (liveTopZone) {
    const zoneName = { A: 'Superior', B: 'Central', C: 'Inferior' }[topZone] || topZone;
    liveTopZone.textContent = `Zona ${zoneName}`;
  }
  
  // Update zone bars
  const maxZoneCount = Math.max(zoneCounts.A, zoneCounts.B, zoneCounts.C, 1);
  
  const zoneA = document.getElementById('zoneA');
  const zoneACount = document.getElementById('zoneACount');
  if (zoneA) zoneA.style.width = `${(zoneCounts.A / maxZoneCount) * 100}%`;
  if (zoneACount) zoneACount.textContent = String(zoneCounts.A);
  
  const zoneB = document.getElementById('zoneB');
  const zoneBCount = document.getElementById('zoneBCount');
  if (zoneB) zoneB.style.width = `${(zoneCounts.B / maxZoneCount) * 100}%`;
  if (zoneBCount) zoneBCount.textContent = String(zoneCounts.B);
  
  const zoneC = document.getElementById('zoneC');
  const zoneCCount = document.getElementById('zoneCCount');
  if (zoneC) zoneC.style.width = `${(zoneCounts.C / maxZoneCount) * 100}%`;
  if (zoneCCount) zoneCCount.textContent = String(zoneCounts.C);
}

function updatePerfHUD(fps, latencyMs) {
  const f = (fps ?? 0);
  const l = (latencyMs ?? 0);

  const topFps = document.getElementById('topbarFps');
  const topLat = document.getElementById('topbarLatency');
  const hudFps = document.getElementById('hudFps');

  if (topFps) topFps.textContent = String(Number(f).toFixed(1));
  if (topLat) topLat.textContent = String(Number(l).toFixed(0));
  if (hudFps) hudFps.textContent = String(Number(f).toFixed(1));
}

// ============================================
// DETECTION HANDLING (desde WS)
// ============================================
function handleNewDetection(detection) {
  // guardar
  State.events.push(detection);
  if (State.events.length > CONFIG.MAX_EVENTS) State.events.shift();

  // heatmap counts
  if (detection.zone && State.heatmapCounts[detection.zone] !== undefined) {
    State.heatmapCounts[detection.zone] += 1;
  }

  // UI
  updateMetricsUI();
  renderRecentDetections();
  renderTopZones();
  renderEventsTable();
  updateCharts();
  updateHeatmapGrid();
}

// ============================================
// METRICS UI (KPIs dashboard)
// ============================================
function updateMetricsUI() {
  const now = Date.now();

  // total hoy (en este front: total de eventos en memoria)
  const totalToday = State.events.length;

  // per minute
  const lastMinute = State.events.filter(e => {
    const t = new Date(e.timestamp || Date.now()).getTime();
    return (now - t) <= 60000;
  }).length;

  // peak hour (del día) aproximado con lo que hay en memoria
  const hourCounts = {};
  for (const e of State.events) {
    const dt = new Date(e.timestamp || Date.now());
    const h = dt.getHours();
    hourCounts[h] = (hourCounts[h] || 0) + 1;
  }
  let peakHour = '--:--';
  const entries = Object.entries(hourCounts);
  if (entries.length > 0) {
    entries.sort((a,b) => b[1]-a[1]);
    peakHour = `${String(entries[0][0]).padStart(2,'0')}:00`;
  }

  // promedios
  let avgConfidence = 0, avgFps = 0, avgLatency = 0;
  if (State.events.length > 0) {
    avgConfidence = (State.events.reduce((s,e) => s + (e.confidence||0), 0) / State.events.length) * 100;
    avgFps = State.events.reduce((s,e) => s + (e.fps||0), 0) / State.events.length;
    avgLatency = State.events.reduce((s,e) => s + (e.latency||0), 0) / State.events.length;
  }

  // KPIs dashboard
  const kpiTotalToday = document.getElementById('kpiTotalToday');
  const kpiPerMinute = document.getElementById('kpiPerMinute');
  const kpiPeakHour = document.getElementById('kpiPeakHour');
  const kpiAvgConf = document.getElementById('kpiAvgConf');
  const kpiAvgFps = document.getElementById('kpiAvgFps');
  const kpiAvgLatency = document.getElementById('kpiAvgLatency');

  if (kpiTotalToday) kpiTotalToday.textContent = String(totalToday);
  if (kpiPerMinute) kpiPerMinute.textContent = String(lastMinute.toFixed ? lastMinute.toFixed(1) : lastMinute);
  if (kpiPeakHour) kpiPeakHour.textContent = peakHour;
  if (kpiAvgConf) kpiAvgConf.textContent = `${avgConfidence.toFixed(1)}%`;
  if (kpiAvgFps) kpiAvgFps.textContent = `${avgFps.toFixed(1)}`;
  if (kpiAvgLatency) kpiAvgLatency.textContent = `${avgLatency.toFixed(0)}ms`;
}

// ============================================
// CAMERA SIDEBAR RENDER
// ============================================
function renderRecentDetections() {
  const container = document.getElementById('recentDetections');
  if (!container) return;

  const recent = State.events.slice(-12).reverse();

  if (recent.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <i class="fas fa-inbox"></i>
        <p>Sin detecciones recientes</p>
      </div>`;
    return;
  }

  container.innerHTML = recent.map(ev => {
    const time = new Date(ev.timestamp || Date.now()).toLocaleTimeString();
    const conf = ((ev.confidence || 0) * 100).toFixed(0);
    return `
      <div class="event-item" style="border-left:3px solid #00ffff;">
        <div class="event-header">
          <span class="event-zone">Zona ${ev.zone || '-'}</span>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-details">
          <span>Conf: ${conf}%</span>
          <span>Track #${ev.track_id ?? '-'}</span>
        </div>
      </div>`;
  }).join('');
}

function renderTopZones() {
  const container = document.getElementById('topZones');
  if (!container) return;

  // conteo por zonas con lo acumulado
  const counts = {};
  for (const z of CONFIG.ZONES) counts[z] = 0;
  for (const ev of State.events) {
    if (ev.zone && counts[ev.zone] !== undefined) counts[ev.zone]++;
  }

  const sorted = Object.entries(counts).sort((a,b) => b[1]-a[1]).slice(0,5);
  const max = Math.max(1, ...sorted.map(x => x[1]));

  container.innerHTML = sorted.map(([z, c]) => {
    const pct = (c / max) * 100;
    return `
      <div class="zone-item">
        <span class="zone-label">Zona ${z}</span>
        <div class="zone-bar">
          <div class="zone-fill" style="width:${pct.toFixed(0)}%"></div>
        </div>
        <span class="zone-count">${c}</span>
      </div>`;
  }).join('');
}

// ============================================
// DASHBOARD EVENTS TABLE
// ============================================
function renderEventsTable() {
  const tbody = document.getElementById('eventsTableBody');
  if (!tbody) return;

  const rows = State.events.slice(-40).reverse();
  if (rows.length === 0) {
    tbody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6"><i class="fas fa-inbox"></i> Sin eventos registrados</td>
      </tr>`;
    return;
  }

  tbody.innerHTML = rows.map(ev => {
    const ts = new Date(ev.timestamp || Date.now()).toLocaleString();
    const conf = ((ev.confidence || 0) * 100).toFixed(0) + '%';
    const bbox = ev.bbox ? `[${ev.bbox.map(n => Number(n).toFixed(0)).join(', ')}]` : '-';
    const trackId = ev.track_id !== undefined ? `ID:${ev.track_id}` : '-';
    return `
      <tr>
        <td>${ts}</td>
        <td>${ev.zone || '-'}</td>
        <td>${conf}</td>
        <td style="max-width:240px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${bbox}</td>
        <td>${ev.source || '-'}</td>
        <td><strong>${trackId}</strong></td>
      </tr>`;
  }).join('');
}

// ============================================
// CHARTS (IDs correctos)
// ============================================
function initializeCharts() {
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' } },
      x: { grid: { color: 'rgba(255,255,255,0.1)' } }
    }
  };

  const tsCanvas = document.getElementById('chartTimeseries');
  if (tsCanvas) {
    State.charts.timeseries = new Chart(tsCanvas.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Detecciones', data: [], tension: 0.35, fill: true }] },
      options: common
    });
  }

  const zonesCanvas = document.getElementById('chartZones');
  if (zonesCanvas) {
    State.charts.zones = new Chart(zonesCanvas.getContext('2d'), {
      type: 'bar',
      data: { labels: CONFIG.ZONES, datasets: [{ label: 'Por zona', data: new Array(9).fill(0) }] },
      options: common
    });
  }

  const confCanvas = document.getElementById('chartConfidence');
  if (confCanvas) {
    State.charts.confidence = new Chart(confCanvas.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels: ['Alta (>80%)', 'Media (60-80%)', 'Baja (<60%)'],
        datasets: [{ data: [0,0,0] }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  // pequeño timer para refrescar series en dashboard
  setInterval(() => {
    if (State.currentView === 'dashboard') updateCharts();
  }, CONFIG.CHART_UPDATE_INTERVAL);
}

function updateCharts() {
  // timeseries: detecciones por minuto (ventana 20 puntos)
  if (State.charts.timeseries) {
    const now = new Date();
    const label = now.toLocaleTimeString();

    const nowMs = Date.now();
    const lastMinute = State.events.filter(e => {
      const t = new Date(e.timestamp || nowMs).getTime();
      return (nowMs - t) <= 60000;
    }).length;

    State.charts.timeseries.data.labels.push(label);
    State.charts.timeseries.data.datasets[0].data.push(lastMinute);

    if (State.charts.timeseries.data.labels.length > 20) {
      State.charts.timeseries.data.labels.shift();
      State.charts.timeseries.data.datasets[0].data.shift();
    }
    State.charts.timeseries.update('none');
  }

  // zones
  if (State.charts.zones) {
    const counts = new Array(9).fill(0);
    for (const e of State.events) {
      const idx = CONFIG.ZONES.indexOf(e.zone);
      if (idx !== -1) counts[idx]++;
    }
    State.charts.zones.data.datasets[0].data = counts;
    State.charts.zones.update('none');
  }

  // confidence buckets
  if (State.charts.confidence) {
    let high=0, mid=0, low=0;
    for (const e of State.events) {
      const c = (e.confidence || 0) * 100;
      if (c > 80) high++;
      else if (c >= 60) mid++;
      else low++;
    }
    State.charts.confidence.data.datasets[0].data = [high, mid, low];
    State.charts.confidence.update('none');
  }
}

// ============================================
// HEATMAP GRID (div 3x3)
// ============================================
function initializeHeatmapGrid() {
  const grid = document.getElementById('heatmapGrid');
  if (!grid) return;

  // crea 3x3
  grid.innerHTML = '';
  for (let i=0;i<9;i++) {
    const z = CONFIG.ZONES[i];
    const cell = document.createElement('div');
    cell.className = 'heatmap-cell';
    cell.dataset.zone = z;
    cell.innerHTML = `
      <div class="hm-zone">Zona ${z}</div>
      <div class="hm-count">0</div>
    `;
    grid.appendChild(cell);
  }

  updateHeatmapGrid();
}

function resetHeatmapCounts() {
  for (const z of CONFIG.ZONES) State.heatmapCounts[z] = 0;
}

function updateHeatmapGrid() {
  const grid = document.getElementById('heatmapGrid');
  if (!grid) return;

  const max = Math.max(1, ...Object.values(State.heatmapCounts));

  grid.querySelectorAll('.heatmap-cell').forEach(cell => {
    const z = cell.dataset.zone;
    const c = State.heatmapCounts[z] || 0;
    const intensity = c / max; // 0..1

    const countEl = cell.querySelector('.hm-count');
    if (countEl) countEl.textContent = String(c);

    // estilo inline simple (si tu CSS no lo tiene)
    cell.style.background = `rgba(0, 255, 255, ${0.08 + intensity * 0.35})`;
    cell.style.border = '1px solid rgba(255,255,255,0.08)';
    cell.style.borderRadius = '12px';
    cell.style.padding = '10px';
    cell.style.minHeight = '70px';
    cell.style.display = 'flex';
    cell.style.flexDirection = 'column';
    cell.style.justifyContent = 'space-between';
  });
}

// ============================================
// EXPORT
// ============================================
async function exportJSON() {
  try {
    if (State.apiConnected) {
      await API.exportJSON();
      showToast('JSON exportado desde backend', 'success');
      return;
    }
  } catch {}

  // fallback local
  const json = JSON.stringify(State.events, null, 2);
  downloadText(json, `fishwatch_export_${Date.now()}.json`, 'application/json');
  showToast('JSON exportado localmente', 'success');
}

async function exportCSV() {
  try {
    if (State.apiConnected) {
      await API.exportCSV();
      showToast('CSV exportado desde backend', 'success');
      return;
    }
  } catch {}

  // fallback local
  const csv = convertToCSV(State.events);
  downloadText(csv, `fishwatch_export_${Date.now()}.csv`, 'text/csv');
  showToast('CSV exportado localmente', 'success');
}

function convertToCSV(data) {
  if (!data || data.length === 0) return '';
  const headers = ['timestamp', 'zone', 'confidence', 'track_id', 'source', 'id'];
  const rows = data.map(e => ([
    e.timestamp || '',
    e.zone || '',
    ((e.confidence || 0) * 100).toFixed(2),
    e.track_id ?? '',
    e.source || '',
    e.id || ''
  ]));
  return [headers, ...rows].map(r => r.join(',')).join('\n');
}

// ============================================
// CHATBOT
// ============================================
function toggleChatbot() {
  const panel = document.getElementById('chatbotPanel');
  if (panel) panel.classList.toggle('active');
}

function sendChatMessage() {
  const input = document.getElementById('chatInput');
  if (!input) return;
  const question = input.value.trim();
  if (!question) return;

  addChatMessage(question, 'user');
  input.value = '';
  setTimeout(() => processQuestion(question), 300);
}

function askQuestion(question) {
  addChatMessage(question, 'user');
  setTimeout(() => processQuestion(question), 300);
}

async function processQuestion(question) {
  if (State.apiConnected) {
    const result = await API.askNLP(question);
    addChatMessage(result.answer || 'Sin respuesta', 'bot');
  } else {
    addChatMessage('Backend no conectado para NLP.', 'bot');
  }
}

function addChatMessage(text, sender) {
  const box = document.getElementById('chatbotMessages');
  if (!box) return;

  const div = document.createElement('div');
  div.className = `chat-message ${sender}`;
  div.innerHTML = `
    <i class="fas fa-${sender === 'bot' ? 'robot' : 'user'}"></i>
    <p>${escapeHtml(text)}</p>
  `;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function handleChatKeypress(event) {
  if (event.key === 'Enter') sendChatMessage();
}

// ============================================
// SETTINGS
// ============================================
function toggleSettings() {
  const modal = document.getElementById('settingsModal');
  if (modal) modal.classList.toggle('active');
}

function saveSettings() {
  const apiUrl = document.getElementById('apiUrlInput')?.value?.trim();
  const autoReconnect = document.getElementById('autoReconnect')?.checked ?? true;

  if (apiUrl) {
    CONFIG.API_BASE_URL = apiUrl;
    CONFIG.WS_URL = apiUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    localStorage.setItem('fishwatch_api_url', apiUrl);
  }
  CONFIG.AUTO_RECONNECT = autoReconnect;

  toggleSettings();
  showToast('Configuración guardada', 'success');
  checkAPIConnection();

  // re-conectar ws si ya estaba
  if (State.websocket && State.websocket.readyState === WebSocket.OPEN) {
    initializeWebSocket();
  }
}

// ============================================
// PIP
// ============================================
function startPipLoop() {
  if (State.pipInterval) return;
  State.pipInterval = setInterval(() => drawPiP(), 250);
}

function stopPipLoop() {
  if (State.pipInterval) {
    clearInterval(State.pipInterval);
    State.pipInterval = null;
  }
}

function drawPiP() {
  if (!State.pipCanvas || !State.pipCtx) return;
  if (!State.videoElement || State.videoElement.videoWidth === 0) return;

  const w = State.pipCanvas.width = 420;
  const h = State.pipCanvas.height = 236;

  // dibujar frame del video
  try {
    State.pipCtx.drawImage(State.videoElement, 0, 0, w, h);
    
    // Dibujar detecciones en el PiP
    const video = State.videoElement;
    const currentFrame = Math.floor(video.currentTime * State.videoFps);
    const detections = State.detectionsByFrame.get(currentFrame) || [];
    
    if (detections.length > 0) {
      const scaleX = w / video.videoWidth;
      const scaleY = h / video.videoHeight;
      
      State.pipCtx.strokeStyle = '#0ea5e9';
      State.pipCtx.lineWidth = 2;
      State.pipCtx.font = '10px Inter';
      State.pipCtx.fillStyle = '#0ea5e9';
      
      detections.forEach(det => {
        const x = det.x * scaleX;
        const y = det.y * scaleY;
        const width = det.width * scaleX;
        const height = det.height * scaleY;
        
        State.pipCtx.strokeRect(x, y, width, height);
      });
    }
    
    // Actualizar contador en PiP HUD
    const pipCount = document.getElementById('pipCount');
    if (pipCount) {
      pipCount.textContent = detections.length;
    }
  } catch (e) {
    // ignore (a veces durante load)
  }
}

function expandPip() { switchView('camera'); }
function closePip() {
  switchView('camera');
}

// ============================================
// STREAM BUTTON (opcional, mantengo tu HTML)
// ============================================
function toggleStream() {
  // En tu HTML el startBtn está oculto.
  // Si luego lo activas, esto solo conecta WS y muestra estado.
  if (!State.websocket || State.websocket.readyState !== WebSocket.OPEN) {
    initializeWebSocket();
    showToast('WebSocket conectado', 'success');
  } else {
    try { State.websocket.close(); } catch {}
    showToast('WebSocket cerrado', 'info');
  }
}

// ============================================
// CAPTURE
// ============================================
function captureFrame() {
  const video = State.videoElement;
  if (!video || video.videoWidth === 0) {
    showToast('No hay video para capturar', 'error');
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    if (!blob) return;
    downloadBlob(blob, `fishwatch_capture_${Date.now()}.png`);
    showToast('Frame capturado', 'success');
  }, 'image/png');
}

// ============================================
// UTILS
// ============================================
function formatTime(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`;
}

function downloadText(content, filename, type) {
  const blob = new Blob([content], { type });
  downloadBlob(blob, filename);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed;
    bottom: 100px;
    right: 32px;
    background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#b4b4dc'};
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    z-index: 99999;
    animation: slideIn 0.3s ease;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ============================================
// DASHBOARD FUNCTIONS FOR AQUACULTURE
// ============================================
async function refreshDashboard() {
  try {
    // Get summary metrics
    const summaryRes = await fetch(`${CONFIG.API_BASE_URL}/api/metrics/summary?window=1h`);
    const summary = await summaryRes.json();
    
    // Update KPIs
    document.getElementById('kpiTotalToday').textContent = String(summary.totalToday || 0);
    document.getElementById('kpiPerMinute').textContent = (summary.perMinute || 0).toFixed(1);
    
    const density = summary.totalToday > 0 ? Math.ceil(summary.totalToday / 3) : 0;
    document.getElementById('kpiDensity').textContent = String(density);
    
    // Get database stats
    const dbRes = await fetch(`${CONFIG.API_BASE_URL}/api/db/stats`);
    const dbStats = await dbRes.json();
    
    if (dbStats.last_timestamp) {
      const lastTime = new Date(dbStats.last_timestamp).toLocaleTimeString('es-ES', { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
      document.getElementById('kpiLastSession').textContent = lastTime;
    }
    
    // Update health indicators
    updateHealthIndicators(summary);
    
    // Load sessions
    await loadSessions();
    
  } catch (error) {
    console.error('Error refreshing dashboard:', error);
  }
}

function updateHealthIndicators(summary) {
  // Distribution health (based on zone balance)
  const distributionHealth = Math.min(100, (summary.totalToday || 0) * 2);
  document.getElementById('distributionHealth').style.width = `${distributionHealth}%`;
  document.getElementById('distributionValue').textContent = distributionHealth > 70 ? 'Balanceada' : 'Revisar';
  
  // Activity health (based on per minute rate)
  const activityHealth = Math.min(100, (summary.perMinute || 0) * 10);
  document.getElementById('activityHealth').style.width = `${activityHealth}%`;
  document.getElementById('activityValue').textContent = activityHealth > 50 ? 'Normal' : 'Baja';
  
  // Precision health (based on avg confidence)
  const precisionHealth = (summary.avgConfidence || 0);
  document.getElementById('precisionHealth').style.width = `${precisionHealth}%`;
  document.getElementById('precisionValue').textContent = `${precisionHealth.toFixed(0)}%`;
  
  // Update tank status
  const tankStatus = document.getElementById('tankStatus');
  if (distributionHealth > 70 && activityHealth > 50) {
    tankStatus.textContent = 'NORMAL';
    tankStatus.className = 'status-badge status-good';
  } else if (distributionHealth > 40 || activityHealth > 30) {
    tankStatus.textContent = 'REVISAR';
    tankStatus.className = 'status-badge status-warning';
  } else {
    tankStatus.textContent = 'ALERTA';
    tankStatus.className = 'status-badge status-danger';
  }
}

async function loadSessions() {
  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/api/db/stats`);
    const data = await res.json();
    
    const sessionsList = document.getElementById('sessionsList');
    if (!sessionsList) return;
    
    if (data.total_records === 0) {
      sessionsList.innerHTML = `
        <div class="empty-state-small">
          <i class="fas fa-clipboard-list"></i>
          <p>No hay sesiones registradas</p>
        </div>
      `;
      return;
    }
    
    // Create session item (simplified - in real app would fetch list)
    sessionsList.innerHTML = `
      <div class="session-item">
        <div class="session-header">
          <span class="session-time">${new Date(data.last_timestamp).toLocaleString('es-ES')}</span>
          <span class="session-fish-count">${data.total_fish} peces</span>
        </div>
        <div class="session-details">
          Confianza: ${data.avg_confidence}% | FPS: ${data.avg_fps} | Latencia: ${data.avg_latency}ms
        </div>
      </div>
    `;
    
  } catch (error) {
    console.error('Error loading sessions:', error);
  }
}

async function refreshSessions() {
  const btn = event.target.closest('button');
  if (btn) {
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
  }
  await loadSessions();
  if (btn) {
    btn.innerHTML = '<i class="fas fa-sync-alt"></i>';
  }
}

async function exportReport() {
  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/api/db/stats`);
    const data = await res.json();
    
    const report = `REPORTE DE MONITOREO DE ACUICULTURA
========================================

Fecha: ${new Date().toLocaleString('es-ES')}

RESUMEN GENERAL:
- Total de peces detectados: ${data.total_fish}
- Registros totales: ${data.total_records}
- Confianza promedio: ${data.avg_confidence}%
- FPS promedio: ${data.avg_fps}
- Latencia promedio: ${data.avg_latency}ms

ÚLTIMA SESIÓN:
- Timestamp: ${data.last_timestamp}

========================================
Generado por FishWatch - Sistema de Monitoreo de Acuicultura
`;
    
    downloadText(report, `fishwatch_report_${Date.now()}.txt`, 'text/plain');
    showToast('Reporte exportado correctamente', 'success');
    
  } catch (error) {
    console.error('Error exporting report:', error);
    showToast('Error al exportar reporte', 'error');
  }
}

// Auto-refresh dashboard every 10 seconds when on dashboard view
setInterval(() => {
  const dashboardView = document.getElementById('dashboardView');
  if (dashboardView && dashboardView.classList.contains('active')) {
    refreshDashboard();
  }
}, 10000);

function escapeHtml(str) {
  return String(str)
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'","&#039;");
}

// ============================================
// SAVE SESSION TO DATABASE
// ============================================
async function saveSessionToDB() {
  if (State.events.length === 0) {
    showToast('No hay detecciones para guardar', 'error');
    return;
  }

  try {
    showToast('Guardando sesión en la base de datos...', 'info');
    
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/session/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error('Error guardando sesión');
    }

    const data = await response.json();
    
    if (data.success) {
      showToast(`✅ Guardado: ${data.unique_fish} peces únicos`, 'success');
      console.log('Sesión guardada:', data);
    } else {
      showToast(data.message || 'Error al guardar', 'error');
    }
  } catch (error) {
    console.error('Error saving session:', error);
    showToast('Error al guardar la sesión', 'error');
  }
}

// ============================================
// Logs
// ============================================
console.log('FishWatch Live - Premium Edition (Fixed)');
console.log('API:', CONFIG.API_BASE_URL);
console.log('WS:', CONFIG.WS_URL);

// ============================================
// HISTORY VIEW - DATA LOADING & MANAGEMENT
// ============================================
let historyData = [];
let historyCurrentPage = 1;
let historyItemsPerPage = 50;
let historyFilteredData = [];

// Load history data when switching to history view
async function loadHistoryData() {
  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/api/db/detections`);
    if (!res.ok) throw new Error('Error cargando datos históricos');
    
    const data = await res.json();
    historyData = data.detections || [];
    historyFilteredData = [...historyData];
    
    updateHistoryKPIs();
    refreshHistoryTable();
    
    console.log(`📊 Cargados ${historyData.length} registros históricos`);
  } catch (error) {
    console.error('Error loading history:', error);
    showToast('Error al cargar histórico', 'error');
  }
}

// Update KPI cards with statistics
function updateHistoryKPIs() {
  const totalRecords = historyFilteredData.length;
  document.getElementById('historyTotalRecords').textContent = totalRecords;
  
  // Calculate unique days
  const uniqueDays = new Set(
    historyFilteredData.map(d => new Date(d.timestamp).toDateString())
  ).size;
  document.getElementById('historyActiveDays').textContent = uniqueDays;
  
  // Average per day
  const avgDaily = uniqueDays > 0 ? Math.round(totalRecords / uniqueDays) : 0;
  document.getElementById('historyAvgDaily').textContent = avgDaily;
  
  // Average confidence
  const avgConf = historyFilteredData.length > 0
    ? historyFilteredData.reduce((sum, d) => sum + d.confidence, 0) / historyFilteredData.length
    : 0;
  document.getElementById('historyAvgConfidence').textContent = avgConf.toFixed(1) + '%';
}

// Refresh table with current page
function refreshHistoryTable() {
  const tbody = document.getElementById('historyTableBody');
  const pageInfo = document.getElementById('historyPageInfo');
  
  const totalPages = Math.ceil(historyFilteredData.length / historyItemsPerPage);
  const start = (historyCurrentPage - 1) * historyItemsPerPage;
  const end = start + historyItemsPerPage;
  const pageData = historyFilteredData.slice(start, end);
  
  if (pageData.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-row">
          <i class="fas fa-inbox"></i>
          <br>No hay registros que coincidan con los filtros
        </td>
      </tr>
    `;
  } else {
    tbody.innerHTML = pageData.map(record => {
      const date = new Date(record.timestamp);
      const dateStr = date.toLocaleDateString('es-ES');
      const timeStr = date.toLocaleTimeString('es-ES');
      const coords = `(${record.x1}, ${record.y1})`;
      
      return `
        <tr>
          <td>${record.id}</td>
          <td>${dateStr}<br><small style="color: rgba(0,0,0,0.5);">${timeStr}</small></td>
          <td><span class="zone-badge" style="background: ${getZoneColor(record.zone)}; display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-weight: 700;">${record.zone}</span></td>
          <td>${record.confidence.toFixed(1)}%</td>
          <td><code style="font-size: 11px;">${coords}</code></td>
          <td>${record.frame_number}</td>
        </tr>
      `;
    }).join('');
  }
  
  // Update pagination
  pageInfo.textContent = `Página ${historyCurrentPage} de ${totalPages || 1}`;
  document.getElementById('btnHistoryPrev').disabled = historyCurrentPage <= 1;
  document.getElementById('btnHistoryNext').disabled = historyCurrentPage >= totalPages;
}

function getZoneColor(zone) {
  const colors = { 'A': '#3b82f6', 'B': '#10b981', 'C': '#f59e0b' };
  return colors[zone] || '#6b7280';
}

// Pagination functions
function historyNextPage() {
  const totalPages = Math.ceil(historyFilteredData.length / historyItemsPerPage);
  if (historyCurrentPage < totalPages) {
    historyCurrentPage++;
    refreshHistoryTable();
  }
}

function historyPrevPage() {
  if (historyCurrentPage > 1) {
    historyCurrentPage--;
    refreshHistoryTable();
  }
}

// Apply filters
function applyHistoryFilters() {
  const startDate = document.getElementById('filterStartDate').value;
  const endDate = document.getElementById('filterEndDate').value;
  const zone = document.getElementById('filterZone').value;
  const minConf = parseFloat(document.getElementById('filterMinConfidence').value) || 0;
  
  historyFilteredData = historyData.filter(record => {
    const recordDate = new Date(record.timestamp).toISOString().split('T')[0];
    
    // Date filter
    if (startDate && recordDate < startDate) return false;
    if (endDate && recordDate > endDate) return false;
    
    // Zone filter
    if (zone !== 'all' && record.zone !== zone) return false;
    
    // Confidence filter
    if (record.confidence < minConf) return false;
    
    return true;
  });
  
  historyCurrentPage = 1;
  updateHistoryKPIs();
  refreshHistoryTable();
  showToast(`Filtrado: ${historyFilteredData.length} registros encontrados`, 'success');
}

// Clear filters
function clearHistoryFilters() {
  document.getElementById('filterStartDate').value = '';
  document.getElementById('filterEndDate').value = '';
  document.getElementById('filterZone').value = 'all';
  document.getElementById('filterMinConfidence').value = '0';
  
  historyFilteredData = [...historyData];
  historyCurrentPage = 1;
  updateHistoryKPIs();
  refreshHistoryTable();
  showToast('Filtros limpiados', 'info');
}

// Export to CSV
async function exportHistoryCSV() {
  if (historyFilteredData.length === 0) {
    showToast('No hay datos para exportar', 'error');
    return;
  }
  
  const headers = ['ID', 'Timestamp', 'Zona', 'Confianza', 'X1', 'Y1', 'X2', 'Y2', 'Frame'];
  const rows = historyFilteredData.map(r => [
    r.id,
    r.timestamp,
    r.zone,
    r.confidence.toFixed(2),
    r.x1,
    r.y1,
    r.x2,
    r.y2,
    r.frame_number
  ]);
  
  const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
  downloadText(csv, `fishwatch_history_${Date.now()}.csv`, 'text/csv');
  showToast(`Exportados ${historyFilteredData.length} registros a CSV`, 'success');
}

// Enhance switchView to load history data
const originalSwitchView = switchView;
switchView = function(view) {
  originalSwitchView(view);
  
  // Load history data when switching to history view
  if (view === 'history' && historyData.length === 0) {
    loadHistoryData();
  }
};
