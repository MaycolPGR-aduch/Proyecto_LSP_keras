# -*- coding: utf-8 -*-
"""
inference_gui_mvp.py — MVP de demostración con UI en pestañas, TTS y estadísticas.
- Pestañas: [DEMO] cámara + predicción + controles, [INFO] guía y estado.
- Botones: Iniciar / Pausar / Reset, Aplicar cámara, Fullscreen.
- Stats: FPS, latencia MediaPipe, latencia red, (CPU % si psutil disponible).
- TTS (gTTS): habla solo el TOP-1 cuando prob ≥ 0.90; acumula en bandeja de frases.
- Mantiene pipeline manos-only (126) y compatibilidad con modelos 1530 (activa FaceMesh solo si aplica).
"""

import os, json, time, argparse, threading, queue, tempfile
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Pillow para video en Tk
try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit("Necesitas instalar Pillow: pip install pillow") from e

# TTS (gTTS + playsound en hilo)
try:
    from gtts import gTTS
    from playsound import playsound
    TTS_OK = True
except Exception:
    TTS_OK = False

# CPU stats (opcional)
try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False

# ============== CLI ==============
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=os.path.join(ROOT, "lsp_sequence_model_frasescomunes.keras"),
                    help="Ruta del modelo .keras (por defecto: ../lsp_sequence_model_frasescomunes.keras)")
parser.add_argument("--camera", type=int, default=0, help="Índice de cámara (0 por defecto)")
parser.add_argument("--width",  type=int, default=1280, help="Ancho deseado (fallback a 640)")
parser.add_argument("--height", type=int, default=720,  help="Alto  deseado (fallback a 480)")
parser.add_argument("--data-dir", type=str, default=os.path.join(ROOT, "coordenadas"),
                    help="Carpeta para scaler_stats.npz / labels.json en caso no estén junto al modelo")
args = parser.parse_args()

MODEL_PATH = os.path.abspath(args.model)
DATA_DIR   = os.path.abspath(args.data_dir)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

# ============== Modelo / shapes / labels / scaler ==============
print("Cargando modelo:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
_, SEQ_LENGTH, FEAT_DIM = model.input_shape
NUM_CLASSES = model.output_shape[-1]
print(f"Modelo -> T={SEQ_LENGTH}, F={FEAT_DIM}, clases={NUM_CLASSES}")

FEAT_HANDS = 21*3*2 # 126
FEAT_FACE  = 468*3  # 1404

def feature_mode_from_dim(d:int):
    return "hands_face" if d >= (FEAT_HANDS + 1000) else "hands"
FEATURE_MODE = feature_mode_from_dim(FEAT_DIM)

def load_labels(model_path: str, data_dir: str, num_classes: int):
    cands = [
        os.path.join(os.path.dirname(model_path), "labels.json"),
        os.path.join(data_dir, "labels.json")
    ]
    for p in cands:
        if os.path.isfile(p):
            try:
                js = json.load(open(p, "r", encoding="utf-8"))
                if isinstance(js, dict) and "labels" in js and len(js["labels"]) == num_classes:
                    return js["labels"]
                if isinstance(js, dict):
                    rev = sorted(js.items(), key=lambda kv: kv[1])
                    labs = [k for k,_ in rev]
                    if len(labs) == num_classes: return labs
            except Exception:
                pass
    # fallback: carpetas
    if os.path.isdir(DATA_DIR):
        labs = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith("_")])
        if len(labs) == num_classes: return labs
    return [f"CLASE_{i}" for i in range(num_classes)]

LABELS = load_labels(MODEL_PATH, DATA_DIR, NUM_CLASSES)
print("Etiquetas:", LABELS)
print("Modo de características:", FEATURE_MODE)

def load_scaler(model_path: str, data_dir: str, feat_dim: int):
    cands = [
        os.path.join(os.path.dirname(model_path), "scaler_stats.npz"),
        os.path.join(data_dir, "scaler_stats.npz")
    ]
    for p in cands:
        if os.path.isfile(p):
            try:
                z = np.load(p)
                mean = z["mean"].astype(np.float32)
                std  = z["std"].astype(np.float32)
                if mean.shape[0]==feat_dim and std.shape[0]==feat_dim:
                    std[std<1e-8] = 1.0
                    print("Scaler cargado:", p)
                    return mean, std
            except Exception:
                pass
    print("[AVISO] No scaler_stats.npz. Usando identidad.")
    return np.zeros((feat_dim,), np.float32), np.ones((feat_dim,), np.float32)

FEAT_MEAN, FEAT_STD = load_scaler(MODEL_PATH, DATA_DIR, FEAT_DIM)

# ============== MediaPipe ==============
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils

def build_hands_fast(det=0.5, trk=0.3, complexity=0):
    return mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=det,
                          min_tracking_confidence=trk,
                          model_complexity=complexity)

def build_hands_reacq(det=0.4, complexity=0):
    return mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                          min_detection_confidence=det,
                          model_complexity=complexity)

# ============== Helpers de features ==============
from collections import deque
buffer     = deque(maxlen=SEQ_LENGTH)
pred_hist  = deque(maxlen=8)
last_valid = np.zeros((FEAT_HANDS,), dtype=np.float32)

def assemble_hands(res_h):
    left  = [0.0]*63
    right = [0.0]*63
    if res_h.multi_hand_landmarks and res_h.multi_handedness:
        for hand_lms, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords=[]
            for p in hand_lms.landmark:
                coords.extend([
                    float(np.clip(p.x, 0.0, 1.0)),
                    float(np.clip(p.y, 0.0, 1.0)),
                    float(np.clip(p.z, -1.0, 1.0)),
                ])
            label = handed.classification[0].label
            if label=="Left":  left  = coords[:63]
            else:              right = coords[:63]
    return np.array(left+right, dtype=np.float32)

def assemble_face(res_f):
    if res_f is None or not res_f.multi_face_landmarks:
        return np.zeros((FEAT_FACE,), dtype=np.float32)
    lm = res_f.multi_face_landmarks[0].landmark
    coords=[]
    for p in lm:
        coords.extend([
            float(np.clip(p.x, 0.0, 1.0)),
            float(np.clip(p.y, 0.0, 1.0)),
            float(np.clip(p.z, -1.0, 1.0)),
        ])
    if len(coords)<FEAT_FACE: coords += [0.0]*(FEAT_FACE-len(coords))
    return np.array(coords[:FEAT_FACE], dtype=np.float32)

def is_valid(vec126, require_two=False):
    left  = float(np.abs(vec126[:63]).sum())
    right = float(np.abs(vec126[63:]).sum())
    return (left>0 and right>0) if require_two else (left+right>0)

def vectorize(res_h, res_f, mode: str, require_two=False):
    global last_valid
    hands_vec = assemble_hands(res_h)
    if not is_valid(hands_vec, require_two=require_two):
        hands_vec = last_valid
    else:
        last_valid = hands_vec
    if mode=="hands": return hands_vec
    face_vec = assemble_face(res_f)
    return np.concatenate([hands_vec, face_vec], axis=0).astype(np.float32)

# ============== Cámara utilidades ==============
def open_camera(idx:int, width:int, height:int):
    backend = cv2.CAP_DSHOW if os.name=="nt" else 0
    c = cv2.VideoCapture(idx, backend)
    try: c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except: pass
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    c.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(0.05)
    ok,_ = c.read()
    if not ok:
        c.release()
        return None, (0,0)
    aw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); ah = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(aw-width)>16 or abs(ah-height)>16:
        print(f"[AVISO] pedido {width}x{height}, cámara {aw}x{ah}")
    return c, (aw, ah)

# ============== TTS worker ==============
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.last_spoken = None
        self.cooldown_s = 1.5  # evita repetir muy rápido
        self._stop = False

    def enqueue(self, text:str):
        if not TTS_OK: return
        self.q.put(text)

    def run(self):
        last_time = 0.0
        while not self._stop:
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            now = time.time()
            if text == self.last_spoken and (now - last_time) < self.cooldown_s:
                continue
            try:
                # gTTS requiere internet. Si falla, no bloquea la UI (hilo)
                tts = gTTS(text=text, lang="es")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    path = tmp.name
                tts.save(path)
                playsound(path)  # bloquea SOLO este hilo
            except Exception as e:
                print("[TTS] error:", e)
            finally:
                try:
                    if os.path.exists(path): os.remove(path)
                except Exception:
                    pass
                self.last_spoken = text
                last_time = time.time()

    def stop(self):
        self._stop = True

tts_worker = TTSWorker()
if TTS_OK:
    tts_worker.start()
else:
    print("[AVISO] gTTS/playsound no disponibles; TTS desactivado.")

# ============== App (Tkinter + Notebook) ==============
class App:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("LSP — MVP de Demostración")
        master.geometry("1150x700")

        # Estado runtime
        self.running = False
        self.fullscreen = False
        self.force_two = tk.BooleanVar(value=False)
        self.draw_hands = tk.BooleanVar(value=True)
        self.draw_face  = tk.BooleanVar(value=(FEATURE_MODE=="hands_face"))
        self.conf_thr   = tk.DoubleVar(value=0.90)  # 90%
        self.smooth_k   = tk.IntVar(value=5)
        self.cam_idx    = tk.IntVar(value=args.camera)
        self.res_var    = tk.StringVar(value=f"{args.width}x{args.height}")
        self.stride2    = tk.BooleanVar(value=True)  # procesar 1 de cada 2 frames
        self._stride_flag = 0

        # Recursos
        self.cap = None
        self.hands_fast  = build_hands_fast()
        self.hands_reacq = build_hands_reacq()
        self.face = (mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)
                     if FEATURE_MODE=="hands_face" else None)

        # Estadísticas
        self.fps = 0.0
        self._frames = 0
        self._t0 = time.time()
        self.lat_mp_ms  = 0.0
        self.lat_net_ms = 0.0

        # Bandeja de frases
        self.spoken_log = []

        # Notebook
        self.nb = ttk.Notebook(master)
        self.nb.pack(fill="both", expand=True)

        # Tab DEMO
        self.tab_demo = ttk.Frame(self.nb)
        self.nb.add(self.tab_demo, text="  DEMO  ")

        # Layout DEMO: izquierda (controles), derecha (video)
        self.tab_demo.columnconfigure(1, weight=1)
        self.tab_demo.rowconfigure(0, weight=1)

        self.left = ttk.Frame(self.tab_demo, padding=8)
        self.left.grid(row=0, column=0, sticky="nsw")
        for c in range(3): self.left.columnconfigure(c, weight=1)

        self.right = ttk.Frame(self.tab_demo)
        self.right.grid(row=0, column=1, sticky="nsew")
        self.right.rowconfigure(0, weight=1)
        self.right.columnconfigure(0, weight=1)

        # Video label
        self.video_label = ttk.Label(self.right)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Controles principales
        ttk.Label(self.left, text="Modelo:").grid(row=0, column=0, sticky="w")
        mpath = ttk.Entry(self.left, width=38)
        mpath.insert(0, MODEL_PATH)
        mpath.configure(state="readonly")
        mpath.grid(row=0, column=1, columnspan=2, sticky="we")

        ttk.Button(self.left, text="Iniciar", command=self.start).grid(row=1, column=0, sticky="we", pady=(6,2))
        self.btn_pause = ttk.Button(self.left, text="Pausar", command=self.toggle_run, state="disabled")
        self.btn_pause.grid(row=1, column=1, sticky="we", pady=(6,2))
        ttk.Button(self.left, text="Reset", command=self.reset).grid(row=1, column=2, sticky="we", pady=(6,2))

        ttk.Button(self.left, text="Fullscreen", command=self.toggle_fullscreen).grid(row=2, column=0, sticky="we")

        # Cámara
        ttk.Label(self.left, text="Cámara:").grid(row=3, column=0, sticky="w", pady=(8,0))
        ttk.Spinbox(self.left, from_=0, to=8, textvariable=self.cam_idx, width=6).grid(row=3, column=1, sticky="w")
        ttk.Label(self.left, text="Resolución:").grid(row=4, column=0, sticky="w")
        resopts = ["640x480", "854x480", "1280x720", "1920x1080"]
        ttk.Combobox(self.left, textvariable=self.res_var, values=resopts, state="readonly", width=10)\
            .grid(row=4, column=1, sticky="w")
        ttk.Button(self.left, text="Aplicar cámara", command=self.apply_camera).grid(row=4, column=2, sticky="we")

        # Parámetros y switches
        ttk.Checkbutton(self.left, text="Forzar 2 manos", variable=self.force_two).grid(row=5, column=0, sticky="w", pady=(8,0))
        ttk.Checkbutton(self.left, text="Ver manos", variable=self.draw_hands).grid(row=5, column=1, sticky="w", pady=(8,0))
        ttk.Checkbutton(self.left, text="Stride x2 (rápido)", variable=self.stride2).grid(row=5, column=2, sticky="w", pady=(8,0))
        ttk.Label(self.left, text="Suavizado").grid(row=6, column=0, sticky="w")
        ttk.Scale(self.left, from_=1, to=8, orient="horizontal", variable=self.smooth_k).grid(row=6, column=1, columnspan=2, sticky="we")
        ttk.Label(self.left, text="Umbral (%)").grid(row=7, column=0, sticky="w")
        self.scale_thr = ttk.Scale(self.left, from_=0, to=100, orient="horizontal")
        self.scale_thr.grid(row=7, column=1, columnspan=2, sticky="we")
        self.scale_thr.set(self.conf_thr.get()*100)

        # Estadísticas
        self.lbl_stats = ttk.Label(self.left, text="FPS: -- | MP: -- ms | NET: -- ms")
        self.lbl_stats.grid(row=8, column=0, columnspan=3, sticky="w", pady=(8,0))
        self.lbl_cpu = ttk.Label(self.left, text=("CPU: --%" if PSUTIL_OK else "CPU: n/d"))
        self.lbl_cpu.grid(row=9, column=0, columnspan=3, sticky="w")

        # Top-3 + TTS y bandeja
        self.lbl_top = ttk.Label(self.left, text="Top-3: —")
        self.lbl_top.grid(row=10, column=0, columnspan=3, sticky="w", pady=(8,0))

        ttk.Label(self.left, text="Bandeja de frases (TTS ≥90%)").grid(row=11, column=0, columnspan=3, sticky="w", pady=(6,0))
        self.txt_tray = tk.Text(self.left, height=8, width=50)
        self.txt_tray.grid(row=12, column=0, columnspan=3, sticky="we")
        ttk.Button(self.left, text="Limpiar", command=self.clear_tray).grid(row=13, column=0, sticky="we", pady=(4,0))
        ttk.Button(self.left, text="Guardar...", command=self.save_tray).grid(row=13, column=1, sticky="we", pady=(4,0))
        ttk.Button(self.left, text="Copiar", command=self.copy_tray).grid(row=13, column=2, sticky="we", pady=(4,0))

        # Tab INFO
        self.tab_info = ttk.Frame(self.nb, padding=12)
        self.nb.add(self.tab_info, text="  INFO  ")
        info_txt = (
            "Sistema de reconocimiento de Lengua de Señas Peruana (MVP)\n"
            "• Solo manos (126 features). Sin rostro.\n"
            "• Re-muestreo temporal + estandarización idéntica a entrenamiento.\n"
            "• TTS (gTTS) para top-1 ≥ 90% (concatenación en bandeja).\n"
            "• Consejos:\n"
            "    - Encadre y distancia constantes.\n"
            "    - Stride x2 para más FPS (menos latencia de predicción).\n"
            "    - Resolución 640×480 si el equipo es lento.\n"
        )
        tk.Message(self.tab_info, text=info_txt, width=900, justify="left").pack(anchor="w")

        # Cámara inicial (no inicia loop hasta presionar Iniciar)
        self.cap, (aw, ah) = open_camera(self.cam_idx.get(), *self._parse_res(self.res_var.get()))
        if self.cap is None:
            self.cap, _ = open_camera(self.cam_idx.get(), 640, 480)
            if self.cap is None:
                messagebox.showerror("Cámara", "No se pudo abrir la cámara.")

        # Cerrar limpio
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- UI actions ---
    def _parse_res(self, s):
        try:
            w,h = map(int, s.split("x"))
            return w,h
        except:
            return (1280,720)

    def start(self):
        if self.cap is None:
            messagebox.showerror("Cámara", "Cámara no disponible.")
            return
        self.running = True
        self.btn_pause.configure(state="normal")
        self.master.after(1, self._loop)

    def toggle_run(self):
        self.running = not self.running
        self.btn_pause.configure(text=("Reanudar" if not self.running else "Pausar"))

    def reset(self):
        buffer.clear(); pred_hist.clear()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.master.attributes("-fullscreen", self.fullscreen)

    def apply_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        w,h = self._parse_res(self.res_var.get())
        self.cap, (aw,ah) = open_camera(self.cam_idx.get(), w, h)
        if self.cap is None:
            self.cap, _ = open_camera(self.cam_idx.get(), 640, 480)
            if self.cap is None:
                messagebox.showerror("Cámara", "No se pudo abrir la cámara.")
        self.reset()

    def clear_tray(self):
        self.spoken_log.clear()
        self.txt_tray.delete("1.0", "end")

    def save_tray(self):
        txt = self.txt_tray.get("1.0", "end").strip()
        if not txt:
            messagebox.showinfo("Guardar", "No hay texto para guardar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texto", "*.txt")])
        if path:
            open(path, "w", encoding="utf-8").write(txt)
            messagebox.showinfo("Guardar", f"Guardado en:\n{path}")

    def copy_tray(self):
        txt = self.txt_tray.get("1.0", "end").strip()
        self.master.clipboard_clear()
        self.master.clipboard_append(txt)

    def on_close(self):
        try:
            if self.cap: self.cap.release()
        except: pass
        try:
            tts_worker.stop()
        except: pass
        self.master.destroy()

    # --- Core loop ---
    def _loop(self):
        ok, frame = self.cap.read()
        if ok:
            self._process_frame(frame)
        # actualizar CPU cada ~0.5s
        if PSUTIL_OK and (time.time() - getattr(self, "_cpu_t", 0)) > 0.5:
            self._cpu_t = time.time()
            try:
                self.lbl_cpu.configure(text=f"CPU: {psutil.cpu_percent(interval=None):.0f}%")
            except Exception:
                self.lbl_cpu.configure(text="CPU: n/d")
        # schedule
        self.master.after(1, self._loop if self.running else lambda: None)

    def _process_frame(self, frame_bgr):
        t0 = time.time()
        frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe (lite)
        res_h = self.hands_fast.process(rgb)
        if self.force_two.get():
            cnt = len(res_h.multi_hand_landmarks) if res_h.multi_hand_landmarks else 0
            if cnt < 2:
                res_h = self.hands_reacq.process(rgb)

        res_f = (self.face.process(rgb) if (FEATURE_MODE=="hands_face" and self.face is not None) else None)
        t1 = time.time()

        # Dibujo
        if self.draw_hands.get() and res_h.multi_hand_landmarks:
            for hl in res_h.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_bgr, hl, mp.solutions.hands.HAND_CONNECTIONS)

        # Buffer (stride opcional)
        if self.running:
            vec = vectorize(res_h, res_f, FEATURE_MODE, require_two=self.force_two.get())
            if self.stride2.get():
                self._stride_flag ^= 1
                if self._stride_flag == 0:
                    buffer.append(vec)
            else:
                buffer.append(vec)

        top_text = "—"
        if len(buffer) == SEQ_LENGTH:
            seq = np.array(buffer, np.float32)
            seq = (seq - FEAT_MEAN[None,:]) / FEAT_STD[None,:]
            t2 = time.time()
            probs = model(seq[None,...], training=False).numpy()[0]
            t3 = time.time()

            # latencias
            self.lat_mp_ms  = (t1 - t0) * 1000.0
            self.lat_net_ms = (t3 - t2) * 1000.0

            pred_hist.append(probs)
            k = max(1, int(self.smooth_k.get()))
            smoothed = np.mean(list(pred_hist)[-k:], axis=0)
            idx_top = smoothed.argsort()[-min(3,len(LABELS)):][::-1]
            best_idx = int(idx_top[0]); best_p = float(smoothed[best_idx])

            thr = float(self.scale_thr.get())/100.0
            top_text = f"{LABELS[best_idx]}  {best_p*100:.1f}%"
            if best_p >= thr:
                # TTS: solo top1 >= 90% (thr), y agregar a bandeja
                spoken = LABELS[best_idx]
                self._speak_and_append(spoken)

            # Pintar top-3 en el frame
            y0=30
            for j,i in enumerate(idx_top):
                txt=f"{j+1}. {LABELS[int(i)]}  {smoothed[int(i)]*100:.1f}%"
                cv2.putText(frame_bgr, txt, (10,y0+24*j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)

        # FPS
        self._frames += 1
        now = time.time()
        if now - self._t0 >= 1.0:
            self.fps = self._frames / (now - self._t0)
            self._frames = 0
            self._t0 = now

        # Overlays de stats
        cv2.putText(frame_bgr, f"FPS: {self.fps:.1f}", (10, frame_bgr.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,200,30), 2)
        self.lbl_stats.configure(text=f"FPS: {self.fps:.1f} | MP: {self.lat_mp_ms:.1f} ms | NET: {self.lat_net_ms:.1f} ms")
        self.lbl_top.configure(text=f"Top-1: {top_text}")

        # Mostrar en Tk
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _speak_and_append(self, text: str):
        # evitar duplicados inmediatos en la bandeja si igual al último
        if not self.spoken_log or (self.spoken_log and self.spoken_log[-1] != text):
            self.spoken_log.append(text)
            # agrega con espacio, forma frases simples
            self.txt_tray.insert("end", ("" if self.txt_tray.index('end-1c')=='1.0' else " ") + text)
            self.txt_tray.see("end")
        if TTS_OK:
            tts_worker.enqueue(text)

# ============== main ==============
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
