# -*- coding: utf-8 -*-
"""
inference_gui_pro.py — MVP con UI estilo dashboard (sidebar + tarjetas)
- Solo manos (126); compat con 1530 si carga un modelo antiguo (activa FaceMesh).
- Pestañas no necesarias: todo concentrado en vista "Traductor" tipo panel.
- Controles: Iniciar, Pausar/Reanudar, Reset, cámara (índice+resolución) y Aplicar.
- TTS (gTTS) sólo top-1 >= 90%; bandeja concatena palabras.
- Stats: FPS / latencia MediaPipe / latencia red / CPU%.
"""

import os, json, time, argparse, threading, queue, tempfile
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Pillow para la vista de cámara
try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit("Necesitas instalar Pillow: pip install pillow") from e

# TTS (opcional)
try:
    from gtts import gTTS
    from playsound import playsound
    TTS_OK = True
except Exception:
    TTS_OK = False

# Stats CPU (opcional)
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
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--data-dir", type=str, default=os.path.join(ROOT, "coordenadas"))
args = parser.parse_args()

MODEL_PATH = os.path.abspath(args.model)
DATA_DIR   = os.path.abspath(args.data_dir)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

# ============== Carga modelo / shapes / labels / scaler ==============
print("Cargando modelo:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
_, SEQ_LENGTH, FEAT_DIM = model.input_shape
NUM_CLASSES = model.output_shape[-1]
print(f"Modelo -> T={SEQ_LENGTH}, F={FEAT_DIM}, clases={NUM_CLASSES}")

FEAT_HANDS = 21*3*2
FEAT_FACE  = 468*3

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
                mean = z["mean"].astype(np.float32); std = z["std"].astype(np.float32)
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

# ============== Features helpers ==============
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

# ============== Cámara ==============
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
        self.cooldown_s = 1.5
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
            path = None
            try:
                tts = gTTS(text=text, lang="es")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    path = tmp.name
                tts.save(path)
                playsound(path)
            except Exception as e:
                print("[TTS] error:", e)
            finally:
                try:
                    if path and os.path.exists(path): os.remove(path)
                except Exception:
                    pass
                self.last_spoken = text
                last_time = time.time()
    def stop(self):
        self._stop = True

tts_worker = TTSWorker()
if TTS_OK: tts_worker.start()
else: print("[AVISO] gTTS/playsound no disponibles; TTS desactivado.")

# ============== UI: estilos y layout tipo dashboard ==============
PRIMARY_BG  = "#0B0F19"
PRIMARY_TXT = "#ffffff"
SURF_BG     = "#F5F7FB"
CARD_BG     = "#FFFFFF"
ACCENT      = "#111827"
BORDER      = "#E5E7EB"

class App:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("SignTalk — Traductor IA (MVP)")
        master.geometry("1280x740")
        master.minsize(1100, 680)

        # ttk theme básico y estilos
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Sidebar.TFrame", background=PRIMARY_BG)
        style.configure("Sidebar.TLabel", background=PRIMARY_BG, foreground="#D1D5DB")
        style.configure("SidebarHeader.TLabel", background=PRIMARY_BG, foreground="#fff", font=("Segoe UI", 13, "bold"))
        style.configure("Nav.TButton", background=PRIMARY_BG, foreground="#E5E7EB", anchor="w", padding=8)
        style.map("Nav.TButton", background=[("active", "#111827")])
        style.configure("Card.TFrame", background=CARD_BG, relief="flat")
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=ACCENT, font=("Segoe UI", 12, "bold"))
        style.configure("Badge.TLabel", background="#F3F4F6", foreground="#111827", padding=(8,3))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Ghost.TButton")
        style.configure("Stat.TLabel", background=CARD_BG, foreground="#6B7280")

        # Estado runtime
        self.running = False
        self.force_two = tk.BooleanVar(value=False)
        self.draw_hands = tk.BooleanVar(value=True)
        self.stride2   = tk.BooleanVar(value=True)
        self.conf_thr  = tk.DoubleVar(value=0.90)
        self.smooth_k  = tk.IntVar(value=5)
        self.cam_idx   = tk.IntVar(value=args.camera)
        self.res_var   = tk.StringVar(value=f"{args.width}x{args.height}")
        self.fullscreen = False

        # Recursos
        self.cap = None
        self.hands_fast  = build_hands_fast()
        self.hands_reacq = build_hands_reacq()
        self.face = (mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)
                     if FEATURE_MODE=="hands_face" else None)

        # Stats
        self.fps=0.0; self._frames=0; self._t0=time.time()
        self.lat_mp_ms=0.0; self.lat_net_ms=0.0

        # Bandeja
        self.spoken_log=[]

        # --- Layout principal: sidebar + main ---
        self.root = ttk.Frame(master)
        self.root.pack(fill="both", expand=True)

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=(16,16))
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self._build_sidebar()

        # Main area
        self.main = ttk.Frame(self.root, padding=(16,16))
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.rowconfigure(2, weight=1)
        self.main.columnconfigure(0, weight=1)

        self._build_header()
        self._build_camera_card()
        self._build_controls_row()
        self._build_translated_card()

        # Cámara inicial (no inicia hasta “Iniciar”)
        self.cap, _ = open_camera(self.cam_idx.get(), *self._parse_res(self.res_var.get()))
        if self.cap is None:
            self.cap, _ = open_camera(self.cam_idx.get(), 640, 480)
            if self.cap is None:
                messagebox.showerror("Cámara", "No se pudo abrir la cámara.")

        master.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI blocks ----------
    def _build_sidebar(self):
        ttk.Label(self.sidebar, text="SignTalk", style="SidebarHeader.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self.sidebar, text="Traductor IA", style="Sidebar.TLabel").grid(row=1, column=0, sticky="w", pady=(0,20))

        def nav_btn(text, selected=False):
            b = ttk.Button(self.sidebar, text=f"  {text}", style="Nav.TButton")
            b.grid(sticky="we", pady=2)
            if selected:
                b.state(["disabled"])
            return b

        nav_btn("Inicio")
        nav_btn("Traductor", selected=True)
        nav_btn("Diccionario")
        nav_btn("Historial")
        nav_btn("Comunidad")
        nav_btn("Estadísticas")
        nav_btn("Configuración")
        nav_btn("Ayuda")

        for i in range(12):
            self.sidebar.rowconfigure(i, weight=0)
        self.sidebar.rowconfigure(99, weight=1)  # empujar

        ttk.Label(self.sidebar, text="v2.1.0 • Beta", style="Sidebar.TLabel").grid(row=99, column=0, sticky="sw")

    def _build_header(self):
        head = ttk.Frame(self.main)
        head.grid(row=0, column=0, sticky="we", pady=(0,12))
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text="Traductor", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.badge = ttk.Label(head, text="Detenido", style="Badge.TLabel")
        self.badge.grid(row=0, column=1, sticky="e")

    def _build_camera_card(self):
        card = ttk.Frame(self.main, style="Card.TFrame")
        card.grid(row=1, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        ttk.Label(card, text="Vista de Cámara", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))

        # Marco de cámara
        self.video_wrap = ttk.Frame(card, style="Card.TFrame")
        self.video_wrap.grid(row=1, column=0, sticky="nsew")
        self.video_wrap.rowconfigure(0, weight=1)
        self.video_wrap.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(self.video_wrap, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Placeholder
        ph = tk.Canvas(self.video_wrap, bg="#EFF2F7", highlightthickness=1, highlightbackground=BORDER)
        ph.grid(row=0, column=0, sticky="nsew")
        ph.create_text(400,140, text="Simulación de cámara activa", font=("Segoe UI", 12), fill="#6B7280")
        ph.create_text(400,170, text="Muestra tus manos para comenzar a traducir", font=("Segoe UI", 10), fill="#9CA3AF")

        self._ph_canvas = ph  # para ocultarlo cuando haya video

    def _build_controls_row(self):
        row = ttk.Frame(self.main)
        row.grid(row=2, column=0, sticky="nsew", pady=(12,0))
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        row.columnconfigure(2, weight=1)

        # ---- Controles ( tarjeta ) ----
        ctr = ttk.Frame(row, style="Card.TFrame", padding=8)
        ctr.grid(row=0, column=0, sticky="nsew")
        ttk.Label(ctr, text="Controles", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))

        btns = ttk.Frame(ctr)
        btns.grid(row=1, column=0, sticky="we")
        self.btn_start = ttk.Button(btns, text="▶  Iniciar", style="Primary.TButton", command=self.start)
        self.btn_start.grid(row=0, column=0, padx=(0,6))
        self.btn_pause = ttk.Button(btns, text="⏸  Pausar", command=self.toggle_run, state="disabled")
        self.btn_pause.grid(row=0, column=1, padx=6)
        ttk.Button(btns, text="↺  Reset", command=self.reset).grid(row=0, column=2, padx=6)

        opt = ttk.Frame(ctr)
        opt.grid(row=2, column=0, sticky="we", pady=(8,0))
        ttk.Checkbutton(opt, text="Forzar 2 manos", variable=self.force_two).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opt, text="Ver manos", variable=self.draw_hands).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(opt, text="Modo rápido (Stride x2)", variable=self.stride2).grid(row=0, column=2, sticky="w")

        sliders = ttk.Frame(ctr)
        sliders.grid(row=3, column=0, sticky="we", pady=(8,0))
        ttk.Label(sliders, text="Umbral (%)").grid(row=0, column=0, sticky="w")
        self.scale_thr = ttk.Scale(sliders, from_=0, to=100, orient="horizontal")
        self.scale_thr.grid(row=0, column=1, sticky="we", padx=8)
        self.scale_thr.set(self.conf_thr.get()*100)

        ttk.Label(sliders, text="Suavizado").grid(row=1, column=0, sticky="w", pady=(4,0))
        self.scale_smooth = ttk.Scale(sliders, from_=1, to=8, orient="horizontal")
        self.scale_smooth.grid(row=1, column=1, sticky="we", padx=8)
        self.scale_smooth.set(self.smooth_k.get())

        for i in range(2): sliders.columnconfigure(i, weight=1)

        # ---- Cámara (tarjeta) ----
        cam = ttk.Frame(row, style="Card.TFrame", padding=8)
        cam.grid(row=0, column=1, sticky="nsew")
        ttk.Label(cam, text="Cámara", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))

        cam_row = ttk.Frame(cam)
        cam_row.grid(row=1, column=0, sticky="we")
        ttk.Label(cam_row, text="Índice").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(cam_row, from_=0, to=8, textvariable=self.cam_idx, width=8).grid(row=0, column=1, sticky="w", padx=(4,12))
        ttk.Label(cam_row, text="Resolución").grid(row=0, column=2, sticky="w")
        ttk.Combobox(cam_row, textvariable=self.res_var, values=["640x480","854x480","1280x720","1920x1080"],
                     width=12, state="readonly").grid(row=0, column=3, sticky="w", padx=(4,12))
        ttk.Button(cam_row, text="Aplicar", command=self.apply_camera).grid(row=0, column=4, sticky="we")

        # ---- Estadísticas (tarjeta) ----
        st = ttk.Frame(row, style="Card.TFrame", padding=8)
        st.grid(row=0, column=2, sticky="nsew")
        ttk.Label(st, text="Estadísticas", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))
        self.lbl_stats = ttk.Label(st, text="FPS: --  |  MP: -- ms  |  NET: -- ms", style="Stat.TLabel")
        self.lbl_stats.grid(row=1, column=0, sticky="w")
        self.lbl_cpu   = ttk.Label(st, text=("CPU: --%" if PSUTIL_OK else "CPU: n/d"), style="Stat.TLabel")
        self.lbl_cpu.grid(row=2, column=0, sticky="w", pady=(4,0))

    def _build_translated_card(self):
        card = ttk.Frame(self.main, style="Card.TFrame", padding=8)
        card.grid(row=3, column=0, sticky="nsew", pady=(12,0))
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Palabras Traducidas", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))
        self.txt_tray = tk.Text(card, height=4, wrap="word")
        self.txt_tray.grid(row=1, column=0, sticky="nsew")
        btns = ttk.Frame(card)
        btns.grid(row=2, column=0, sticky="e", pady=(6,0))
        ttk.Button(btns, text="Limpiar", command=self.clear_tray).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Guardar…", command=self.save_tray).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Copiar", command=self.copy_tray).grid(row=0, column=2, padx=4)

    # ---------- acciones ----------
    def _parse_res(self, s):
        try:
            w,h = map(int, s.split("x")); return w,h
        except:  return (1280,720)

    def start(self):
        if self.cap is None:
            messagebox.showerror("Cámara", "No disponible.")
            return
        self.running = True
        self.btn_pause.configure(state="normal")
        self.badge.configure(text="Activo")
        self._ph_canvas.grid_forget()  # ocultar placeholder
        self.master.after(1, self._loop)

    def toggle_run(self):
        self.running = not self.running
        self.btn_pause.configure(text=("Reanudar" if not self.running else "Pausar"))
        self.badge.configure(text=("Detenido" if not self.running else "Activo"))

    def reset(self):
        buffer.clear(); pred_hist.clear()

    def apply_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        w,h = self._parse_res(self.res_var.get())
        self.cap, _ = open_camera(self.cam_idx.get(), w, h)
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
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texto","*.txt")])
        if path:
            open(path,"w",encoding="utf-8").write(txt)
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

    # ---------- loop principal ----------
    def _loop(self):
        ok, frame = self.cap.read()
        if ok:
            self._process_frame(frame)
        # CPU cada ~0.5s
        if PSUTIL_OK and (time.time() - getattr(self, "_cpu_t", 0)) > 0.5:
            self._cpu_t = time.time()
            try: self.lbl_cpu.configure(text=f"CPU: {psutil.cpu_percent(interval=None):.0f}%")
            except: self.lbl_cpu.configure(text="CPU: n/d")
        self.master.after(1, self._loop if self.running else lambda: None)

    def _process_frame(self, frame_bgr):
        t0 = time.time()
        frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe lite
        res_h = self.hands_fast.process(rgb)
        if self.force_two.get():
            cnt = len(res_h.multi_hand_landmarks) if res_h.multi_hand_landmarks else 0
            if cnt < 2: res_h = self.hands_reacq.process(rgb)
        res_f = (mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False).process(rgb)
                 if FEATURE_MODE=="hands_face" else None)
        t1 = time.time()

        # Dibujo
        if self.draw_hands.get() and res_h.multi_hand_landmarks:
            for hl in res_h.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_bgr, hl, mp.solutions.hands.HAND_CONNECTIONS)

        # Buffer (stride opcional)
        if self.running:
            vec = vectorize(res_h, res_f, FEATURE_MODE, require_two=self.force_two.get())
            if self.stride2.get():
                # añade 1 de cada 2 frames
                if not hasattr(self, "_stride"): self._stride = 0
                self._stride ^= 1
                if self._stride == 0: buffer.append(vec)
            else:
                buffer.append(vec)

        top_text = "—"
        if len(buffer) == SEQ_LENGTH:
            seq = np.array(buffer, np.float32)
            seq = (seq - FEAT_MEAN[None,:]) / FEAT_STD[None,:]
            t2 = time.time()
            probs = model(seq[None,...], training=False).numpy()[0]
            t3 = time.time()

            self.lat_mp_ms  = (t1 - t0) * 1000.0
            self.lat_net_ms = (t3 - t2) * 1000.0

            pred_hist.append(probs)
            k = max(1, int(self.scale_smooth.get()))
            smoothed = np.mean(list(pred_hist)[-k:], axis=0)

            idx_top = smoothed.argsort()[-min(3,len(LABELS)):][::-1]
            best_idx = int(idx_top[0]); best_p = float(smoothed[best_idx])

            thr = float(self.scale_thr.get())/100.0
            top_text = f"{LABELS[best_idx]}  {best_p*100:.1f}%"
            if best_p >= thr:
                self._speak_and_append(LABELS[best_idx])

            # pintar top-3
            y0=30
            for j,i in enumerate(idx_top):
                txt=f"{j+1}. {LABELS[int(i)]}  {smoothed[int(i)]*100:.1f}%"
                cv2.putText(frame_bgr, txt, (10,y0+24*j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)

        # FPS
        self._frames += 1
        now = time.time()
        if now - self._t0 >= 1.0:
            self.fps = self._frames / (now - self._t0)
            self._frames = 0; self._t0 = now

        # Stats en UI
        self.lbl_stats.configure(text=f"FPS: {self.fps:.1f}  |  MP: {self.lat_mp_ms:.1f} ms  |  NET: {self.lat_net_ms:.1f} ms")

        # Mostrar
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _speak_and_append(self, text: str):
        if not self.spoken_log or self.spoken_log[-1] != text:
            self.spoken_log.append(text)
            self.txt_tray.insert("end", ("" if self.txt_tray.index('end-1c')=='1.0' else " ") + text)
            self.txt_tray.see("end")
        if TTS_OK:
            tts_worker.enqueue(text)

# ============== main ==============
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
