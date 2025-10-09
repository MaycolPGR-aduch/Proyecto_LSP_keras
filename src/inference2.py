# -*- coding: utf-8 -*-
"""
inference2.py — Inferencia en tiempo real (SOLO MANOS, 126 features) alineada con preprocess.py
- Carga modelo .keras y deduce (SEQ_LENGTH, FEAT_DIM)
- Usa MediaPipe Hands; FaceMesh solo si el modelo fuera antiguo (1530 features)
- Estandariza con scaler_stats.npz (mean/std) igual que en preprocess (si existe)
- Buffer deslizante de SEQ_LENGTH, suavizado Top-K y umbral de confianza
- Apertura de cámara robusta (intenta 1280x720 y cae a 640x480), MJPG, baja latencia

Requisitos: tensorflow, opencv-python, mediapipe, pillow, tkinter (std)
"""

import os, json, time, argparse
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, messagebox

# Pillow para mostrar video en Tk
try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit("Necesitas instalar Pillow: pip install pillow") from e

# ==================== Paths / CLI ====================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=os.path.join(ROOT, "lsp_sequence_model_frasescomunes.keras"),
                    help="Ruta del modelo .keras (por defecto: ../lsp_sequence_model_frasescomunes.keras)")
parser.add_argument("--camera", type=int, default=0, help="Índice de cámara (0 por defecto)")
parser.add_argument("--width",  type=int, default=1280, help="Ancho deseado de la cámara (fallback a 640)")
parser.add_argument("--height", type=int, default=720,  help="Alto deseado de la cámara (fallback a 480)")
parser.add_argument("--data-dir", type=str, default=os.path.join(ROOT, "coordenadas"),
                    help="Carpeta donde buscar scaler_stats.npz y labels.json si no están junto al modelo")
args = parser.parse_args()

MODEL_PATH = os.path.abspath(args.model)
DATA_DIR   = os.path.abspath(args.data_dir)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

# ==================== Carga modelo / shapes ====================
print("Cargando modelo desde:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
_, SEQ_LENGTH, FEAT_DIM = model.input_shape
NUM_CLASSES = model.output_shape[-1]
print(f"Modelo -> SEQ_LENGTH={SEQ_LENGTH}, FEAT_DIM={FEAT_DIM}, clases={NUM_CLASSES}")

FEAT_HANDS = 21 * 3 * 2   # 126
FEAT_FACE  = 468 * 3      # 1404

def _feature_mode_from_dim(d: int):
    return "hands_face" if d >= (FEAT_HANDS + 1000) else "hands"

FEATURE_MODE = _feature_mode_from_dim(FEAT_DIM)
print("Modo de características:", FEATURE_MODE)

# ==================== Labels ====================
def load_labels(model_path: str, data_dir: str, num_classes: int):
    # 1) Junto al modelo
    lbl1 = os.path.join(os.path.dirname(model_path), "labels.json")
    if os.path.isfile(lbl1):
        try:
            js = json.load(open(lbl1, "r", encoding="utf-8"))
            labs = js.get("labels", None)
            if isinstance(labs, list) and len(labs) == num_classes:
                return labs
        except Exception:
            pass
    # 2) En data_dir
    lbl2 = os.path.join(data_dir, "labels.json")
    if os.path.isfile(lbl2):
        try:
            js = json.load(open(lbl2, "r", encoding="utf-8"))
            # admite mapa {label: idx} o {"labels":[...]}
            if isinstance(js, dict) and "labels" in js and len(js["labels"]) == num_classes:
                return js["labels"]
            if isinstance(js, dict):
                # reconstruye lista por índice
                rev = sorted(js.items(), key=lambda kv: kv[1])
                labs = [k for k, _ in rev]
                if len(labs) == num_classes:
                    return labs
        except Exception:
            pass
    # 3) Fallback: carpetas en data_dir
    if os.path.isdir(data_dir):
        labs = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("_")])
        if len(labs) == num_classes:
            return labs
    return [f"CLASE_{i}" for i in range(num_classes)]

LABELS = load_labels(MODEL_PATH, DATA_DIR, NUM_CLASSES)
print("Etiquetas:", LABELS)

# ==================== Scaler (mean/std) ====================
def load_scaler(model_path: str, data_dir: str, feat_dim: int):
    # prioridad: junto al modelo -> data_dir -> identidad
    cand = [
        os.path.join(os.path.dirname(model_path), "scaler_stats.npz"),
        os.path.join(data_dir, "scaler_stats.npz"),
    ]
    for p in cand:
        if os.path.isfile(p):
            try:
                npz = np.load(p)
                mean = npz["mean"].astype(np.float32)
                std  = npz["std"].astype(np.float32)
                if mean.shape[0] == feat_dim and std.shape[0] == feat_dim:
                    print(f"Scaler cargado desde: {p}")
                    std[std < 1e-8] = 1.0
                    return mean, std
            except Exception:
                pass
    print("[AVISO] No se encontró scaler_stats.npz. Usando identidad (sin estandarizar).")
    return np.zeros((feat_dim,), dtype=np.float32), np.ones((feat_dim,), dtype=np.float32)

FEAT_MEAN, FEAT_STD = load_scaler(MODEL_PATH, DATA_DIR, FEAT_DIM)

# ==================== MediaPipe ====================
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils

def build_hands_fast(det=0.5, trk=0.3):
    return mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=det,
                          min_tracking_confidence=trk, model_complexity=1)

def build_hands_reacq(det=0.4):
    return mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                          min_detection_confidence=det, model_complexity=1)

hands_fast  = build_hands_fast()
hands_reacq = build_hands_reacq()
face        = (mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)
               if FEATURE_MODE == "hands_face" else None)

# ==================== Helpers de features ====================
from collections import deque
buffer     = deque(maxlen=SEQ_LENGTH)
pred_hist  = deque(maxlen=8)
last_valid = np.zeros((FEAT_HANDS,), dtype=np.float32)  # para rellenar huecos de manos

running    = True
draw_hands = True
draw_face  = (FEATURE_MODE == "hands_face")
fullscreen = False
conf_thr   = 0.5
smooth_k   = 5
force_two  = False

fps_value   = 0.0
_last_fps_t = time.time()
_frames     = 0

def _topk_indices(probs: np.ndarray, k: int = 3):
    idx = probs.argsort()[-k:][::-1]
    return idx, probs[idx]

def _assemble_hands(res_h):
    # devuelve vector 126 (63 izq + 63 der), con clipping como en capture
    left  = [0.0] * 63
    right = [0.0] * 63
    if res_h.multi_hand_landmarks and res_h.multi_handedness:
        for hand_lms, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords = []
            for p in hand_lms.landmark:
                coords.extend([
                    float(np.clip(p.x, 0.0, 1.0)),
                    float(np.clip(p.y, 0.0, 1.0)),
                    float(np.clip(p.z, -1.0, 1.0)),
                ])
            label = handed.classification[0].label  # "Left"/"Right"
            if label == "Left":  left  = coords[:63]
            else:                right = coords[:63]
    return np.array(left + right, dtype=np.float32)

def _assemble_face(res_f):
    if res_f is None or not res_f.multi_face_landmarks:
        return np.zeros((FEAT_FACE,), dtype=np.float32)
    lm = res_f.multi_face_landmarks[0].landmark
    coords = []
    for p in lm:
        coords.extend([
            float(np.clip(p.x, 0.0, 1.0)),
            float(np.clip(p.y, 0.0, 1.0)),
            float(np.clip(p.z, -1.0, 1.0)),
        ])
    if len(coords) < FEAT_FACE:
        coords += [0.0] * (FEAT_FACE - len(coords))
    return np.array(coords[:FEAT_FACE], dtype=np.float32)

def _is_valid_hands(vec126: np.ndarray, require_two=False):
    left_sum  = float(np.abs(vec126[:63]).sum())
    right_sum = float(np.abs(vec126[63:]).sum())
    return (left_sum > 0 and right_sum > 0) if require_two else (left_sum + right_sum > 0)

def _vectorize(res_h, res_f, mode: str, require_two=False):
    global last_valid
    hands_vec = _assemble_hands(res_h)
    if not _is_valid_hands(hands_vec, require_two=require_two):
        # relleno huecos con último válido (como interpolate del preprocess pero online)
        hands_vec = last_valid
    else:
        last_valid = hands_vec
    if mode == "hands":
        return hands_vec
    # modo legacy (1530)
    face_vec = _assemble_face(res_f)
    return np.concatenate([hands_vec, face_vec], axis=0).astype(np.float32)

# ==================== Tk UI ====================
root = tk.Tk()
root.title("LSP - Inferencia (solo manos)")

root.rowconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

panel = ttk.Frame(root, padding=8)
panel.grid(row=0, column=0, sticky="nsw")
video_frame = ttk.Frame(root)
video_frame.grid(row=0, column=1, sticky="nsew")
video_frame.rowconfigure(0, weight=1)
video_frame.columnconfigure(0, weight=1)

video_label = ttk.Label(video_frame)
video_label.grid(row=0, column=0, sticky="nsew")

ttk.Label(panel, text="Modelo:").grid(row=0, column=0, sticky="w")
model_var = tk.StringVar(value=MODEL_PATH)
entry_model = ttk.Entry(panel, textvariable=model_var, width=40, state="readonly")
entry_model.grid(row=0, column=1, columnspan=3, sticky="we")

btn_state = tk.StringVar(value="Pausar")
def toggle_run():
    global running
    running = not running
    btn_state.set("Reanudar" if not running else "Pausar")
ttk.Button(panel, textvariable=btn_state, command=toggle_run).grid(row=1, column=0, sticky="we", pady=(4,2))

def reset_buffer():
    buffer.clear(); pred_hist.clear()
ttk.Button(panel, text="Reset", command=reset_buffer).grid(row=1, column=1, sticky="we", pady=(4,2))

def toggle_fullscreen():
    global fullscreen
    fullscreen = not fullscreen
    root.attributes("-fullscreen", fullscreen)
ttk.Button(panel, text="Fullscreen", command=toggle_fullscreen).grid(row=1, column=2, sticky="we", pady=(4,2))

hand_var  = tk.BooleanVar(value=True)
face_var  = tk.BooleanVar(value=(FEATURE_MODE == "hands_face"))
force_var = tk.BooleanVar(value=False)
ttk.Checkbutton(panel, text="Ver manos",  variable=hand_var).grid(row=2, column=0, sticky="w")
ttk.Checkbutton(panel, text="Ver rostro", variable=face_var, state=("normal" if FEATURE_MODE == "hands_face" else "disabled")).grid(row=2, column=1, sticky="w")
ttk.Checkbutton(panel, text="Forzar 2 manos", variable=force_var).grid(row=2, column=2, sticky="w")

ttk.Label(panel, text="Umbral (%)").grid(row=3, column=0, sticky="w", pady=(6,0))
conf_var = tk.DoubleVar(value=50.0)
ttk.Scale(panel, from_=0, to=100, orient="horizontal", variable=conf_var).grid(row=3, column=1, columnspan=3, sticky="we", pady=(6,0))

ttk.Label(panel, text="Suavizado (ventanas)").grid(row=4, column=0, sticky="w")
smooth_var = tk.IntVar(value=5)
ttk.Scale(panel, from_=1, to=8, orient="horizontal", variable=smooth_var).grid(row=4, column=1, columnspan=3, sticky="we")

ttk.Label(panel, text="Cámara:").grid(row=5, column=0, sticky="w", pady=(6,0))
cam_index_var = tk.IntVar(value=args.camera)
cam_entry = ttk.Spinbox(panel, from_=0, to=5, textvariable=cam_index_var, width=5)
cam_entry.grid(row=5, column=1, sticky="w")

ttk.Label(panel, text="Resolución:").grid(row=6, column=0, sticky="w")
res_var = tk.StringVar(value=f"{args.width}x{args.height}")
resolutions = ["640x480","1280x720","1920x1080"]
res_combo = ttk.Combobox(panel, textvariable=res_var, values=resolutions, state="readonly", width=10)
res_combo.grid(row=6, column=1, sticky="w")

ttk.Label(panel, text=f"Input: T={SEQ_LENGTH}, F={FEAT_DIM}  |  Modo: {FEATURE_MODE}").grid(row=7, column=0, columnspan=4, sticky="w", pady=(6,0))
lbl_fps = ttk.Label(panel, text="FPS: --"); lbl_fps.grid(row=8, column=0, columnspan=4, sticky="w")
lbl_top = ttk.Label(panel, text="Predicción: --"); lbl_top.grid(row=9, column=0, columnspan=4, sticky="w")

for c in range(4): panel.columnconfigure(c, weight=1)

# ==================== Cámara robusta ====================
cap = None
def _open_camera(idx: int, width: int, height: int):
    global cap
    if cap is not None: cap.release()
    # DSHOW en Windows para más control de propiedades; MJPG y buffer bajo para latencia
    backend = cv2.CAP_DSHOW if os.name == "nt" else 0
    c = cv2.VideoCapture(idx, backend)
    try: c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except: pass
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    c.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(0.05)
    ok, _ = c.read()
    if not ok:
        c.release()
        return None, (0,0)
    aw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); ah = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(aw - width) > 16 or abs(ah - height) > 16:
        print(f"[AVISO] Se pidió {width}x{height}, pero la cámara devolvió {aw}x{ah}.")
    return c, (aw, ah)

def apply_camera_settings():
    global cap, hands_fast, hands_reacq, face
    try:
        w, h = map(int, res_var.get().split("x"))
    except:
        w, h = args.width, args.height
    cap_new, (aw, ah) = _open_camera(int(cam_index_var.get()), w, h)
    if cap_new is None:
        # fallback único a 640x480
        cap_new, (aw, ah) = _open_camera(int(cam_index_var.get()), 640, 480)
        if cap_new is None:
            messagebox.showerror("Cámara", f"No se pudo abrir la cámara {cam_index_var.get()}.")
            return
    cap = cap_new
    # recrea detectores con los parámetros por defecto (o podrías parametrizarlos si agregas entradas)
    hands_fast  = build_hands_fast()
    hands_reacq = build_hands_reacq()
    if FEATURE_MODE == "hands_face" and face is None:
        face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)
    reset_buffer()

ttk.Button(panel, text="Aplicar cámara", command=apply_camera_settings).grid(row=10, column=0, columnspan=4, sticky="we", pady=(6,0))

cap, (aw, ah) = _open_camera(args.camera, args.width, args.height)
if cap is None:
    # fallback
    cap, (aw, ah) = _open_camera(args.camera, 640, 480)
    if cap is None:
        raise SystemExit("No se pudo abrir la cámara.")

# ==================== Loop de actualización ====================
def _predict_and_render(frame_bgr):
    global _frames, _last_fps_t, fps_value

    frame_bgr = cv2.flip(frame_bgr, 1)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    res_h = hands_fast.process(rgb)
    if force_var.get():
        cnt = len(res_h.multi_hand_landmarks) if res_h.multi_hand_landmarks else 0
        if cnt < 2:
            res_h = hands_reacq.process(rgb)
    res_f = (face.process(rgb) if (FEATURE_MODE == "hands_face" and face is not None) else None)

    # Dibujo landmarks
    if hand_var.get() and res_h.multi_hand_landmarks:
        for hl in res_h.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_bgr, hl, mp.solutions.hands.HAND_CONNECTIONS)
    if FEATURE_MODE == "hands_face" and face is not None and res_f and res_f.multi_face_landmarks:
        for fl in res_f.multi_face_landmarks:
            mp_draw.draw_landmarks(frame_bgr, fl, mp.solutions.face_mesh.FACEMESH_TESSELATION)

    # Vector + buffer
    if running:
        vec = _vectorize(res_h, res_f, FEATURE_MODE, require_two=force_var.get())
        buffer.append(vec)

    # Predicción con estandarización igual a preprocess
    top_line = "—"
    if len(buffer) == SEQ_LENGTH:
        seq = np.array(buffer, dtype=np.float32)  # (T, F)
        # estándar: (T,F) -> (T,F) usando mean/std por feature
        seq = (seq - FEAT_MEAN[None, :]) / FEAT_STD[None, :]
        probs = model.predict(seq[None, ...], verbose=0)[0]  # (C,)

        pred_hist.append(probs)
        k = max(1, int(smooth_var.get()))
        smoothed = np.mean(list(pred_hist)[-k:], axis=0)

        idx_top = smoothed.argsort()[-min(3, len(LABELS)):][::-1]
        best_idx = int(idx_top[0]); best_p = float(smoothed[best_idx])
        thr = float(conf_var.get())/100.0
        top_line = f"{LABELS[best_idx]}  {best_p*100:.1f}%" if best_p>=thr \
                   else f"—  {best_p*100:.1f}% (<{int(thr*100)}%)"

        # Render top-3
        y0 = 30
        for j, i in enumerate(idx_top):
            txt = f"{j+1}. {LABELS[int(i)]}  {smoothed[int(i)]*100:.1f}%"
            cv2.putText(frame_bgr, txt, (10, y0+24*j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

    # FPS
    _frames += 1
    now = time.time()
    if now - _last_fps_t >= 1.0:
        fps_value = _frames / (now - _last_fps_t)
        _frames = 0
        _last_fps_t = now
    cv2.putText(frame_bgr, f"FPS: {fps_value:.1f}", (10, frame_bgr.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,200,30), 2)

    # Mostrar en Tk
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    lbl_fps.config(text=f"FPS: {fps_value:.1f}")
    lbl_top.config(text=f"Predicción: {top_line}")

def _loop():
    ok, frame = cap.read()
    if ok:
        _predict_and_render(frame)
    root.after(1, _loop)

def _on_close():
    try:
        if cap: cap.release()
    except: pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", _on_close)
root.after(1, _loop)
root.mainloop()
