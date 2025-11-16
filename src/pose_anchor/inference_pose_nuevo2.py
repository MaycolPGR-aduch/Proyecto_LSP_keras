# -*- coding: utf-8 -*-
"""
inference_pose_unificado.py

Inferencia en tiempo real para señas LSP con el flujo unificado:
- capture_pose.py  -> coordenadas relativas al cuerpo (126 dims)
- preprocess_pose.py -> re-muestreo + (opcional) jitter
- train_pose_unificado.py -> modelo Conv1D + scaler + labels en models_pose/

Características:
- Usa Pose + Hands para obtener 126 features: 2 manos * 21 pts * (x,y,z) relativos al torso.
- Ventana temporal fija (SEQ_LENGTH = input del modelo).
- Suavizado temporal de probabilidades.
- Detección de seña "estable" diferenciando estáticas y dinámicas.
- Interfaz gráfica con Tkinter:
  * Video a la izquierda.
  * Controles a la derecha (Iniciar/Detener, Reset, sliders de umbral y suavizado).
  * Etiqueta grande con la seña aceptada.
"""

import os
import argparse
from collections import deque
import time

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json

# ---------------------------------------------------------------------
# Paths y argparse
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # Proyecto_LSP_keras
MODELS_DIR = os.path.join(ROOT, "models_pose")

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODELS_DIR, "lsp_pose_v1.keras"),
        help="Ruta al modelo .keras entrenado",
    )
    p.add_argument("--cam", type=int, default=0, help="ID de cámara")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--stride", type=int, default=1, help="Procesar 1 de cada N frames (opcional)")
    return p

args = build_argparser().parse_args()
MODEL_PATH = os.path.abspath(args.model)
MODEL_DIR = os.path.dirname(MODEL_PATH)

# ---------------------------------------------------------------------
# Carga de modelo, scaler y labels
# ---------------------------------------------------------------------
print(f"[INFO] Cargando modelo desde {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    _, SEQ_LENGTH, FEAT_DIM = model.input_shape
except ValueError:
    raise ValueError(f"[ERROR] input_shape inesperado: {model.input_shape}")

print(f"[INFO] Modelo cargado. SEQ_LENGTH={SEQ_LENGTH}, FEAT_DIM={FEAT_DIM}")

if FEAT_DIM != 126:
    print(f"[ADVERTENCIA] El modelo espera FEAT_DIM={FEAT_DIM}, "
          f"pero la tubería actual genera 126 features.")

scaler_path = os.path.join(MODEL_DIR, "scaler_stats.npz")
stats = np.load(scaler_path)
MEAN = stats["mean"].astype(np.float32)
STD = stats["std"].astype(np.float32)
STD[STD < 1e-8] = 1.0
print(f"[INFO] scaler_stats cargado de: {scaler_path}")

labels_path = os.path.join(MODEL_DIR, "labels.json")
with open(labels_path, "r", encoding="utf-8") as f:
    data_l = json.load(f)
    LABELS = data_l["labels"] if isinstance(data_l, dict) and "labels" in data_l else data_l
NUM_CLASSES = len(LABELS)
print(f"[INFO] {NUM_CLASSES} clases: {LABELS}")

# ---------------------------------------------------------------------
# MediaPipe Pose + Hands
# ---------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,  # más permisivo para no perder manos quietas
)

# Marco corporal (suavizado)
EMA_ALPHA_CENTER = 0.25
_body_center = None   # (cx,cy)
_body_scale = None    # escala basada en hombros

def get_body_frame(res_p):
    """Devuelve (center, scale) del torso con suavizado exponencial."""
    global _body_center, _body_scale
    if not res_p or not res_p.pose_landmarks:
        return _body_center, _body_scale

    lm = res_p.pose_landmarks.landmark
    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LHP = mp_pose.PoseLandmark.LEFT_HIP.value
    RHP = mp_pose.PoseLandmark.RIGHT_HIP.value

    def xy(i):
        p = lm[i]
        return float(np.clip(p.x, 0.0, 1.0)), float(np.clip(p.y, 0.0, 1.0))

    lsh = xy(LSH)
    rsh = xy(RSH)
    lhp = xy(LHP)
    rhp = xy(RHP)

    cx = (lsh[0] + rsh[0] + lhp[0] + rhp[0]) / 4.0
    cy = (lsh[1] + rsh[1] + lhp[1] + rhp[1]) / 4.0
    center = np.array([cx, cy], dtype=np.float32)

    shoulder_width = np.linalg.norm(np.array(lsh) - np.array(rsh)) + 1e-6
    scale = float(shoulder_width)

    if _body_center is None:
        _body_center = center
        _body_scale = scale
    else:
        _body_center = (1.0 - EMA_ALPHA_CENTER) * _body_center + EMA_ALPHA_CENTER * center
        _body_scale = (1.0 - EMA_ALPHA_CENTER) * _body_scale + EMA_ALPHA_CENTER * scale

    return _body_center, _body_scale

def hands_to_body_relative(res_h, center, scale):
    """
    Vector de 126 features:
      [mano_izq(21*3), mano_der(21*3)]
    coords relativas al torso: (x-cx)/s, (y-cy)/s, z/s, con clipping.
    """
    left = [0.0] * 63
    right = [0.0] * 63

    if (res_h and res_h.multi_hand_landmarks and res_h.multi_handedness and
            center is not None and scale is not None):
        cx, cy = center
        s = max(1e-3, float(scale))

        for hland, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords = []
            for p in hland.landmark:
                x = (float(np.clip(p.x, 0.0, 1.0)) - cx) / s
                y = (float(np.clip(p.y, 0.0, 1.0)) - cy) / s
                z = float(np.clip(p.z, -1.0, 1.0)) / s
                coords.extend([
                    float(np.clip(x, -2.0, 2.0)),
                    float(np.clip(y, -2.0, 2.0)),
                    float(np.clip(z, -2.0, 2.0)),
                ])
            if handed.classification[0].label.lower() == "left":
                left = coords[:63]
            else:
                right = coords[:63]

    return np.array(left + right, dtype=np.float32)  # (126,)

def is_valid(vec126, min_mag=1e-4):
    """Chequea si al menos una mano tiene magnitud suficiente."""
    if vec126 is None:
        return False
    v = np.asarray(vec126, dtype=np.float32)
    if np.allclose(v, 0.0):
        return False
    mag = float(np.mean(np.abs(v)))
    return mag > min_mag

# ---------------------------------------------------------------------
# Lógica de detección estable (estáticas vs dinámicas)
# ---------------------------------------------------------------------
EPS_STILL = 0.006       # umbral de "quietud"
K_STATIC = 8            # frames consecutivos para estáticas
K_DYNAMIC = 6           # para dinámicas
DEBOUNCE_STATIC = 45    # evitar repeticiones seguidas
DEBOUNCE_DYNAMIC = 30

ema_probs = None
consec_ok = 0
last_emit = {"label": None, "frame_idx": -9999}

prev_vec = None
last_valid_vec = None
still_hist = deque(maxlen=SEQ_LENGTH)
hist_probs = deque(maxlen=16)
buffer = deque(maxlen=SEQ_LENGTH)

step = 0  # índice global de frames válidos usados

def reset_detection_state():
    """Resetea todo el estado de detección (para botón Reset)."""
    global ema_probs, consec_ok, last_emit
    global prev_vec, last_valid_vec, still_hist, hist_probs, buffer, step

    ema_probs = None
    consec_ok = 0
    last_emit = {"label": None, "frame_idx": -9999}

    prev_vec = None
    last_valid_vec = None
    still_hist.clear()
    hist_probs.clear()
    buffer.clear()
    step = 0

def compute_stillness(prev_vec126, curr_vec126):
    """
    Mide velocidad media sobre 6 puntos clave:
      wrist(0), thumb_tip(4), index_tip(8) en ambas manos.
    """
    if prev_vec126 is None or curr_vec126 is None:
        return 0.0
    a = np.asarray(prev_vec126, dtype=np.float32).reshape(2, 21, 3)  # (2,21,3)
    b = np.asarray(curr_vec126, dtype=np.float32).reshape(2, 21, 3)

    idxs = [0, 4, 8]
    pts_a = np.concatenate([a[:, i, :] for i in idxs], axis=0)  # (6,3)
    pts_b = np.concatenate([b[:, i, :] for i in idxs], axis=0)

    dv = pts_b - pts_a
    speed = np.linalg.norm(dv, axis=1).mean()
    return float(speed)

def ema_update(probs, ema, alpha):
    if ema is None:
        return probs.copy()
    return alpha * ema + (1.0 - alpha) * probs

def decide_label(smoothed_probs, class_names, stillness, frame_idx, thr_ui):
    """
    Decide etiqueta de forma estable:
    - Umbral diferente para estáticas y dinámicas.
    - Número mínimo de frames consecutivos (K_STATIC / K_DYNAMIC).
    - Debounce para evitar repeticiones.
    """
    global ema_probs, consec_ok, last_emit

    thr_dyn = float(thr_ui)
    thr_stat = max(0.40, min(thr_dyn - 0.10, 0.85))

    is_static = (stillness < EPS_STILL)
    Kneed = K_STATIC if is_static else K_DYNAMIC
    debounce = DEBOUNCE_STATIC if is_static else DEBOUNCE_DYNAMIC

    alpha = 0.6 if is_static else 0.8
    ema = ema_update(smoothed_probs, ema_probs, alpha)
    ema_probs = ema

    top_idx = int(np.argmax(ema))
    label = class_names[top_idx]
    p = float(ema[top_idx])

    if "no_sena" in class_names and label == "no_sena":
        consec_ok = 0
        return None

    thr = thr_stat if is_static else thr_dyn

    if p >= thr:
        consec_ok += 1
    else:
        consec_ok = 0

    if consec_ok >= Kneed:
        # debounce
        if label == last_emit["label"] and (frame_idx - last_emit["frame_idx"]) < debounce:
            return None
        last_emit = {"label": label, "frame_idx": frame_idx}
        consec_ok = 0
        return label

    return None

# ---------------------------------------------------------------------
# Cámara
# ---------------------------------------------------------------------
cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara {args.cam}")

frames_count = 0
t0 = time.time()
fps = 0.0

# ---------------------------------------------------------------------
# UI con Tkinter (interfaz más pulida)
# ---------------------------------------------------------------------
root = tk.Tk()
root.title("Intérprete LSP - Inferencia en tiempo real")

# Estilo básico
style = ttk.Style(root)
try:
    style.theme_use("clam")
except Exception:
    pass

main_frame = ttk.Frame(root, padding=8)
main_frame.grid(row=0, column=0, sticky="nsew")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Subframes: video a la izquierda, controles a la derecha
left_frame = ttk.Frame(main_frame)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
right_frame = ttk.LabelFrame(main_frame, text="Controles", padding=8)
right_frame.grid(row=0, column=1, sticky="nsew")

main_frame.columnconfigure(0, weight=3)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)

# Video
video_label = tk.Label(left_frame, bd=1, relief="sunken")
video_label.pack(fill="both", expand=True)

# Estado
running = {"flag": False}

def toggle_run():
    running["flag"] = not running["flag"]
    btn_run.config(text="Detener" if running["flag"] else "Iniciar")
    if running["flag"]:
        status_var.set("Detección en curso...")
    else:
        status_var.set("Detección en pausa.")

def do_reset():
    reset_detection_state()
    current_label_var.set("—")
    info_var.set("Esperando seña...")
    status_var.set("Estado reseteado.")

btn_run = ttk.Button(right_frame, text="Iniciar", command=toggle_run)
btn_run.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

btn_reset = ttk.Button(right_frame, text="Resetear modelo", command=do_reset)
btn_reset.grid(row=0, column=2, sticky="ew", padx=(6, 0))

# Sliders
ttk.Label(right_frame, text="Umbral confianza (estables):").grid(row=1, column=0, columnspan=3, sticky="w")
thr_var = tk.DoubleVar(value=0.75)
thr_slider = ttk.Scale(
    right_frame,
    from_=0.40,
    to=0.95,
    orient="horizontal",
    variable=thr_var,
)
thr_slider.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 6))

ttk.Label(right_frame, text="Suavizado temporal (frames):").grid(row=3, column=0, columnspan=3, sticky="w")
smooth_var = tk.IntVar(value=4)
smooth_slider = ttk.Scale(
    right_frame,
    from_=1,
    to=10,
    orient="horizontal",
    variable=smooth_var,
)
smooth_slider.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 6))

# Info de predicción
ttk.Label(right_frame, text="Seña detectada:").grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
current_label_var = tk.StringVar(value="—")
lbl_current = ttk.Label(
    right_frame,
    textvariable=current_label_var,
    font=("Segoe UI", 16, "bold"),
    foreground="#0078D7",
)
lbl_current.grid(row=6, column=0, columnspan=3, sticky="w")

info_var = tk.StringVar(value="Esperando seña...")
lbl_info = ttk.Label(right_frame, textvariable=info_var, wraplength=260, justify="left")
lbl_info.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))

# FPS / estado
fps_var = tk.StringVar(value="FPS: 0.0")
lbl_fps = ttk.Label(right_frame, textvariable=fps_var)
lbl_fps.grid(row=8, column=0, sticky="w", pady=(8, 0))

status_var = tk.StringVar(value="Detección en pausa.")
lbl_status = ttk.Label(right_frame, textvariable=status_var, foreground="#555555")
lbl_status.grid(row=8, column=1, columnspan=2, sticky="e", pady=(8, 0))

for c in range(3):
    right_frame.columnconfigure(c, weight=1)

# ---------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------
def loop():
    global frames_count, t0, fps
    global prev_vec, last_valid_vec, step

    ret, frame = cap.read()
    if not ret:
        status_var.set("No se pudo leer de la cámara.")
        root.after(30, loop)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_p = pose.process(rgb)
    center, scale = get_body_frame(res_p)
    res_h = hands.process(rgb)

    # Dibujo de pose y manos
    if res_p and res_p.pose_landmarks:
        mp_drawing.draw_landmarks(frame, res_p.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if res_h and res_h.multi_hand_landmarks:
        for hl in res_h.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # Vector de características relativo al cuerpo
    raw_vec = hands_to_body_relative(res_h, center, scale)

    if running["flag"]:
        # Llenado del buffer con manejo de huecos (last_valid_vec)
        if is_valid(raw_vec):
            vec = raw_vec
            last_valid_vec = vec.copy()
        else:
            if last_valid_vec is None:
                # no tenemos aún nada válido
                vec = None
                still_hist.append(0.0)
            else:
                vec = last_valid_vec.copy()

        if vec is not None:
            if prev_vec is not None:
                still = compute_stillness(prev_vec, vec)
            else:
                still = 0.0
            still_hist.append(still)
            prev_vec = vec.copy()
            buffer.append(vec)
            step += 1

        # Inferencia cuando la ventana está llena
        if len(buffer) == SEQ_LENGTH:
            seq = np.asarray(buffer, dtype=np.float32)
            seq_norm = (seq - MEAN[None, :]) / STD[None, :]
            x = seq_norm[None, ...]  # (1,T,F)

            probs = model(x, training=False).numpy()[0]
            hist_probs.append(probs)

            k = max(1, int(smooth_var.get()))
            sm = np.mean(list(hist_probs)[-k:], axis=0)

            still_window = float(np.mean(list(still_hist)[-k:])) if len(still_hist) >= 2 else 0.0
            thr_ui = float(thr_var.get())

            label_dec = decide_label(sm, LABELS, still_window, step, thr_ui)

            # Top-3 para overlay
            top = sm.argsort()[-min(3, len(LABELS)):][::-1]
            for j, i in enumerate(top):
                txt = f"{j+1}. {LABELS[int(i)]}: {sm[int(i)]:.2f}"
                cv2.putText(
                    frame,
                    txt,
                    (10, 30 + 24 * j),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if label_dec is not None:
                current_label_var.set(label_dec)
                info_var.set(f"Seña detectada de forma estable.\n(Umbral={thr_ui:.2f}, suavizado={k})")
                cv2.putText(
                    frame,
                    label_dec,
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
        else:
            # Todavía llenando ventana
            if running["flag"]:
                info_var.set(f"Llenando ventana: {len(buffer)}/{SEQ_LENGTH} frames ...")
    else:
        info_var.set("Detección en pausa. Pulsa 'Iniciar' para comenzar.")

    # FPS
    frames_count += 1
    if frames_count >= 10:
        t1 = time.time()
        fps = frames_count / (t1 - t0 + 1e-6)
        t0 = t1
        frames_count = 0
    fps_var.set(f"FPS: {fps:.1f}")

    # Mostrar frame en la UI
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(1, loop)

def on_close():
    try:
        cap.release()
    except Exception:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
reset_detection_state()
root.after(1, loop)
root.mainloop()
