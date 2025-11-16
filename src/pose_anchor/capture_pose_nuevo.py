# -*- coding: utf-8 -*-
"""
capture_pose.py — Captura unificada de secuencias LSP con ancla corporal (Pose+Hands)

Características:
- Un solo método de captura secuencial para TODAS las señas (estáticas y dinámicas).
- Pre-roll de 2 segundos antes de empezar a grabar.
- Ventana de duración fija (en segundos) configurable en la UI.
- Guarda:
    * .npy con forma (L, 126) = (L, 2 manos * 21 puntos * (x,y,z))
    * .json con metadatos (etiqueta, tipo_sena auto: 'estatica'/'dinamica', motion_energy, signer_id, etc.)
- Carpeta de salida: Proyecto_LSP_keras/coordenadas_pose/<ETIQUETA>/<ETIQUETA_###.npy>
"""

import os
import cv2
import json
import time
import unicodedata
from datetime import datetime

import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox

# -------------------------------------------------------------------
# Configuración de rutas (usa coordenadas_pose como antes)
# -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # Proyecto_LSP_keras
BASE_DIR = os.path.join(ROOT, "coordenadas_pose")
os.makedirs(BASE_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Parámetros globales
# -------------------------------------------------------------------
FEAT_DIM = 21 * 3 * 2         # 2 manos * 21 landmarks * (x,y,z) = 126
PRE_ROLL_SECS = 1             # cuenta atrás antes de empezar a grabar
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Umbral para clasificar secuencias en estáticas/dinámicas
MOTION_ENERGY_THR_STATIC = 0.015   # se puede ajustar más adelante

# -------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------
def sanitize_label(s: str) -> str:
    """Normaliza la etiqueta para usarla como nombre de carpeta/archivo."""
    s = s.strip().upper()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s)
                if not unicodedata.combining(c))
    s = ''.join(c if (c.isalnum() or c in (' ', '_', '-')) else '_' for c in s)
    return s.replace(' ', '_')

def ensure_label_dir(label_safe: str) -> str:
    """Crea (si no existe) y devuelve la carpeta de la etiqueta."""
    p = os.path.join(BASE_DIR, label_safe)
    os.makedirs(p, exist_ok=True)
    return p

def count_existing(label_safe: str) -> int:
    """Cuenta cuántos .npy existen para esa etiqueta (para numerar muestras)."""
    folder = ensure_label_dir(label_safe)
    return sum(
        1
        for f in os.listdir(folder)
        if f.lower().endswith(".npy") and f.startswith(label_safe + "_")
    )

def estimate_motion_energy(seq) -> float:
    """
    Estima la "energía de movimiento" de una secuencia:
    - seq: lista o array (L, FEAT_DIM)
    Devuelve la media de ||Δframe|| a lo largo del tiempo.
    """
    arr = np.asarray(seq, dtype=np.float32)
    if arr.shape[0] < 2:
        return 0.0
    diffs = arr[1:] - arr[:-1]        # (L-1, FEAT_DIM)
    norms = np.linalg.norm(diffs, axis=1)  # (L-1,)
    return float(norms.mean())

def classify_sequence_type(seq) -> (str, float):
    """
    Clasifica una secuencia en 'estatica' o 'dinamica' según su energía de movimiento.
    Devuelve (tipo_sena, motion_energy).
    """
    e = estimate_motion_energy(seq)
    tipo = "estatica" if e < MOTION_ENERGY_THR_STATIC else "dinamica"
    return tipo, e

# -------------------------------------------------------------------
# MediaPipe Pose + Hands
# -------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,   # más permisivo para no perder manos quietas
    model_complexity=0,
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Marco corporal suavizado
EMA_ALPHA = 0.25
_body_center = None   # (cx, cy) en [0,1]
_body_scale = None    # escalar basado en ancho de hombros

def get_body_frame(res_p):
    """Calcula centro (torso) y escala (ancho de hombros) con suavizado exponencial."""
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
        _body_center = (1.0 - EMA_ALPHA) * _body_center + EMA_ALPHA * center
        _body_scale = (1.0 - EMA_ALPHA) * _body_scale + EMA_ALPHA * scale

    return _body_center, _body_scale

def hands_to_body_relative(res_h, center, scale):
    """
    Devuelve un vector de longitud 126:
      [mano_izq(21*3), mano_der(21*3)]
    Cada coord (x,y,z) está normalizada respecto al torso: (x-cx)/s, (y-cy)/s, z/s.
    """
    left = [0.0] * 63
    right = [0.0] * 63

    if (res_h and res_h.multi_hand_landmarks and res_h.multi_handedness and
            center is not None and scale is not None):

        cx, cy = center
        s = max(1e-3, float(scale))

        for hand_lms, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords = []
            for p in hand_lms.landmark:
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

    return left + right  # 126

def is_valid(vec126, require_two=False):
    """
    Comprueba si el vector tiene señal (al menos una mano, o ambas si require_two=True).
    """
    v = np.asarray(vec126, dtype=np.float32)
    l = float(np.sum(np.abs(v[:63])))
    r = float(np.sum(np.abs(v[63:])))
    return (l > 0.0 and r > 0.0) if require_two else (l + r > 0.0)

# -------------------------------------------------------------------
# UI con Tkinter
# -------------------------------------------------------------------
root = tk.Tk()
root.title("Captura LSP (Pose Anchor — Unificado)")

label_display = None   # etiqueta tal cual la escribe el usuario
label_safe = None      # etiqueta normalizada para nombre de archivo
samples_per_label = {}

signer_id_var = tk.StringVar(value="S1")
dom_hand_var = tk.StringVar(value="right")   # "right" o "left"
dyn_secs = tk.IntVar(value=4)                # duración de ventana (segundos)
force_two = tk.BooleanVar(value=False)       # exigir 2 manos visibles

# Frame principal
frm = ttk.Frame(root, padding=8)
frm.grid(row=0, column=0, sticky="nsew")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Label de seña
ttk.Label(frm, text="Etiqueta (ej. LETRA_A):").grid(row=0, column=0, sticky="w")
entry_label = ttk.Entry(frm, width=28)
entry_label.grid(row=0, column=1, sticky="w")

def log(msg: str):
    txt_log.config(state="normal")
    txt_log.insert("end", msg + "\n")
    txt_log.see("end")
    txt_log.config(state="disabled")

def set_label():
    global label_display, label_safe, samples_per_label
    t = entry_label.get().strip()
    if not t:
        messagebox.showwarning("Etiqueta", "Ingresa una etiqueta.")
        return
    label_display = t
    label_safe = sanitize_label(t)
    ensure_label_dir(label_safe)
    samples_per_label[label_safe] = count_existing(label_safe)
    log(f"Etiqueta establecida: {label_display}  (safe={label_safe})  muestras={samples_per_label[label_safe]}")

btn_set_label = ttk.Button(frm, text="Establecer", command=set_label)
btn_set_label.grid(row=0, column=2, padx=6, sticky="w")

# Duración y manos
ttk.Label(frm, text="Duración ventana (segundos):").grid(row=1, column=0, sticky="w")
spin_secs = ttk.Spinbox(frm, from_=1, to=10, textvariable=dyn_secs, width=5)
spin_secs.grid(row=1, column=1, sticky="w")

ttk.Checkbutton(frm, text="Requiere 2 manos visibles", variable=force_two).grid(row=1, column=2, sticky="w")

# Info del señante
ttk.Label(frm, text="ID del señante:").grid(row=2, column=0, sticky="w")
entry_signer = ttk.Entry(frm, width=12, textvariable=signer_id_var)
entry_signer.grid(row=2, column=1, sticky="w")

ttk.Label(frm, text="Mano dominante:").grid(row=2, column=2, sticky="w")
frame_dom = ttk.Frame(frm)
frame_dom.grid(row=2, column=3, sticky="w")
ttk.Radiobutton(frame_dom, text="Derecha", variable=dom_hand_var, value="right").pack(side="left")
ttk.Radiobutton(frame_dom, text="Izquierda", variable=dom_hand_var, value="left").pack(side="left")

# Log de texto
txt_log = tk.Text(frm, height=8, width=90, state="disabled")
txt_log.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
frm.rowconfigure(3, weight=1)

log("Uso:")
log(" 1) Escribe la etiqueta y pulsa 'Establecer'.")
log(" 2) Ajusta duración de ventana y si requieres 2 manos.")
log(" 3) Colócate en posición, pulsa ENTER para iniciar pre-roll (2s).")
log(" 4) Cuando aparezca 'YA', ejecuta la seña (o mantenla si es estática).")
log(" 5) Se guardará .npy + .json en coordenadas_pose/<ETIQUETA>/")

# -------------------------------------------------------------------
# Guardado de secuencias
# -------------------------------------------------------------------
def save_seq(seq_frames, duration, fps_est):
    """
    Guarda la secuencia en npy + json con metadatos.
    - seq_frames: lista de vectores 126 (L, 126)
    """
    global label_display, label_safe, samples_per_label

    if label_safe is None or label_display is None:
        log("⚠ Establece una etiqueta antes de guardar.")
        return

    arr = np.asarray(seq_frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != FEAT_DIM:
        log(f"⚠ Forma inesperada de secuencia: {arr.shape}")
        return

    # Clasificación automática por energía de movimiento
    tipo_sena, motion_energy = classify_sequence_type(arr)

    folder = ensure_label_dir(label_safe)
    idx = samples_per_label.get(label_safe, 0) + 1
    npy_path = os.path.join(folder, f"{label_safe}_{idx:03d}.npy")
    json_path = npy_path.replace(".npy", ".json")

    np.save(npy_path, arr)
    samples_per_label[label_safe] = idx

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": SESSION_ID,
        "label_display": label_display,
        "label_safe": label_safe,
        "frames": int(arr.shape[0]),
        "features": int(arr.shape[1]),
        "feature_type": "hands_body_relative",
        "pose_anchors": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
        "capture_mode": "ventana_temporal",
        "tipo_sena": tipo_sena,               # 'estatica' o 'dinamica' (auto)
        "motion_energy": float(motion_energy),
        "duration_secs": float(duration),
        "fps_est": float(fps_est),
        "signer_id": signer_id_var.get().strip() or "S1",
        "dominant_hand": dom_hand_var.get(),
        "require_two_hands": bool(force_two.get()),
        "camera": {
            "width": int(CAM_W),
            "height": int(CAM_H),
            "fps_nominal": int(CAM_FPS),
        },
        "script_version": "capture_pose_unificado_v2",
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f"✓ Guardado {os.path.basename(npy_path)}  L={arr.shape[0]} F={arr.shape[1]}  tipo_sena={tipo_sena}  E={motion_energy:.4f}")

# -------------------------------------------------------------------
# Captura con OpenCV
# -------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cv2.namedWindow("Pose-Anchor Capture", cv2.WINDOW_NORMAL)

recording = False
prerec = False
rec_start = 0.0
prerec_start = 0.0
seq_frames = []

last_fps_t = time.time()
frames = 0
fps = 0.0

def draw_bar(img, cur, total, color=(0, 200, 0), txt=""):
    h, w = img.shape[:2]
    bw = int(w * 0.6)
    bh = 14
    x = int((w - bw) / 2)
    y = h - 30
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 60), -1)
    fi = int(bw * min(cur, total) / max(total, 1))
    cv2.rectangle(img, (x, y), (x + fi, y + bh), color, -1)
    if txt:
        cv2.putText(
            img,
            txt,
            (x + bw + 10, y + bh - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

while True:
    # mantienen viva la UI de Tkinter
    root.update()

    ok, frame = cap.read()
    if not ok:
        log("⚠ No se pudo leer de la cámara.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose + manos
    res_p = pose.process(rgb)
    center, scale = get_body_frame(res_p)
    res_h = hands.process(rgb)

    # Vector de características
    feat = hands_to_body_relative(res_h, center, scale)

    # Dibujar pose/manos
    if res_p and res_p.pose_landmarks:
        mp_draw.draw_landmarks(frame, res_p.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if res_h and res_h.multi_hand_landmarks:
        for hl in res_h.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # FPS
    frames += 1
    now = time.time()
    if now - last_fps_t >= 1.0:
        fps = frames / (now - last_fps_t + 1e-6)
        frames = 0
        last_fps_t = now
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (30, 200, 30),
        2,
    )

    # Pre-roll (cuenta regresiva)
    if prerec:
        rem = PRE_ROLL_SECS - (now - prerec_start)
        txtc = "¡YA!" if rem <= 0 else f"{int(np.ceil(rem))}"
        cv2.putText(
            frame,
            txtc,
            (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
        )
        draw_bar(
            frame,
            PRE_ROLL_SECS - rem,
            PRE_ROLL_SECS,
            (0, 0, 255),
            f"Comienza en {max(0.0, rem):.1f}s",
        )
        if rem <= 0:
            prerec = False
            recording = True
            rec_start = now
            seq_frames = []

    # Grabación de ventana
    if recording:
        # Sólo guardamos si hay señal (y opcionalmente 2 manos)
        if is_valid(feat, require_two=force_two.get()):
            seq_frames.append(feat)

        elapsed = now - rec_start
        total = max(1, int(dyn_secs.get()))
        draw_bar(
            frame,
            elapsed,
            total,
            (0, 0, 255),
            f"{elapsed:.1f}/{total:.0f}s",
        )

        if elapsed >= total:
            recording = False
            if len(seq_frames) < 10:
                log(f"⚠ Grabación muy corta ({len(seq_frames)} frames). Ignorada.")
            else:
                dur = elapsed
                fps_est = len(seq_frames) / max(dur, 1e-6)
                save_seq(seq_frames, dur, fps_est)
                seq_frames = []

    cv2.imshow("Pose-Anchor Capture", frame)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:  # ESC
        break
    if k == 13:  # ENTER -> iniciar pre-roll
        if not label_safe:
            log("⚠ Establece una etiqueta primero.")
            continue
        # opcional: comprobar manos visibles antes de empezar
        if force_two.get() and (not is_valid(feat, require_two=True)):
            log("⚠ Se requieren 2 manos visibles para esta muestra.")
            continue
        prerec = True
        prerec_start = time.time()
        log(f"Iniciando captura con pre-roll de {PRE_ROLL_SECS}s y ventana de {dyn_secs.get()}s...")

    # Atajo rápido: tecla alfanumérica = nueva etiqueta con ese carácter
    if (65 <= k <= 90) or (97 <= k <= 122) or (48 <= k <= 57):
        entry_label.delete(0, "end")
        entry_label.insert(0, chr(k).upper())
        set_label()

cap.release()
cv2.destroyAllWindows()
try:
    root.destroy()
except Exception:
    pass
