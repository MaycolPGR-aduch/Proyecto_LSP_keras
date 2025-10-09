# -*- coding: utf-8 -*-
"""
capture_pose.py — Captura de secuencias con ancla corporal (MediaPipe Pose)
- Calcula un marco corporal usando los hombros y caderas.
- Convierte los 21*2 landmarks de mano (x,y,z) a coordenadas relativas al cuerpo.
- Guarda .npy (L,126) + .json sin tocar tu flujo principal.
"""

import os, cv2, json, time, unicodedata
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import mediapipe as mp
from datetime import datetime

# -------------------- Config --------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", "coordenadas_pose"))
os.makedirs(BASE_DIR, exist_ok=True)

FEAT_DIM = 21*3*2   # 126
STATIC_FRAMES = 30
PRE_ROLL_SECS = 2
REQUIRE_TWO_HANDS = False

# cámara
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30

# -------------------- GUI --------------------
root = tk.Tk()
root.title("Captura LSP (Pose Anchor)")

mode = tk.StringVar(value="static")     # "static" | "dynamic"
dyn_secs = tk.IntVar(value=4)
force_two = tk.BooleanVar(value=REQUIRE_TWO_HANDS)

label_display = None
label_safe = None
samples = {}
buffer_static = deque(maxlen=STATIC_FRAMES)

def sanitize(s: str) -> str:
    s = s.strip().upper()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = ''.join(c if (c.isalnum() or c in (' ','_','-')) else '_' for c in s)
    s = s.replace(' ','_')
    while '__' in s: s=s.replace('__','_')
    return s

def ensure_label_dir(lab: str):
    p = os.path.join(BASE_DIR, lab); os.makedirs(p, exist_ok=True); return p

def get_count(lab: str) -> int:
    p = ensure_label_dir(lab)
    return sum(1 for f in os.listdir(p) if f.lower().endswith('.npy') and f.startswith(lab+'_'))

def set_label():
    global label_display, label_safe, buffer_static
    t = entry_label.get().strip()
    if not t:
        messagebox.showwarning("Etiqueta", "Ingresa una etiqueta.")
        return
    label_display = t; label_safe = sanitize(t)
    ensure_label_dir(label_safe)
    samples[label_safe] = get_count(label_safe)
    buffer_static.clear()
    log(f"Etiqueta: {label_display}  (safe={label_safe})  muestras={samples[label_safe]}")

def save_seq(label_safe, label_display, seq, capture_mode, duration, fps_est):
    arr = np.asarray(seq, dtype=np.float32)
    folder = ensure_label_dir(label_safe)
    idx = samples[label_safe] + 1
    npy = os.path.join(folder, f"{label_safe}_{idx:03d}.npy")
    jsn = npy.replace(".npy", ".json")
    np.save(npy, arr)
    samples[label_safe] = idx

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label_display": label_display,
        "label_safe": label_safe,
        "frames": int(arr.shape[0]),
        "features": FEAT_DIM,
        "include_face": False,
        "capture_mode": capture_mode,
        "duration_secs": float(duration),
        "fps_est": float(fps_est),
        "feature_type": "hands_body_relative",
        "pose_anchors": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
    }
    json.dump(meta, open(jsn,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"✓ Guardado {npy}  L={arr.shape[0]} F={arr.shape[1]}")

# -------------------- MediaPipe --------------------
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.3,
                       model_complexity=0)
pose  = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                     enable_segmentation=False, min_detection_confidence=0.5)

# -------------------- Marco corporal --------------------
# Tomamos hombros y caderas para definir centro y escala.
EMA_ALPHA = 0.25
body_center = None   # (cx, cy)
body_scale  = None   # escalar >0

def get_body_frame(res_p):
    """
    Devuelve (cx, cy, scale) en coords normalizadas imagen.
    cx, cy: centro entre hombros y caderas.
    scale : tamaño torso (distancia media hombro-hip / diagonales).
    """
    global body_center, body_scale
    if not res_p or not res_p.pose_landmarks: 
        return body_center, body_scale
    lm = res_p.pose_landmarks.landmark
    def xy(idx): 
        p = lm[idx]; return float(np.clip(p.x,0,1)), float(np.clip(p.y,0,1))
    # keypoints
    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LHP = mp_pose.PoseLandmark.LEFT_HIP.value
    RHP = mp_pose.PoseLandmark.RIGHT_HIP.value
    lsh = xy(LSH); rsh = xy(RSH); lhp = xy(LHP); rhp = xy(RHP)
    # centro: promedio de hombros y caderas
    cx = (lsh[0]+rsh[0]+lhp[0]+rhp[0])/4.0
    cy = (lsh[1]+rsh[1]+lhp[1]+rhp[1])/4.0
    # escala: combinación de distancia hombros y hombro-hip
    dist_shoulders = np.hypot(lsh[0]-rsh[0], lsh[1]-rsh[1])
    dist_torso = ( np.hypot(lsh[0]-lhp[0], lsh[1]-lhp[1])
                 + np.hypot(rsh[0]-rhp[0], rsh[1]-rhp[1]) ) / 2.0
    scale = max(1e-3, 0.6*dist_shoulders + 0.4*dist_torso)

    # suavizado exponencial
    if body_center is None: body_center = (cx, cy)
    else:
        body_center = ( (1-EMA_ALPHA)*body_center[0] + EMA_ALPHA*cx,
                        (1-EMA_ALPHA)*body_center[1] + EMA_ALPHA*cy )
    if body_scale is None: body_scale = scale
    else:
        body_scale = (1-EMA_ALPHA)*body_scale + EMA_ALPHA*scale

    return body_center, body_scale

def hands_to_body_relative(res_h, center, scale):
    """
    Convierte landmarks de manos a marco corporal:
    (x',y',z') = ((x-cx)/s, (y-cy)/s, z/s)  truncado a [-2,2]
    Retorna vector de 126 (Left + Right).
    """
    left = [0.0]*63; right=[0.0]*63
    if res_h.multi_hand_landmarks and res_h.multi_handedness and center and scale:
        cx, cy = center; s = max(1e-3, float(scale))
        for lms, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords=[]
            for p in lms.landmark:
                x = (float(np.clip(p.x,0,1)) - cx)/s
                y = (float(np.clip(p.y,0,1)) - cy)/s
                z = float(np.clip(p.z,-1,1))/s
                # limitar rango para robustez
                coords.extend([float(np.clip(x,-2,2)), float(np.clip(y,-2,2)), float(np.clip(z,-2,2))])
            if handed.classification[0].label == "Left":  left  = coords[:63]
            else:                                         right = coords[:63]
    return left + right

def is_valid(vec126, require_two=False):
    l = float(np.sum(np.abs(vec126[:63]))); r = float(np.sum(np.abs(vec126[63:])))
    return (l>0 and r>0) if require_two else (l+r>0)

# -------------------- Ventana / Controles --------------------
frm = ttk.Frame(root, padding=8); frm.grid(row=0,column=0,sticky="nsew")
root.rowconfigure(0,weight=1); root.columnconfigure(0,weight=1)

ttk.Label(frm, text="Etiqueta:").grid(row=0, column=0, sticky="w")
entry_label = ttk.Entry(frm, width=28); entry_label.grid(row=0,column=1,sticky="w")
ttk.Button(frm, text="Establecer", command=set_label).grid(row=0,column=2, padx=6, sticky="w")

ttk.Label(frm, text="Modo:").grid(row=1,column=0,sticky="w")
ttk.Radiobutton(frm, text="Estática (30f)", variable=mode, value="static").grid(row=1,column=1,sticky="w")
ttk.Radiobutton(frm, text="Dinámica (N s)", variable=mode, value="dynamic").grid(row=1,column=2,sticky="w")
ttk.Label(frm, text="Segundos:").grid(row=2,column=0,sticky="w")
spin_secs = ttk.Spinbox(frm, from_=1,to=10,textvariable=dyn_secs, width=5); spin_secs.grid(row=2,column=1,sticky="w")
ttk.Checkbutton(frm, text="Requiere 2 manos", variable=force_two).grid(row=2,column=2,sticky="w")

txt = tk.Text(frm, height=8, width=80, state="disabled"); txt.grid(row=3,column=0,columnspan=3,sticky="nsew", pady=(8,0))
frm.rowconfigure(3, weight=1)

def log(m): txt.config(state="normal"); txt.insert("end",m+"\n"); txt.see("end"); txt.config(state="disabled")

# -------------------- Loop de cámara --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cv2.namedWindow("Pose-Anchor Capture", cv2.WINDOW_NORMAL)

recording=False; prerec=False; rec_start=0.0; prerec_start=0.0; dyn_frames=[]
last_fps_t=time.time(); frames=0; fps=0.0

def draw_bar(img, cur, total, color=(0,200,0), txt=""):
    h,w=img.shape[:2]; bw=int(w*0.6); bh=14; x=int((w-bw)/2); y=h-30
    cv2.rectangle(img,(x,y),(x+bw,y+bh),(60,60,60),-1)
    fi=int(bw*min(cur,total)/max(total,1))
    cv2.rectangle(img,(x,y),(x+fi,y+bh),color,-1)
    if txt: cv2.putText(img, txt, (x+bw+10,y+bh-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

while True:
    root.update()
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose + Hands
    res_p = pose.process(rgb)
    center, scale = get_body_frame(res_p)
    res_h = hands.process(rgb)

    # Vector relativo al cuerpo
    feat = hands_to_body_relative(res_h, center, scale)

    # Dibujar
    if res_p and res_p.pose_landmarks:
        mp_draw.draw_landmarks(frame, res_p.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if res_h.multi_hand_landmarks:
        for hl in res_h.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # Lógica buffers
    if mode.get()=="static":
        if is_valid(feat, require_two=force_two.get()):
            buffer_static.append(feat)

    # FPS
    frames+=1; now=time.time()
    if now-last_fps_t>=1.0: fps=frames/(now-last_fps_t); frames=0; last_fps_t=now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,200,30), 2)

    # Pre-roll / Dinámica
    if prerec:
        rem = PRE_ROLL_SECS - (now - prerec_start)
        txtc = "¡YA!" if rem<=0 else f"{int(np.ceil(rem))}"
        cv2.putText(frame, txtc, (frame.shape[1]//2-20, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        draw_bar(frame, PRE_ROLL_SECS-rem, PRE_ROLL_SECS, (0,0,255), f"Comienza en {max(0.0, rem):.1f}s")
        if rem<=0:
            prerec=False; recording=True; rec_start=time.time(); dyn_frames=[]
    if recording:
        if is_valid(feat, require_two=force_two.get()):
            dyn_frames.append(feat)
        el = now - rec_start
        total = max(1,int(dyn_secs.get()))
        draw_bar(frame, el, total, (0,0,255), f"{el:.1f}/{total}s")
        if el>=total:
            recording=False
            if len(dyn_frames)<10:
                log(f"⚠ Grabación muy corta ({len(dyn_frames)}f).")
            else:
                dur = el; fps_est = len(dyn_frames)/max(dur,1e-6)
                save_seq(label_safe,label_display,dyn_frames,"dynamic",dur,fps_est)
                dyn_frames=[]

    # Barra de progreso estática
    if mode.get()=="static":
        draw_bar(frame, len(buffer_static), STATIC_FRAMES, (80,200,80),
                 f"{len(buffer_static)}/{STATIC_FRAMES}")

    cv2.imshow("Pose-Anchor Capture", frame)
    k = cv2.waitKey(1) & 0xFF
    if k==27: break
    if k==13:  # ENTER
        if not label_safe:
            log("⚠ Establece etiqueta primero."); continue
        if mode.get()=="static":
            if len(buffer_static)<STATIC_FRAMES:
                log("⚠ Pose estable insuficiente."); continue
            seq = list(buffer_static)[-STATIC_FRAMES:]
            dur = STATIC_FRAMES/max(fps,30.0)
            save_seq(label_safe,label_display,seq,"static",dur,fps)
            buffer_static.clear()
        else:
            if force_two.get() and (not is_valid(feat, require_two=True)):
                log("⚠ Se requieren 2 manos visibles."); continue
            prerec=True; prerec_start=time.time()
    # atajo de etiqueta por tecla alfanumérica
    if (65<=k<=90) or (97<=k<=122) or (48<=k<=57):
        entry_label.delete(0,"end"); entry_label.insert(0, chr(k).upper()); set_label()

cap.release(); cv2.destroyAllWindows()
try: root.destroy()
except: pass
