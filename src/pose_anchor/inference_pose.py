# -*- coding: utf-8 -*-
"""
inference_pose.py — Inferencia con anclas corporales (Pose) y manos relativas al cuerpo
- Misma FEAT_DIM=126. Carga scaler_stats.npz del modelo.
- Barra de top-3 y umbral. Stride x2 opcional para más FPS.
"""

import os, json, time, argparse, numpy as np, cv2, mediapipe as mp, tensorflow as tf
from collections import deque

# Pillow para Tk UI simple
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ---------- CLI ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=os.path.join(ROOT,"models_pose","lsp_pose_v1.keras"))
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

MODEL_PATH = os.path.abspath(args.model)
model = tf.keras.models.load_model(MODEL_PATH)
_, SEQ_LENGTH, FEAT_DIM = model.input_shape
print("Modelo:", MODEL_PATH, "T,F=", SEQ_LENGTH, FEAT_DIM)

# scaler
stats = np.load(os.path.join(os.path.dirname(MODEL_PATH), "scaler_stats.npz"))
MEAN, STD = stats["mean"].astype(np.float32), stats["std"].astype(np.float32); STD[STD<1e-8]=1.0

# labels
labs_path = os.path.join(os.path.dirname(MODEL_PATH),"labels.json")
LABELS = json.load(open(labs_path,"r",encoding="utf-8"))["labels"] if os.path.exists(labs_path) else [f"C{i}" for i in range(model.output_shape[-1])]

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.3, model_complexity=0)
pose  = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False, min_detection_confidence=0.5)

EMA_ALPHA=0.25
body_center=None; body_scale=None
def get_body_frame(res_p):
    global body_center, body_scale
    if not res_p or not res_p.pose_landmarks: return body_center, body_scale
    lm = res_p.pose_landmarks.landmark
    LSH,RSH,LHP,RHP = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                       mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                       mp_pose.PoseLandmark.LEFT_HIP.value,
                       mp_pose.PoseLandmark.RIGHT_HIP.value]
    def xy(i): p=lm[i]; return float(np.clip(p.x,0,1)), float(np.clip(p.y,0,1))
    lsh,rsh,lhp,rhp = xy(LSH), xy(RSH), xy(LHP), xy(RHP)
    cx = (lsh[0]+rsh[0]+lhp[0]+rhp[0])/4.0
    cy = (lsh[1]+rsh[1]+lhp[1]+rhp[1])/4.0
    dist_shoulders = np.hypot(lsh[0]-rsh[0], lsh[1]-rsh[1])
    dist_torso = (np.hypot(lsh[0]-lhp[0], lsh[1]-lhp[1]) + np.hypot(rsh[0]-rhp[0], rsh[1]-rhp[1]))/2.0
    scale = max(1e-3, 0.6*dist_shoulders+0.4*dist_torso)
    if body_center is None: body_center=(cx,cy)
    else: body_center=((1-EMA_ALPHA)*body_center[0]+EMA_ALPHA*cx, (1-EMA_ALPHA)*body_center[1]+EMA_ALPHA*cy)
    if body_scale is None: body_scale=scale
    else: body_scale=(1-EMA_ALPHA)*body_scale + EMA_ALPHA*scale
    return body_center, body_scale

def hands_to_body_relative(res_h, center, scale):
    left=[0.0]*63; right=[0.0]*63
    if res_h.multi_hand_landmarks and res_h.multi_handedness and center and scale:
        cx,cy = center; s=max(1e-3,float(scale))
        for hl,hd in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords=[]
            for p in hl.landmark:
                x=(float(np.clip(p.x,0,1))-cx)/s
                y=(float(np.clip(p.y,0,1))-cy)/s
                z=float(np.clip(p.z,-1,1))/s
                coords += [float(np.clip(x,-2,2)), float(np.clip(y,-2,2)), float(np.clip(z,-2,2))]
            if hd.classification[0].label=="Left": left=coords[:63]
            else: right=coords[:63]
    return np.array(left+right, np.float32)

def is_valid(vec126, require_two=False):
    l=float(np.sum(np.abs(vec126[:63]))); r=float(np.sum(np.abs(vec126[63:])))
    return (l>0 and r>0) if require_two else (l+r>0)

# ---------- UI simple ----------
root = tk.Tk(); root.title("Inferencia Pose Anchor")
panel = ttk.Frame(root, padding=8); panel.grid(row=0,column=0,sticky="nsew")
root.rowconfigure(0,weight=1); root.columnconfigure(0,weight=1)
video_label = ttk.Label(panel); video_label.grid(row=0,column=0,sticky="nsew")
panel.rowconfigure(0,weight=1); panel.columnconfigure(0,weight=1)

ctrl = ttk.Frame(panel); ctrl.grid(row=1,column=0,sticky="we", pady=(6,0))
running = {'flag': True}
def toggle(): running['flag']=not running['flag']; btn.config(text=("Reanudar" if not running['flag'] else "Pausar"))
btn = ttk.Button(ctrl, text="Pausar", command=toggle); btn.grid(row=0,column=0, padx=4)
thr_var = tk.DoubleVar(value=0.90); ttk.Label(ctrl,text="Umbral").grid(row=0,column=1); ttk.Scale(ctrl,from_=0,to=100,orient="horizontal", length=160, command=lambda v: None).grid(row=0,column=2)
smooth_var = tk.IntVar(value=5)

cap = cv2.VideoCapture(args.camera); cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height); cap.set(cv2.CAP_PROP_FPS,30)

buffer = deque(maxlen=SEQ_LENGTH); hist = deque(maxlen=8)
frames=0; t0=time.time(); fps=0.0

# --- added: utility to make frame square (reduce MediaPipe NORM_RECT warning) ---
def make_square_frame(frame):
    h, w = frame.shape[:2]
    if h == w:
        return frame
    s = max(h, w)
    pad_top = (s - h) // 2
    pad_bottom = s - h - pad_top
    pad_left = (s - w) // 2
    pad_right = s - w - pad_left
    return cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right,
                              borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

# --- added: verify camera opened ---
if not cap.isOpened():
    from tkinter import messagebox
    messagebox.showerror("Cámara", f"No se pudo abrir la cámara {args.camera}. Verifica conexión y permisos.")
    try: cap.release()
    except: pass
    raise SystemExit("Camera not available")

def loop():
    global frames, t0, fps
    try:
        ok, frame = cap.read()
        if not ok:
            # mostrar error y detener loop si la cámara deja de responder
            try: messagebox.showerror("Cámara", "No se recibió frame de la cámara. Cerrando.")
            except: pass
            on_close()
            return

        # pad a cuadrado para evitar advertencia de MediaPipe cuando ROI no es cuadrado
        frame = make_square_frame(frame)
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res_p = pose.process(rgb)
        center, scale = get_body_frame(res_p)
        res_h = hands.process(rgb)

        if running['flag']:
            vec = hands_to_body_relative(res_h, center, scale)
            buffer.append(vec)

        # draw
        if res_h.multi_hand_landmarks:
            for hl in res_h.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        if res_p and res_p.pose_landmarks:
            mp_draw.draw_landmarks(frame, res_p.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if len(buffer)==SEQ_LENGTH:
            seq = np.array(buffer, np.float32)
            seq = (seq - MEAN[None,:])/STD[None,:]
            x = seq[None,...].astype(np.float32)  # asegurar dtype correcto

            # Invocar el modelo de forma consistente para evitar warnings/errores.
            # Primero intentar predict (API pública); si falla, llamar usando input name.
            try:
                preds = model.predict(x, verbose=0)
                probs = preds[0]
            except Exception:
                try:
                    probs = model({model.input_names[0]: x}, training=False).numpy()[0]
                except Exception as e:
                    # mostrar error y parar inferencia si el modelo falla
                    try: messagebox.showerror("Modelo", f"Error al ejecutar el modelo: {e}")
                    except: pass
                    on_close()
                    return

            hist.append(probs)
            k = max(1, int(smooth_var.get()))
            sm = np.mean(list(hist)[-k:], axis=0)
            top = sm.argsort()[-min(3,len(LABELS)):][::-1]
            for j,i in enumerate(top):
                cv2.putText(frame, f"{j+1}. {LABELS[int(i)]}  {sm[int(i)]*100:.1f}%", (10,30+24*j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)

        frames += 1; now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0; t0 = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,200,30),2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

        root.after(1, loop)

    except tk.TclError:
        # la ventana pudo haber sido cerrada por el sistema; asegurar limpieza
        on_close()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try: messagebox.showerror("Error inesperado", f"{e}\nRevisa la consola para más detalles.")
        except: pass
        print("ERROR en loop():", e)
        print(tb)
        on_close()

def on_close():
    try: cap.release()
    except: pass
    try: root.destroy()
    except: pass

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(1, loop)
root.mainloop()
