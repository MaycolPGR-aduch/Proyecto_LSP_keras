import os
import cv2
import json
import time
import numpy as np
import unicodedata
from datetime import datetime
from collections import deque
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox

# ========= CONFIG =========
BASE_DIR   = os.path.join(os.getcwd(), "..", "coordenadas")
INDEX_CSV  = os.path.join(BASE_DIR, "_index.csv")

FEAT_HANDS = 21*3*2        # 126
FEAT_FACE  = 468*3         # 1404
FEAT_DIM   = FEAT_HANDS + FEAT_FACE  # 1530

STATIC_FRAMES   = 30
PRE_ROLL_SECS   = 3         # cuenta atrás antes de grabar dinámica
SAVE_SAMPLE_CSV = False     # desactiva CSV gigante por muestra

SHOW_HANDS = True
SHOW_FACE  = True

# ========= ESTADO =========
current_label_display = None
current_label_safe    = None
sample_count = {}
static_buffer = deque(maxlen=STATIC_FRAMES)

root = tk.Tk()  # Inicializa la ventana principal de tkinter
mode     = tk.StringVar(master=root, value="static")  # "static" | "dynamic"
dyn_secs = tk.IntVar(master=root, value=4)

recording_on   = False         # grabación dinámica activa
pre_roll_on    = False         # fase de cuenta atrás
recording_type = None          # "static"|"dynamic"
rec_start_time = 0.0
pre_roll_start = 0.0
dyn_frames     = []            # frames capturados en dinámica

# ========= MEDIAPIPE =========
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
face  = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)

# ========= UTILS =========
def sanitize_label(text: str) -> str:
    if not text: return ""
    x = text.upper().strip()
    x = ''.join(c for c in unicodedata.normalize('NFKD', x) if not unicodedata.combining(c))
    x = ''.join(c if (c.isalnum() or c in (' ', '_', '-')) else '_' for c in x)
    x = x.replace(' ', '_')
    while '__' in x: x = x.replace('__', '_')
    return x

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ensure_label_dir(label_safe: str) -> str:
    p = os.path.join(BASE_DIR, label_safe)
    ensure_dir(p)
    return p

def get_existing_count(label_safe: str) -> int:
    folder = ensure_label_dir(label_safe)
    return sum(1 for f in os.listdir(folder)
               if f.lower().endswith(".npy") and f.startswith(label_safe + "_"))

def append_index_row(ts: str, display: str, safe: str, path: str):
    new_file = not os.path.exists(INDEX_CSV)
    ensure_dir(BASE_DIR)
    with open(INDEX_CSV, "a", encoding="utf-8", newline="") as f:
        if new_file:
            f.write("timestamp,label_display,label_safe,filepath\n")
        f.write(f"{ts},{display},{safe},{path}\n")

def log(msg: str):
    text_log.configure(state="normal")
    text_log.insert("end", msg + "\n")
    text_log.see("end")
    text_log.configure(state="disabled")

def update_status_labels():
    lbl_current.configure(text=f"Etiqueta actual: {current_label_display or '(ninguna)'}  |  Carpeta: {current_label_safe or '-'}")
    cnt = sample_count.get(current_label_safe, 0) if current_label_safe else 0
    lbl_count.configure(text=f"Muestras guardadas: {cnt}")
    # habilita/deshabilita segundos según modo
    spin_secs.configure(state=("normal" if mode.get()=="dynamic" else "disabled"))

def set_label_from_gui():
    global current_label_display, current_label_safe
    lab = entry_label.get().strip()
    if not lab:
        messagebox.showwarning("Etiqueta vacía", "Ingresa un carácter o frase para la seña.")
        return
    current_label_display = lab
    current_label_safe = sanitize_label(lab)
    ensure_label_dir(current_label_safe)
    sample_count[current_label_safe] = get_existing_count(current_label_safe)
    static_buffer.clear()
    log(f"Etiqueta -> '{current_label_display}' (safe='{current_label_safe}'). Muestras existentes: {sample_count[current_label_safe]}")
    update_status_labels()

def save_sample(label_safe: str, label_display: str, seq, capture_mode: str, duration_sec: float, fps_est: float):
    folder = ensure_label_dir(label_safe)
    idx = sample_count[label_safe] + 1
    npy_path = os.path.join(folder, f"{label_safe}_{idx:03d}.npy")
    json_path = npy_path.replace(".npy", ".json")
    csv_path  = npy_path.replace(".npy", "_coords.csv")

    arr = np.array(seq, dtype=np.float32)  # (L, FEAT_DIM)
    np.save(npy_path, arr)
    sample_count[label_safe] = idx

    if SAVE_SAMPLE_CSV:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join([f"f{i}" for i in range(arr.shape[1])]) + "\n")
            for row in arr:
                f.write(",".join(str(v) for v in row) + "\n")

    ts = datetime.now().isoformat(timespec="seconds")
    meta = {
        "timestamp": ts,
        "label_display": label_display,
        "label_safe": label_safe,
        "frames": int(arr.shape[0]),
        "features": FEAT_DIM,
        "include_face": True,
        "capture_mode": capture_mode,  # "static" o "dynamic"
        "duration_secs": float(duration_sec),
        "fps_est": float(fps_est),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    append_index_row(ts, label_display, label_safe, npy_path)
    log(f"Guardado: {npy_path} | frames={arr.shape[0]} feats={FEAT_DIM} | total '{label_display}': {idx}")

# ========= FEATURES =========
def assemble_frame_vector(res_h, res_f):
    # Manos Left->Right (63 cada una)
    left  = [0.0]*63
    right = [0.0]*63
    if res_h.multi_hand_landmarks and res_h.multi_handedness:
        for hand_lms, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords = []
            for p in hand_lms.landmark:
                coords.extend([p.x, p.y, p.z])
            label = handed.classification[0].label
            if label == "Left":
                left = coords[:63]
            else:
                right = coords[:63]
    hands_vec = left + right  # 126

    # Rostro 468*3 (si no hay, ceros)
    face_vec = [0.0]*FEAT_FACE
    if res_f.multi_face_landmarks:
        lm = res_f.multi_face_landmarks[0].landmark
        coords = []
        for p in lm:
            coords.extend([p.x, p.y, p.z])
        if len(coords) >= FEAT_FACE:
            face_vec = coords[:FEAT_FACE]
        else:
            face_vec = coords + [0.0]*(FEAT_FACE - len(coords))

    return hands_vec + face_vec  # 1530

# ========= DINÁMICA: PRE-ROLL + GRABACIÓN =========
def start_dynamic_recording_with_preroll():
    global pre_roll_on, recording_on, recording_type, pre_roll_start, dyn_frames
    if not current_label_safe:
        messagebox.showinfo("Falta etiqueta", "Primero establece una etiqueta o frase.")
        return
    secs = max(1, int(dyn_secs.get()))
    dyn_secs.set(secs)
    dyn_frames = []
    pre_roll_on    = True
    recording_on   = False
    recording_type = "dynamic"
    pre_roll_start = time.time()
    log(f"[Dinámica] Preparado: '{current_label_display}' por {secs}s. Pre-roll {PRE_ROLL_SECS}s…")

# ========= TK GUI =========
root = tk.Tk()
root.title("Captura LSP - GUI (manos + rostro)")

frm = ttk.Frame(root, padding=8)
frm.grid(row=0, column=0, sticky="nsew")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Etiqueta/Frase
ttk.Label(frm, text="Etiqueta o frase:").grid(row=0, column=0, sticky="w")
entry_label = ttk.Entry(frm, width=30)
entry_label.grid(row=0, column=1, sticky="w")
btn_set = ttk.Button(frm, text="Establecer", command=set_label_from_gui)
btn_set.grid(row=0, column=2, padx=6, sticky="w")

# Modo
def on_mode_change():
    update_status_labels()
ttk.Label(frm, text="Tipo de seña:").grid(row=1, column=0, sticky="w", pady=(6,0))
rb_static  = ttk.Radiobutton(frm, text="Estática (30 frames, Enter para guardar)",
                             variable=mode, value="static", command=on_mode_change)
rb_dynamic = ttk.Radiobutton(frm, text="Dinámica (N segundos)",
                             variable=mode, value="dynamic", command=on_mode_change)
rb_static.grid(row=1, column=1, sticky="w", pady=(6,0))
rb_dynamic.grid(row=1, column=2, sticky="w", pady=(6,0))

# Segundos dinámicas
ttk.Label(frm, text="Segundos (dinámicas):").grid(row=2, column=0, sticky="w")
spin_secs = ttk.Spinbox(frm, from_=1, to=10, textvariable=dyn_secs, width=5)
spin_secs.grid(row=2, column=1, sticky="w")

btn_start_dynamic = ttk.Button(frm, text="Iniciar DINÁMICA", command=start_dynamic_recording_with_preroll)
btn_start_dynamic.grid(row=2, column=2, padx=6, sticky="w")

# Estado
lbl_current = ttk.Label(frm, text="Etiqueta actual: (ninguna)")
lbl_current.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8,0))
lbl_count = ttk.Label(frm, text="Muestras guardadas: 0")
lbl_count.grid(row=4, column=0, columnspan=3, sticky="w")

# Log
text_log = tk.Text(frm, height=8, width=70, state="disabled")
text_log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(8,0))
frm.rowconfigure(5, weight=1)

update_status_labels()  # Actualiza el estado del Spinbox según el modo seleccionado
mode.trace_add('write', lambda *args: update_status_labels())  # Actualiza automáticamente cuando cambia el modo

# ========= LOOP VIDEO =========
cap = cv2.VideoCapture(0)
cv2.namedWindow("Vista LSP", cv2.WINDOW_NORMAL)

last_fps_time = time.time()
frames_seen = 0
fps_value = 0.0

def draw_overlay(img, msg): cv2.putText(img, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
def draw_big_countdown(img, number):
    h, w = img.shape[:2]
    cv2.putText(img, str(number), (w//2-30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 6, cv2.LINE_AA)
def draw_bar(img, cur, total, color=(0,200,0), label_text=""):
    h, w = img.shape[:2]
    bar_w = int(w * 0.6); bar_h = 14
    x0 = int((w-bar_w)/2); y0 = h - 30
    cv2.rectangle(img, (x0, y0), (x0+bar_w, y0+bar_h), (60,60,60), -1)
    fill = int(bar_w * min(cur,total) / max(total,1))
    cv2.rectangle(img, (x0, y0), (x0+fill, y0+bar_h), color, -1)
    if label_text:
        cv2.putText(img, label_text, (x0+bar_w+10, y0+bar_h-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

while True:
    root.update()
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe
    res_h = hands.process(rgb)
    res_f = face.process(rgb)

    # Dibujo
    if res_h.multi_hand_landmarks and SHOW_HANDS:
        for hl in res_h.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
    if res_f.multi_face_landmarks and SHOW_FACE:
        for fl in res_f.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_TESSELATION)

    # Vector por frame
    feat_vec = assemble_frame_vector(res_h, res_f)

    # Actualiza buffers según modo
    if mode.get() == "static":
        static_buffer.append(feat_vec)

    # FPS
    frames_seen += 1
    now = time.time()
    if now - last_fps_time >= 1.0:
        fps_value = frames_seen / (now - last_fps_time)
        frames_seen = 0
        last_fps_time = now

    # Overlays
    label_txt = f"Etiqueta: {current_label_display or '(ninguna)'} | Carpeta: {current_label_safe or '-'} | FPS: {fps_value:.1f}"
    draw_overlay(frame, label_txt)

    # PRE-ROLL (cuenta atrás grande)
    if pre_roll_on:
        elapsed = now - pre_roll_start
        remaining = PRE_ROLL_SECS - elapsed
        num = max(0, int(np.ceil(remaining)))
        draw_big_countdown(frame, num if num>0 else "GO")
        draw_bar(frame, int((elapsed/PRE_ROLL_SECS)*100), 100, color=(0,0,255),
                 label_text=f"Comienza en {max(0.0, remaining):.1f}s")
        if remaining <= 0:
            # iniciar grabación dinámica
            pre_roll_on  = False
            recording_on = True
            recording_type = "dynamic"
            dyn_frames = []
            rec_start_time = time.time()

    # Grabación dinámica (con cuenta atrás de tiempo restante)
    if recording_on and recording_type == "dynamic":
        dyn_frames.append(feat_vec)
        total = max(1, int(dyn_secs.get()))
        elapsed = now - rec_start_time
        remaining = max(0.0, total - elapsed)
        pct = int(min(1.0, elapsed/total) * 100)
        draw_bar(frame, pct, 100, color=(0,0,255),
                 label_text=f"{elapsed:.1f}/{total:.0f}s (resta {remaining:.1f}s)")
        cv2.putText(frame, "REC", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if elapsed >= total:
            # guardar dinámica
            recording_on = False
            seq = list(dyn_frames)
            dyn_frames = []
            duration = elapsed
            fps_est = len(seq) / max(duration, 1e-6)
            save_sample(current_label_safe, current_label_display, seq,
                        capture_mode="dynamic", duration_sec=duration, fps_est=fps_est)

    # Estática: progreso de 30 frames
    if mode.get() == "static":
        draw_bar(frame, len(static_buffer), STATIC_FRAMES, color=(80,200,80),
                 label_text=f"{len(static_buffer)}/{STATIC_FRAMES}")

    cv2.imshow("Vista LSP", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    # ENTER: acciona según modo
    if key == 13:
        if not current_label_safe:
            log("Primero establece etiqueta/frase.")
            continue
        if mode.get() == "static":
            if len(static_buffer) < STATIC_FRAMES:
                log(f"Buffer corto: {len(static_buffer)}/{STATIC_FRAMES}. Mantén la pose un instante.")
            else:
                seq = list(static_buffer)[-STATIC_FRAMES:]
                save_sample(current_label_safe, current_label_display, seq,
                            capture_mode="static", duration_sec=len(seq)/max(fps_value,30.0),
                            fps_est=fps_value)
                static_buffer.clear()
                update_status_labels()
        else:
            # dinámica: ENTER dispara pre-roll + grabación N s
            start_dynamic_recording_with_preroll()

    # Letras/números como atajo para establecer etiqueta
    if (65 <= key <= 90) or (97 <= key <= 122) or (48 <= key <= 57):
        entry_label.delete(0, "end")
        entry_label.insert(0, chr(key).upper())
        set_label_from_gui()

cap.release()
cv2.destroyAllWindows()
try:
    root.destroy()
except:
    pass
