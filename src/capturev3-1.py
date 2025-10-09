"""
capturev3-1.py — Captura LSP SOLO MANOS (sin rostro) con resolución FIJA
- Estáticas (buffer de 30 frames)
- Dinámicas (N segundos + pre-roll)
- Guarda .npy (L x 126) + .json de metadatos
- Resolución fija por defecto: 1280x720 (HD). Si no se puede, cae una vez a 640x480.
- Mantiene: ventana redimensionable, pantalla completa (F), zoom de vista (G).
Requisitos: OpenCV, MediaPipe, Tkinter, NumPy
"""

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
import platform

# ========= CONFIG =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "..", "coordenadas")
INDEX_CSV  = os.path.join(BASE_DIR, "_index.csv")

FEAT_HANDS = 21 * 3 * 2   # 126 (solo manos)
FEAT_DIM   = FEAT_HANDS

STATIC_FRAMES   = 30
PRE_ROLL_SECS   = 3
SAVE_SAMPLE_CSV = False
SHOW_HANDS = True

# Cámara (permite cambiar por variables de entorno)
CAMERA_INDEX   = int(os.environ.get("LSP_CAMERA_INDEX", "0"))
TARGET_WIDTH   = int(os.environ.get("LSP_TARGET_WIDTH",  "1280"))  # fija por defecto a 1280x720
TARGET_HEIGHT  = int(os.environ.get("LSP_TARGET_HEIGHT", "720"))
FALLBACK_WIDTH = 640   # único fallback por robustez
FALLBACK_HEIGHT= 480

# ========= ESTADO =========
root = tk.Tk()
root.title("Captura LSP - GUI (solo manos, resolución fija)")

current_label_display = None
current_label_safe    = None
sample_count = {}
static_buffer = deque(maxlen=STATIC_FRAMES)

mode     = tk.StringVar(master=root, value="static")  # "static" | "dynamic"
dyn_secs = tk.IntVar(master=root, value=4)
force_two_hands = tk.BooleanVar(master=root, value=True)

recording_on   = False
pre_roll_on    = False
recording_type = None
rec_start_time = 0.0
pre_roll_start = 0.0
dyn_frames     = []

# Tracking manos
hands_detected = 0
left_hand_present = False
right_hand_present = False

# ========= MEDIAPIPE =========
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_fast_cfg = dict(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
    model_complexity=1
)
hands_reacq_cfg = dict(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.4,
    model_complexity=1
)

# ========= UTILS =========
def sanitize_label(text: str) -> str:
    if not text:
        return ""
    x = text.upper().strip()
    x = ''.join(c for c in unicodedata.normalize('NFKD', x) if not unicodedata.combining(c))
    x = ''.join(c if (c.isalnum() or c in (' ', '_', '-')) else '_' for c in x)
    x = x.replace(' ', '_')
    while '__' in x:
        x = x.replace('__', '_')
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
    spin_secs.configure(state=("normal" if mode.get() == "dynamic" else "disabled"))

    hand_status = f"Manos: {hands_detected}/2"
    if hands_detected == 2:
        hand_status += " ✓"
        lbl_hands_status.configure(text=hand_status, foreground="green")
    elif hands_detected == 1:
        hand_status += " (falta una)"
        lbl_hands_status.configure(text=hand_status, foreground="orange")
    else:
        hand_status += " ✗"
        lbl_hands_status.configure(text=hand_status, foreground="red")

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
        "include_face": False,
        "capture_mode": capture_mode,
        "duration_secs": float(duration_sec),
        "fps_est": float(fps_est),
        "force_two_hands": bool(force_two_hands.get()),
        "feature_type": "hands_only"
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    append_index_row(ts, label_display, label_safe, npy_path)
    log(f"✓ Guardado: {npy_path} | frames={arr.shape[0]} feats={FEAT_DIM} | total '{label_display}': {idx}")

# ========= FEATURES (SOLO MANOS) =========
def assemble_frame_vector(res_h):
    global hands_detected, left_hand_present, right_hand_present

    left  = [0.0] * 63
    right = [0.0] * 63

    hands_detected = 0
    left_hand_present = False
    right_hand_present = False

    if res_h.multi_hand_landmarks and res_h.multi_handedness:
        for hand_lms, handed in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords = []
            for p in hand_lms.landmark:
                coords.extend([
                    float(np.clip(p.x, 0.0, 1.0)),
                    float(np.clip(p.y, 0.0, 1.0)),
                    float(np.clip(p.z, -1.0, 1.0)),
                ])
            label = handed.classification[0].label  # "Left" | "Right"
            if label == "Left":
                left = coords[:63]
                left_hand_present = True
                hands_detected += 1
            else:
                right = coords[:63]
                right_hand_present = True
                hands_detected += 1

    return left + right  # 126

# ========= VALIDACIÓN =========
def is_frame_valid(feat_vec, require_two_hands=False):
    if require_two_hands:
        left_sum = sum(abs(v) for v in feat_vec[:63])
        right_sum = sum(abs(v) for v in feat_vec[63:126])
        return left_sum > 0 and right_sum > 0
    return sum(abs(v) for v in feat_vec) > 0

# ========= DINÁMICA =========
def start_dynamic_recording_with_preroll():
    global pre_roll_on, recording_on, recording_type, pre_roll_start, dyn_frames
    if not current_label_safe:
        messagebox.showinfo("Falta etiqueta", "Primero establece una etiqueta o frase.")
        return
    if force_two_hands.get() and hands_detected < 2:
        messagebox.showwarning("Manos insuficientes", "Se requieren 2 manos visibles. Ajusta tu posición.")
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
frm = ttk.Frame(root, padding=8)
frm.grid(row=0, column=0, sticky="nsew")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(frm, text="Etiqueta o frase:").grid(row=0, column=0, sticky="w")
entry_label = ttk.Entry(frm, width=30)
entry_label.grid(row=0, column=1, sticky="w")
btn_set = ttk.Button(frm, text="Establecer", command=set_label_from_gui)
btn_set.grid(row=0, column=2, padx=6, sticky="w")

def on_mode_change(): update_status_labels()
ttk.Label(frm, text="Tipo de seña:").grid(row=1, column=0, sticky="w", pady=(6, 0))
rb_static  = ttk.Radiobutton(frm, text="Estática (30 frames, Enter para guardar)",
                             variable=mode, value="static", command=on_mode_change)
rb_dynamic = ttk.Radiobutton(frm, text="Dinámica (N segundos)",
                             variable=mode, value="dynamic", command=on_mode_change)
rb_static.grid(row=1, column=1, sticky="w", pady=(6, 0))
rb_dynamic.grid(row=1, column=2, sticky="w", pady=(6, 0))

ttk.Label(frm, text="Segundos (dinámicas):").grid(row=2, column=0, sticky="w")
spin_secs = ttk.Spinbox(frm, from_=1, to=10, textvariable=dyn_secs, width=5)
spin_secs.grid(row=2, column=1, sticky="w")
btn_start_dynamic = ttk.Button(frm, text="Iniciar DINÁMICA", command=start_dynamic_recording_with_preroll)
btn_start_dynamic.grid(row=2, column=2, padx=6, sticky="w")

chk_two = ttk.Checkbutton(frm, text="Requiere 2 manos (recomendado para LSP)", variable=force_two_hands)
chk_two.grid(row=2, column=3, padx=10, sticky="w")

lbl_current = ttk.Label(frm, text="Etiqueta actual: (ninguna)")
lbl_current.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
lbl_count = ttk.Label(frm, text="Muestras guardadas: 0")
lbl_count.grid(row=4, column=0, columnspan=4, sticky="w")
lbl_hands_status = ttk.Label(frm, text="Manos: 0/2", foreground="red")
lbl_hands_status.grid(row=5, column=0, columnspan=4, sticky="w")

text_log = tk.Text(frm, height=8, width=80, state="disabled")
text_log.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
frm.rowconfigure(6, weight=1)

instructions = (
    "INSTRUCCIONES:\n"
    "• Enter: Capturar estática (o iniciar dinámica con pre-roll)\n"
    "• F: Pantalla completa ON/OFF\n"
    "• G: Zoom de vista ON/OFF (no afecta la captura)\n"
    "• Letras/Números: Atajo rápido para establecer etiqueta\n"
    "• ESC: Salir\n"
    "• Para LSP se recomienda tener ambas manos visibles\n"
    f"• Resolución fija solicitada: {TARGET_WIDTH}x{TARGET_HEIGHT} (fallback automático a {FALLBACK_WIDTH}x{FALLBACK_HEIGHT} si no está disponible)"
)
lbl_instructions = ttk.Label(frm, text=instructions, justify="left", foreground="gray")
lbl_instructions.grid(row=7, column=0, columnspan=4, sticky="w", pady=(4, 0))

update_status_labels()
mode.trace_add('write', lambda *args: update_status_labels())

# ========= CÁMARA / VENTANA (resolución FIJA) =========
WINDOW_NAME = "Vista LSP - Captura de Manos (resolución fija)"
fullscreen_on = False
display_scale = 1.0

def open_camera_fixed(index=0, width=1280, height=720, fps=30):
    """Intenta abrir a (width,height). Si falla, cae a FALLBACK_WIDTH x FALLBACK_HEIGHT."""
    def _try_open(w, h, label):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(0.05)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None, (0,0)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if abs(aw - w) <= 16 and abs(ah - h) <= 16:
            log(f"Cámara abierta @{aw}x{ah} ~{fps}FPS ({label})")
            return cap, (aw, ah)
        # aceptamos lo que dio el driver, pero avisamos
        log(f"[AVISO] Se pidió {w}x{h} ({label}), pero la cámara devolvió {aw}x{ah}.")
        return cap, (aw, ah)

    cap, size = _try_open(width, height, "TARGET")
    if cap is not None and size != (0,0):
        return cap, size

    log(f"[AVISO] No se pudo usar {width}x{height}. Probando fallback {FALLBACK_WIDTH}x{FALLBACK_HEIGHT}...")
    cap, size = _try_open(FALLBACK_WIDTH, FALLBACK_HEIGHT, "FALLBACK")
    if cap is not None and size != (0,0):
        return cap, size
    raise RuntimeError("La cámara no devolvió frames ni en el modo fallback.")

def apply_window_defaults(w, h):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    init_w = int(min(w, 1600))
    init_h = int(min(h, 900))
    try:
        cv2.resizeWindow(WINDOW_NAME, init_w, init_h)
    except:
        pass

# ========= LOOP VIDEO =========
cap, (cap_w, cap_h) = open_camera_fixed(index=CAMERA_INDEX, width=TARGET_WIDTH, height=TARGET_HEIGHT, fps=30)
apply_window_defaults(cap_w, cap_h)

last_fps_time = time.time()
frames_seen = 0
fps_value = 30.0

def draw_overlay(img, msg):
    cv2.putText(img, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def draw_big_countdown(img, number):
    h, w = img.shape[:2]
    text = str(number) if isinstance(number, int) else number
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
    x = (w - size[0]) // 2
    y = (h + size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)

def draw_bar(img, cur_pct, total_pct=100, color=(0, 200, 0), label_text=""):
    h, w = img.shape[:2]
    bar_w = int(w * 0.6)
    bar_h = 14
    x0 = int((w - bar_w) / 2)
    y0 = h - 30
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * min(cur_pct, total_pct) / max(total_pct, 1))
    cv2.rectangle(img, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
    if label_text:
        cv2.putText(img, label_text, (x0 + bar_w + 10, y0 + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

def draw_hand_status(img):
    h, w = img.shape[:2]
    status_y = 60
    color_left = (0, 255, 0) if left_hand_present else (0, 0, 200)
    cv2.putText(img, "IZQ", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_left, 2)
    cv2.circle(img, (55, status_y - 5), 8, color_left, -1 if left_hand_present else 2)
    color_right = (0, 255, 0) if right_hand_present else (0, 0, 200)
    cv2.putText(img, "DER", (90, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_right, 2)
    cv2.circle(img, (135, status_y - 5), 8, color_right, -1 if right_hand_present else 2)

with mp_hands.Hands(**hands_fast_cfg) as hands_fast, mp_hands.Hands(**hands_reacq_cfg) as hands_reacq:
    try:
        while True:
            root.update()
            ok, frame = cap.read()
            if not ok:
                log("⚠ No se pudo leer de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res_h = hands_fast.process(rgb)
            if force_two_hands.get():
                count = len(res_h.multi_hand_landmarks) if res_h.multi_hand_landmarks else 0
                if count < 2:
                    res_h = hands_reacq.process(rgb)

            feat_vec = assemble_frame_vector(res_h)

            if res_h.multi_hand_landmarks and res_h.multi_handedness and SHOW_HANDS:
                for hand_landmarks, handedness in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
                    hand_label = handedness.classification[0].label
                    if hand_label == "Left":
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)
                        )
                    else:
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
                        )

            if mode.get() == "static":
                if is_frame_valid(feat_vec, require_two_hands=force_two_hands.get()):
                    static_buffer.append(feat_vec)

            frames_seen += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps_value = frames_seen / (now - last_fps_time)
                frames_seen = 0
                last_fps_time = now

            label_txt = f"Etiqueta: {current_label_display or '(ninguna)'} | FPS: {fps_value:.1f} | {cap_w}x{cap_h} (fija)"
            draw_overlay(frame, label_txt)
            draw_hand_status(frame)

            if force_two_hands.get() and hands_detected < 2 and not recording_on and not pre_roll_on:
                cv2.putText(frame, "¡Se requieren 2 manos visibles!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if pre_roll_on:
                elapsed = now - pre_roll_start
                remaining = PRE_ROLL_SECS - elapsed
                num = max(0, int(np.ceil(remaining)))
                draw_big_countdown(frame, num if num > 0 else "¡YA!")
                draw_bar(frame, int((elapsed / PRE_ROLL_SECS) * 100), 100, color=(0, 0, 255),
                         label_text=f"Comienza en {max(0.0, remaining):.1f}s")
                if remaining <= 0:
                    pre_roll_on  = False
                    recording_on = True
                    recording_type = "dynamic"
                    dyn_frames = []
                    rec_start_time = time.time()

            if recording_on and recording_type == "dynamic":
                if is_frame_valid(feat_vec, require_two_hands=force_two_hands.get()):
                    dyn_frames.append(feat_vec)

                total = max(1, int(dyn_secs.get()))
                elapsed = now - rec_start_time
                remaining = max(0.0, total - elapsed)
                pct = int(min(1.0, elapsed / total) * 100)
                draw_bar(frame, pct, 100, color=(0, 0, 255),
                         label_text=f"{elapsed:.1f}/{total:.0f}s (resta {remaining:.1f}s)")
                cv2.putText(frame, "● REC", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Frames válidos: {len(dyn_frames)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if elapsed >= total:
                    recording_on = False
                    if len(dyn_frames) < 10:
                        log(f"⚠ Grabación muy corta ({len(dyn_frames)} frames). Intenta de nuevo.")
                    else:
                        seq = list(dyn_frames)
                        dyn_frames = []
                        duration = elapsed
                        fps_est = len(seq) / max(duration, 1e-6)
                        save_sample(current_label_safe, current_label_display, seq,
                                    capture_mode="dynamic", duration_sec=duration, fps_est=fps_est)

            if mode.get() == "static":
                valid_frames = len(static_buffer)
                draw_bar(frame, valid_frames, STATIC_FRAMES, color=(80, 200, 80),
                         label_text=f"{valid_frames}/{STATIC_FRAMES}")

            # Mostrar (con escala opcional sin afectar la captura)
            disp = frame
            if display_scale != 1.0:
                nh = int(frame.shape[0] * display_scale)
                nw = int(frame.shape[1] * display_scale)
                disp = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if key == 13:  # ENTER
                if not current_label_safe:
                    log("⚠ Primero establece etiqueta/frase.")
                    continue
                if mode.get() == "static":
                    if len(static_buffer) < STATIC_FRAMES:
                        log(f"⚠ Buffer incompleto: {len(static_buffer)}/{STATIC_FRAMES}. Mantén la pose.")
                    else:
                        valid_count = sum(1 for f in static_buffer
                                          if is_frame_valid(f, require_two_hands=force_two_hands.get()))
                        if valid_count < STATIC_FRAMES * 0.8:
                            log(f"⚠ Calidad insuficiente: {valid_count}/{STATIC_FRAMES} frames válidos.")
                        else:
                            seq = list(static_buffer)[-STATIC_FRAMES:]
                            duration = len(seq) / max(fps_value, 1e-6)
                            save_sample(current_label_safe, current_label_display, seq,
                                        capture_mode="static", duration_sec=duration, fps_est=fps_value)
                            static_buffer.clear()
                else:
                    start_dynamic_recording_with_preroll()

            # Atajo para etiqueta (A-Z, 0-9)
            if (65 <= key <= 90) or (97 <= key <= 122) or (48 <= key <= 57):
                entry_label.delete(0, "end")
                entry_label.insert(0, chr(key).upper())
                set_label_from_gui()

            # Pantalla completa
            if key in (ord('f'), ord('F')):
                fullscreen_on = not fullscreen_on
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen_on else cv2.WINDOW_NORMAL
                )

            # Zoom de vista (no afecta la captura)
            elif key in (ord('g'), ord('G')):
                display_scale = 1.0 if display_scale != 1.5 else 1.5

    finally:
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        try: root.destroy()
        except: pass

print("Sistema cerrado correctamente.")