# -*- coding: utf-8 -*-
"""
inference_gui_pro2.py — UI tipo dashboard con toolbar FIJO + área scrollable.
- Toolbar (siempre visible): Iniciar, Pausar/Reanudar, Reset, Cámara (índice, resolución, Aplicar).
- Contenido scrollable: Vista de cámara y Palabras traducidas.
- Límite de altura del video (MAX_VIDEO_H=480) + resize para no empujar la UI.
- Resto del pipeline intacto (manos-only 126, scaler, stridex2, TTS top1≥90%, stats).
"""

import os, json, time, argparse, threading, queue, tempfile
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Pillow
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

# CPU stats (opcional)
try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False

# ====== CLI ======
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=os.path.join(ROOT, "lsp_sequence_model_frasescomunes.keras"))
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--data-dir", type=str, default=os.path.join(ROOT, "coordenadas"))
args = parser.parse_args()

MODEL_PATH = os.path.abspath(args.model)
DATA_DIR   = os.path.abspath(args.data_dir)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

# ====== Modelo / labels / scaler ======
print("Cargando modelo:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
_, SEQ_LENGTH, FEAT_DIM = model.input_shape
NUM_CLASSES = model.output_shape[-1]
print(f"Modelo -> T={SEQ_LENGTH}, F={FEAT_DIM}, clases={NUM_CLASSES}")

FEAT_HANDS = 21*3*2
FEAT_FACE  = 468*3
def feature_mode_from_dim(d:int): return "hands_face" if d >= (FEAT_HANDS + 1000) else "hands"
FEATURE_MODE = feature_mode_from_dim(FEAT_DIM)

def load_labels(model_path: str, data_dir: str, n: int):
    for p in [os.path.join(os.path.dirname(model_path),"labels.json"),
              os.path.join(data_dir,"labels.json")]:
        if os.path.isfile(p):
            try:
                js = json.load(open(p,"r",encoding="utf-8"))
                if isinstance(js, dict) and "labels" in js and len(js["labels"])==n: return js["labels"]
                if isinstance(js, dict):
                    rev = sorted(js.items(), key=lambda kv: kv[1]); labs = [k for k,_ in rev]
                    if len(labs)==n: return labs
            except: pass
    if os.path.isdir(DATA_DIR):
        labs = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR,d)) and not d.startswith("_")])
        if len(labs)==n: return labs
    return [f"CLASE_{i}" for i in range(n)]
LABELS = load_labels(MODEL_PATH, DATA_DIR, NUM_CLASSES)

def load_scaler(model_path: str, data_dir: str, feat_dim: int):
    for p in [os.path.join(os.path.dirname(model_path),"scaler_stats.npz"),
              os.path.join(data_dir,"scaler_stats.npz")]:
        if os.path.isfile(p):
            try:
                z = np.load(p); mean=z["mean"].astype(np.float32); std=z["std"].astype(np.float32)
                if mean.shape[0]==feat_dim and std.shape[0]==feat_dim:
                    std[std<1e-8]=1.0; print("Scaler:", p); return mean,std
            except: pass
    print("[AVISO] sin scaler -> identidad")
    return np.zeros((feat_dim,),np.float32), np.ones((feat_dim,),np.float32)

FEAT_MEAN, FEAT_STD = load_scaler(MODEL_PATH, DATA_DIR, FEAT_DIM)

# ====== MediaPipe ======
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils
def build_hands_fast(det=0.5, trk=0.3, complexity=0):
    return mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=det, min_tracking_confidence=trk,
                          model_complexity=complexity)
def build_hands_reacq(det=0.4, complexity=0):
    return mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                          min_detection_confidence=det, model_complexity=complexity)

# ====== Features ======
from collections import deque
buffer     = deque(maxlen=SEQ_LENGTH)
pred_hist  = deque(maxlen=8)
last_valid = np.zeros((FEAT_HANDS,), dtype=np.float32)

def assemble_hands(res_h):
    left=[0.0]*63; right=[0.0]*63
    if res_h.multi_hand_landmarks and res_h.multi_handedness:
        for hl,hd in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
            coords=[]
            for p in hl.landmark:
                coords.extend([float(np.clip(p.x,0,1)), float(np.clip(p.y,0,1)), float(np.clip(p.z,-1,1))])
            if hd.classification[0].label=="Left": left=coords[:63]
            else: right=coords[:63]
    return np.array(left+right, dtype=np.float32)

def assemble_face(res_f):
    if res_f is None or not res_f.multi_face_landmarks:
        return np.zeros((FEAT_FACE,), np.float32)
    lm = res_f.multi_face_landmarks[0].landmark
    coords=[]; 
    for p in lm: coords.extend([float(np.clip(p.x,0,1)), float(np.clip(p.y,0,1)), float(np.clip(p.z,-1,1))])
    if len(coords)<FEAT_FACE: coords += [0.0]*(FEAT_FACE-len(coords))
    return np.array(coords[:FEAT_FACE], np.float32)

def is_valid(vec126, require_two=False):
    l=float(np.abs(vec126[:63]).sum()); r=float(np.abs(vec126[63:]).sum())
    return (l>0 and r>0) if require_two else (l+r>0)

def vectorize(res_h, res_f, mode:str, require_two=False):
    global last_valid
    hv = assemble_hands(res_h)
    if not is_valid(hv, require_two=require_two): hv = last_valid
    else: last_valid = hv
    if mode=="hands": return hv
    return np.concatenate([hv, assemble_face(res_f)], axis=0).astype(np.float32)

# ====== Cámara ======
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
    if not ok: c.release(); return None,(0,0)
    return c,(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# ====== TTS ======
class TTSWorker(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.q=queue.Queue(); self.last=None; self.cooldown=1.5; self.stop_flag=False
    def enqueue(self, txt): 
        if TTS_OK: self.q.put(txt)
    def run(self):
        last_t=0
        while not self.stop_flag:
            try: txt=self.q.get(timeout=0.2)
            except queue.Empty: continue
            now=time.time()
            if txt==self.last and (now-last_t)<self.cooldown: continue
            path=None
            try:
                tts=gTTS(text=txt, lang="es")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp: path=tmp.name
                tts.save(path); playsound(path)
            except Exception as e: print("[TTS] error:", e)
            finally:
                try:
                    if path and os.path.exists(path): os.remove(path)
                except: pass
                self.last=txt; last_t=time.time()
    def stop(self): self.stop_flag=True
tts_worker=TTSWorker()
if TTS_OK: tts_worker.start()

# ====== UI ======
PRIMARY_BG="#0B0F19"; CARD_BG="#FFFFFF"; BORDER="#E5E7EB"
MAX_VIDEO_H = 480  # altura máxima de render del video (para no empujar la UI)

class App:
    def __init__(self, master: tk.Tk):
        self.master=master
        master.title("SignTalk — Traductor IA (MVP)")
        master.geometry("1280x740"); master.minsize(1100,680)

        style = ttk.Style(); style.theme_use("clam")
        style.configure("Sidebar.TFrame", background=PRIMARY_BG)
        style.configure("Sidebar.TLabel", background=PRIMARY_BG, foreground="#D1D5DB")
        style.configure("SidebarHeader.TLabel", background=PRIMARY_BG, foreground="#fff", font=("Segoe UI", 13, "bold"))
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground="#111827", font=("Segoe UI", 12, "bold"))
        style.configure("Badge.TLabel", background="#F3F4F6", foreground="#111827", padding=(8,3))
        style.configure("Stat.TLabel", background=CARD_BG, foreground="#6B7280")

        # Estado
        self.running=False
        self.force_two=tk.BooleanVar(value=False)
        self.draw_hands=tk.BooleanVar(value=True)
        self.stride2=tk.BooleanVar(value=True)
        self.conf_thr=tk.DoubleVar(value=0.90)
        self.smooth_k=tk.IntVar(value=5)
        self.cam_idx=tk.IntVar(value=args.camera)
        self.res_var=tk.StringVar(value=f"{args.width}x{args.height}")
        self.fullscreen=False

        # Recursos
        self.cap=None
        self.hands_fast  = build_hands_fast()
        self.hands_reacq = build_hands_reacq()
        self.face = (mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)
                     if FEATURE_MODE=="hands_face" else None)

        # Stats
        self.fps=0.0; self._frames=0; self._t0=time.time()
        self.lat_mp_ms=0.0; self.lat_net_ms=0.0
        self.spoken_log=[]

        # ---- Layout raíz: sidebar + main ----
        root = ttk.Frame(master); root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0); root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)

        # Sidebar
        side = ttk.Frame(root, style="Sidebar.TFrame", padding=(16,16))
        side.grid(row=0, column=0, sticky="ns")
        ttk.Label(side, text="SignTalk", style="SidebarHeader.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(side, text="Traductor IA", style="Sidebar.TLabel").grid(row=1, column=0, sticky="w", pady=(0,20))
        for name in ["Inicio","Traductor","Diccionario","Historial","Comunidad","Estadísticas","Configuración","Ayuda"]:
            b=ttk.Label(side, text=f"• {name}", style="Sidebar.TLabel"); b.grid(sticky="w", pady=2)
        side.rowconfigure(99, weight=1)
        ttk.Label(side, text="v2.1.0 • Beta", style="Sidebar.TLabel").grid(row=99, column=0, sticky="sw")

        # Main (columna derecha)
        self.main = ttk.Frame(root, padding=(16,16)); self.main.grid(row=0, column=1, sticky="nsew")
        self.main.rowconfigure(2, weight=1)  # fila scrollable
        self.main.columnconfigure(0, weight=1)

        # Header
        head = ttk.Frame(self.main); head.grid(row=0, column=0, sticky="we")
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text="Traductor", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.badge = ttk.Label(head, text="Detenido", style="Badge.TLabel"); self.badge.grid(row=0, column=1, sticky="e")

        # ======== TOOLBAR FIJO (no desaparece) ========
        toolbar = ttk.Frame(self.main, padding=(0,8)); toolbar.grid(row=1, column=0, sticky="we")
        # Controles
        self.btn_start = ttk.Button(toolbar, text="▶  Iniciar", command=self.start); self.btn_start.grid(row=0, column=0, padx=(0,6))
        self.btn_pause = ttk.Button(toolbar, text="⏸  Pausar", command=self.toggle_run, state="disabled"); self.btn_pause.grid(row=0, column=1, padx=6)
        ttk.Button(toolbar, text="↺  Reset", command=self.reset).grid(row=0, column=2, padx=6)
        ttk.Checkbutton(toolbar, text="2 manos", variable=self.force_two).grid(row=0, column=3, padx=(12,6))
        ttk.Checkbutton(toolbar, text="Ver manos", variable=self.draw_hands).grid(row=0, column=4, padx=6)
        ttk.Checkbutton(toolbar, text="Modo rápido", variable=self.stride2).grid(row=0, column=5, padx=6)
        # Cámara
        ttk.Label(toolbar, text="Cam").grid(row=0, column=6, padx=(18,4))
        ttk.Spinbox(toolbar, from_=0, to=8, textvariable=self.cam_idx, width=4).grid(row=0, column=7)
        ttk.Label(toolbar, text="Res").grid(row=0, column=8, padx=(12,4))
        ttk.Combobox(toolbar, textvariable=self.res_var, values=["640x480","854x480","1280x720","1920x1080"], width=10, state="readonly").grid(row=0, column=9)
        ttk.Button(toolbar, text="Aplicar", command=self.apply_camera).grid(row=0, column=10, padx=(8,0))
        # Umbral / Suavizado
        ttk.Label(toolbar, text="Umbral").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.scale_thr = ttk.Scale(toolbar, from_=0, to=100, orient="horizontal", length=160)
        self.scale_thr.grid(row=1, column=1, columnspan=3, sticky="w", pady=(8,0))
        self.scale_thr.set(self.conf_thr.get()*100)
        ttk.Label(toolbar, text="Suavizado").grid(row=1, column=4, sticky="w", pady=(8,0))
        self.scale_smooth = ttk.Scale(toolbar, from_=1, to=8, orient="horizontal", length=120)
        self.scale_smooth.grid(row=1, column=5, columnspan=2, sticky="w", pady=(8,0))

        # ======== ÁREA SCROLLABLE (cámara + bandeja) ========
        scroll_wrap = ttk.Frame(self.main); scroll_wrap.grid(row=2, column=0, sticky="nsew")
        scroll_wrap.rowconfigure(0, weight=1); scroll_wrap.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(scroll_wrap, highlightthickness=0)
        vbar = ttk.Scrollbar(scroll_wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")

        self.scrolled = ttk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=self.scrolled, anchor="nw")
        self.scrolled.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # (wheel)
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # --- Tarjeta: Vista de Cámara ---
        cam_card = ttk.Frame(self.scrolled, style="Card.TFrame")
        cam_card.grid(row=0, column=0, sticky="we", pady=(0,12))
        cam_card.columnconfigure(0, weight=1)
        ttk.Label(cam_card, text="Vista de Cámara", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(4,6))
        self.video_label = ttk.Label(cam_card)
        self.video_label.grid(row=1, column=0, sticky="nsew")
        # Altura máxima del video para no romper layout
        self.video_label.configure(width=1100)
        self.max_video_h = MAX_VIDEO_H

        # --- Tarjeta: Palabras traducidas ---
        tray_card = ttk.Frame(self.scrolled, style="Card.TFrame")
        tray_card.grid(row=1, column=0, sticky="nsew")
        tray_card.columnconfigure(0, weight=1)
        ttk.Label(tray_card, text="Palabras Traducidas", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(4,6))
        self.txt_tray = tk.Text(tray_card, height=5, wrap="word")
        self.txt_tray.grid(row=1, column=0, sticky="nsew")
        b = ttk.Frame(tray_card); b.grid(row=2, column=0, sticky="e", pady=(6,8))
        ttk.Button(b, text="Limpiar", command=self.clear_tray).grid(row=0, column=0, padx=4)
        ttk.Button(b, text="Guardar…", command=self.save_tray).grid(row=0, column=1, padx=4)
        ttk.Button(b, text="Copiar", command=self.copy_tray).grid(row=0, column=2, padx=4)

        # Stats bar (arriba a la derecha, dentro del header)
        self.lbl_stats = ttk.Label(head, text="FPS: -- | MP: -- ms | NET: -- ms", style="Stat.TLabel")
        self.lbl_stats.grid(row=1, column=0, sticky="w")
        self.lbl_cpu = ttk.Label(head, text=("CPU: --%" if PSUTIL_OK else "CPU: n/d"), style="Stat.TLabel")
        self.lbl_cpu.grid(row=1, column=1, sticky="e")

        # Cámara inicial
        self.cap,_ = open_camera(self.cam_idx.get(), *self._parse_res(self.res_var.get()))
        if self.cap is None:
            self.cap,_ = open_camera(self.cam_idx.get(), 640, 480)
            if self.cap is None:
                messagebox.showerror("Cámara","No se pudo abrir la cámara.")

        master.protocol("WM_DELETE_WINDOW", self.on_close)

    # --------- acciones ----------
    def _parse_res(self, s):
        try: w,h=map(int,s.split("x")); return w,h
        except: return (1280,720)

    def start(self):
        if self.cap is None:
            messagebox.showerror("Cámara","No disponible."); return
        self.running=True; self.btn_pause.configure(state="normal"); self.badge.configure(text="Activo")
        self.master.after(1, self._loop)

    def toggle_run(self):
        self.running = not self.running
        self.btn_pause.configure(text=("Reanudar" if not self.running else "Pausar"))
        self.badge.configure(text=("Detenido" if not self.running else "Activo"))

    def reset(self): buffer.clear(); pred_hist.clear()

    def apply_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        w,h=self._parse_res(self.res_var.get())
        self.cap,_ = open_camera(self.cam_idx.get(), w, h)
        if self.cap is None:
            self.cap,_ = open_camera(self.cam_idx.get(), 640, 480)
            if self.cap is None: messagebox.showerror("Cámara","No se pudo abrir la cámara.")
        self.reset()

    def clear_tray(self): self.spoken_log.clear(); self.txt_tray.delete("1.0","end")
    def save_tray(self):
        txt=self.txt_tray.get("1.0","end").strip()
        if not txt: messagebox.showinfo("Guardar","No hay texto."); return
        p= filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texto","*.txt")])
        if p: open(p,"w",encoding="utf-8").write(txt); messagebox.showinfo("Guardar", f"Guardado en:\n{p}")
    def copy_tray(self): txt=self.txt_tray.get("1.0","end").strip(); self.master.clipboard_clear(); self.master.clipboard_append(txt)

    def on_close(self):
        try:
            if self.cap: self.cap.release()
        except: pass
        try:
            tts_worker.stop()
        except: pass
        self.master.destroy()

    # --------- loop ----------
    def _loop(self):
        ok, frame = self.cap.read()
        if ok: self._process_frame(frame)
        if PSUTIL_OK and (time.time()-getattr(self,"_cpu_t",0))>0.5:
            self._cpu_t=time.time()
            try: self.lbl_cpu.configure(text=f"CPU: {psutil.cpu_percent(interval=None):.0f}%")
            except: self.lbl_cpu.configure(text="CPU: n/d")
        self.master.after(1, self._loop if self.running else lambda: None)

    def _process_frame(self, bgr):
        t0=time.time()
        bgr=cv2.flip(bgr,1); rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res_h=self.hands_fast.process(rgb)
        if self.force_two.get():
            cnt=len(res_h.multi_hand_landmarks) if res_h.multi_hand_landmarks else 0
            if cnt<2: res_h=self.hands_reacq.process(rgb)
        res_f=(mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False).process(rgb)
               if FEATURE_MODE=="hands_face" else None)
        t1=time.time()

        if self.draw_hands.get() and res_h.multi_hand_landmarks:
            for hl in res_h.multi_hand_landmarks:
                mp_draw.draw_landmarks(bgr, hl, mp.solutions.hands.HAND_CONNECTIONS)

        if self.running:
            vec = vectorize(res_h, res_f, FEATURE_MODE, require_two=self.force_two.get())
            if self.stride2.get():
                if not hasattr(self,"_stride"): self._stride=0
                self._stride ^= 1
                if self._stride==0: buffer.append(vec)
            else: buffer.append(vec)

        top_text="—"
        if len(buffer)==SEQ_LENGTH:
            seq=np.array(buffer,np.float32)
            seq=(seq-FEAT_MEAN[None,:])/FEAT_STD[None,:]
            t2=time.time()
            probs=model(seq[None,...], training=False).numpy()[0]
            t3=time.time()
            self.lat_mp_ms=(t1-t0)*1000.0; self.lat_net_ms=(t3-t2)*1000.0
            pred_hist.append(probs); k=max(1,int(self.scale_smooth.get()))
            smoothed=np.mean(list(pred_hist)[-k:], axis=0)
            idx_top=smoothed.argsort()[-min(3,len(LABELS)):][::-1]
            best_idx=int(idx_top[0]); best_p=float(smoothed[best_idx])
            thr=float(self.scale_thr.get())/100.0
            top_text=f"{LABELS[best_idx]}  {best_p*100:.1f}%"
            if best_p>=thr: self._speak_and_append(LABELS[best_idx])
            # dibuja top-3
            y0=30
            for j,i in enumerate(idx_top):
                txt=f"{j+1}. {LABELS[int(i)]}  {smoothed[int(i)]*100:.1f}%"
                cv2.putText(bgr, txt, (10,y0+24*j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)

        # FPS
        self._frames+=1; now=time.time()
        if now-self._t0>=1.0: self.fps=self._frames/(now-self._t0); self._frames=0; self._t0=now
        self.lbl_stats.configure(text=f"FPS: {self.fps:.1f} | MP: {self.lat_mp_ms:.1f} ms | NET: {self.lat_net_ms:.1f} ms")

        # ---- Render con límite de altura ----
        # Calcula tamaño destino manteniendo aspect ratio para no superar MAX_VIDEO_H
        h, w = bgr.shape[:2]
        scale = min(1.0, float(self.max_video_h) / float(h))
        # intenta respetar el ancho del label si ya está medido
        target_w = self.video_label.winfo_width() or w
        scale_w = min(scale, float(target_w)/float(w)) if target_w>0 else scale
        scale = min(scale, scale_w)
        if scale != 1.0:
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _speak_and_append(self, text: str):
        if not self.spoken_log or self.spoken_log[-1]!=text:
            self.spoken_log.append(text)
            self.txt_tray.insert("end", ("" if self.txt_tray.index('end-1c')=='1.0' else " ")+text)
            self.txt_tray.see("end")
        if TTS_OK: tts_worker.enqueue(text)

# ====== main ======
if __name__=="__main__":
    root=tk.Tk()
    app=App(root)
    root.mainloop()
