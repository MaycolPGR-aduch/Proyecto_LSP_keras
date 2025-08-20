import cv2
import numpy as np
import os
from collections import deque
import mediapipe as mp
import unicodedata

# ------------- Configuración -------------
SEQ_LENGTH = 30     # frames por muestra
FEATURES   = 126    # 21 landmarks * 3 coords * (hasta) 2 manos
DATA_DIR   = os.path.join(os.getcwd(), "..", "data")

# Estado
buffer = deque(maxlen=SEQ_LENGTH)
current_label_display = None        # etiqueta visible (con tildes/espacios)
current_label_safe    = None        # etiqueta sanitizada para carpeta/archivo
sample_count = {}                   # contador por etiqueta (clave = label_safe)

# MediaPipe
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw  = mp.solutions.drawing_utils

# ---------- Utilidades ----------
def sanitize_label(text: str) -> str:
    """
    'Buenos días, ¿cómo estás?' -> 'BUENOS_DIAS_COMO_ESTAS'
    - Mayúsculas, sin tildes/diacríticos
    - Espacios/símbolos -> '_'
    - Colapsa múltiples '_'
    """
    if not text:
        return ""
    x = text.upper().strip()
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if not unicodedata.combining(c))
    x = ''.join(c if (c.isalnum() or c in (' ', '_', '-')) else '_' for c in x)
    x = x.replace(' ', '_')
    while '__' in x:
        x = x.replace('__', '_')
    return x

def ensure_dir(label_safe: str) -> str:
    path = os.path.join(DATA_DIR, label_safe)
    os.makedirs(path, exist_ok=True)
    return path

def get_existing_count(label_safe: str) -> int:
    """Cuenta muestras .npy existentes para no reiniciar en 0."""
    folder = ensure_dir(label_safe)
    cnt = 0
    for fname in os.listdir(folder):
        if fname.lower().endswith(".npy") and fname.startswith(label_safe + "_"):
            cnt += 1
    return cnt

def set_label(display_text: str):
    """Define etiqueta actual, asegura carpeta, lee contador y limpia buffer."""
    global current_label_display, current_label_safe, buffer, sample_count
    current_label_display = display_text
    current_label_safe = sanitize_label(display_text)
    if not current_label_safe:
        return
    ensure_dir(current_label_safe)
    sample_count[current_label_safe] = get_existing_count(current_label_safe)
    buffer.clear()

# ------------- Captura -------------
cap = cv2.VideoCapture(0)
print("Controles (haz CLICK en la ventana de OpenCV para que tome el foco):")
print("- Letras/números: cambia etiqueta rápida (A..Z / 0..9)")
print("- F o P: ingresar frase/etiqueta por consola (ej.: HOLA, BUENOS DIAS)")
print("- F2: idem (si tu teclado/driver lo reporta)")
print("- ENTER: guardar muestra (30 frames)")
print("- ?: imprimir keycode (debug)")
print("- ESC: salir")

cv2.namedWindow("Captura LSP")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    # landmarks x,y,z
    lm = []
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            for p in hand_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # padding/trunc a 126 por frame
    if len(lm) < FEATURES:
        lm = lm + [0.0] * (FEATURES - len(lm))
    else:
        lm = lm[:FEATURES]

    buffer.append(lm)

    # overlay
    label_line = f"Etiqueta: {current_label_display or '(ninguna)'}"
    if current_label_safe:
        label_line += f"  [dir: {current_label_safe}]  (muestras: {sample_count.get(current_label_safe, 0)})"
    cv2.putText(frame, label_line, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "F/P: frase | F2: frase (si disponible) | A-Z/0-9: rapida | ENTER: guardar | ?: keycode | ESC: salir",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Captura LSP", frame)

    # usar waitKeyEx para teclas especiales
    key = cv2.waitKeyEx(1)

    # ESC
    if key == 27:
        print("\nFinalizando captura...")
        break

    # Debug keycode
    if key == ord('?'):
        print(f"Keycode: {key}")
        continue

    # F2 (virtual-key Windows = 0x71 = 113). Si no llega, usa F o P (abajo).
    if key == 0x71:
        cv2.destroyAllWindows()
        try:
            frase = input("\nIngresa la frase/etiqueta (ej.: HOLA, COMO ESTAS, BUENOS DIAS): ").strip()
        except EOFError:
            frase = ""
        if frase:
            set_label(frase)
            print(f"Cambiada etiqueta/frase a '{current_label_display}' (safe='{current_label_safe}')")
            print(f"Muestras actuales: {sample_count[current_label_safe]}")
        else:
            print("Etiqueta/frase vacía. No se cambió.")
        cv2.namedWindow("Captura LSP")
        continue

    # F o P -> abrir prompt de frase (fiable)
    if key in (ord('f'), ord('F'), ord('p'), ord('P')):
        cv2.destroyAllWindows()
        try:
            frase = input("\nIngresa la frase/etiqueta (ej.: HOLA, COMO ESTAS, BUENOS DIAS): ").strip()
        except EOFError:
            frase = ""
        if frase:
            set_label(frase)
            print(f"Cambiada etiqueta/frase a '{current_label_display}' (safe='{current_label_safe}')")
            print(f"Muestras actuales: {sample_count[current_label_safe]}")
        else:
            print("Etiqueta/frase vacía. No se cambió.")
        cv2.namedWindow("Captura LSP")
        continue

    # Letras A-Z o números 0-9 (rápidas)
    if (65 <= key <= 90) or (97 <= key <= 122) or (48 <= key <= 57):
        quick = chr(key).upper()
        set_label(quick)
        print(f"Cambiada etiqueta rápida a '{current_label_display}'")
        print(f"Muestras actuales: {sample_count[current_label_safe]}")
        continue

    # ENTER: guardar
    if key == 13 and current_label_safe:
        if len(buffer) == SEQ_LENGTH:
            folder = ensure_dir(current_label_safe)
            idx = sample_count[current_label_safe] + 1
            out_path = os.path.join(folder, f"{current_label_safe}_{idx:03d}.npy")
            np.save(out_path, np.array(buffer, dtype=np.float32))
            sample_count[current_label_safe] = idx
            print(f"Guardado: {out_path}")
            print(f"Etiqueta/frase: '{current_label_display}' | Total muestras: {idx}")
        else:
            print(f"Buffer incompleto: {len(buffer)}/{SEQ_LENGTH} frames. No se guardó muestra.")

cap.release()
cv2.destroyAllWindows()
