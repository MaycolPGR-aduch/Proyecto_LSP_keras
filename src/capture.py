import cv2
import numpy as np
import os
from collections import deque
import mediapipe as mp

# ------------- Configuración -------------
SEQ_LENGTH = 30                           # frames por muestra
DATA_DIR   = os.path.join(os.getcwd(), "..", "data")
LABELS     = []                           # lista dinámica de etiquetas
buffer     = deque(maxlen=SEQ_LENGTH)    # almacena últimos SEQ_LENGTH landmarks
current_label = None                      # etiqueta/frase activa
sample_count   = {}                       # contador de muestras por etiqueta/frase

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw  = mp.solutions.drawing_utils

# Crea la carpeta para cada etiqueta si no existe
def ensure_dir(label):
    path = os.path.join(DATA_DIR, label)
    os.makedirs(path, exist_ok=True)
    return path

# ------------- Captura -------------

cap = cv2.VideoCapture(0)
print("F2: ingresar frase/etiqueta. ENTER: grabar muestra. ESC: salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear y dibujar
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    # Extraer landmarks (x,y,z) de cada mano
    lm = []
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            for p in hand_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
            # Opcional: dibujar puntos en la mano
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # después de construir lm con los landmark x,y,z de 0–2 manos...
    # Asegurar longitud fija de 126 floats:
    if len(lm) < 126:
        lm = lm + [0.0] * (126 - len(lm))
    else:
        lm = lm[:126]

    buffer.append(lm)

    # Mostrar etiqueta activa
    # Mostrar frase/etiqueta activa
    label_text = f"Etiqueta: {current_label}" if current_label else "Etiqueta: (ninguna)"
    cv2.putText(frame, label_text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Captura LSP", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("\nFinalizando captura...")
        break
    # F2: ingresar frase/etiqueta
    elif key == 113:  # F2 key
        cv2.destroyAllWindows()
        print("\nIngresa la frase/etiqueta para capturar (ejemplo: HOLA, COMO ESTAS):")
        frase = input().strip()
        if frase:
            current_label = frase.upper()
            sample_count.setdefault(current_label, 0)
            if current_label not in LABELS:
                LABELS.append(current_label)
            print(f"Cambiada etiqueta/frase a '{current_label}'")
            print(f"Muestras actuales de '{current_label}': {sample_count[current_label]}")
        else:
            print("Etiqueta/frase vacía. No se cambió.")
        # Reabrir ventana de OpenCV
        cv2.namedWindow("Captura LSP")
    # A–Z (mayúsculas y minúsculas) para etiquetas rápidas
    elif (65 <= key <= 90) or (97 <= key <= 122) or (48 <= key <= 57):
        current_label = chr(key).upper()
        sample_count.setdefault(current_label, 0)
        if current_label not in LABELS:
            LABELS.append(current_label)
        print(f"Cambiada etiqueta rápida a '{current_label}'")
        print(f"Muestras actuales de '{current_label}': {sample_count[current_label]}")
    # ENTER: guardar secuencia completa
    elif key == 13 and current_label:
        if len(buffer) == SEQ_LENGTH:
            folder = ensure_dir(current_label)
            idx = sample_count[current_label] + 1
            # Reemplazar caracteres no válidos en nombre de archivo/carpeta
            safe_label = ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in current_label)
            filename = os.path.join(folder, f"{safe_label}_{idx:03d}.npy")
            np.save(filename, np.array(buffer))
            sample_count[current_label] = idx
            print(f"Guardado: {filename}")
            print(f"Etiqueta/frase: '{current_label}' | Total muestras: {idx}")
        else:
            print(f"Buffer incompleto: {len(buffer)}/{SEQ_LENGTH} frames. No se guardó muestra.")

cap.release()
cv2.destroyAllWindows()
