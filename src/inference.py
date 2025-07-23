import cv2
import numpy as np
import os
from collections import deque
import mediapipe as mp
import tensorflow as tf

# ------------- Configuración -------------
SEQ_LENGTH = 30
FEATURES   = 126
MODEL_DIR  = os.path.join(os.getcwd(), "..", "lsp_sequence_model.keras")
DATA_DIR   = os.path.join(os.getcwd(), "..", "data")

# 1) Carga etiquetas (subcarpetas de data/)
labels = sorted([d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d))])

# 2) Carga modelo
print("Cargando modelo desde:", MODEL_DIR)
model = tf.keras.models.load_model(MODEL_DIR)

# 3) Prepara MediaPipe
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw  = mp.solutions.drawing_utils

# 4) Buffer para últimos SEQ_LENGTH landmarks
buffer = deque(maxlen=SEQ_LENGTH)

# 5) Inicia captura de cámara
cap = cv2.VideoCapture(0)
print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espejar imagen y procesar
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    # Extraer landmarks x,y,z
    lm = []
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            for p in hand_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Asegurar longitud FEATURES
    if len(lm) < FEATURES:
        lm = lm + [0.0] * (FEATURES - len(lm))
    else:
        lm = lm[:FEATURES]

    # Añadir al buffer
    buffer.append(lm)

    # Cuando el buffer esté lleno, predecir
    if len(buffer) == SEQ_LENGTH:
        seq = np.array(buffer, dtype=np.float32)[None, ...]  # shape (1,30,126)
        preds = model.predict(seq, verbose=0)[0]            # vector de probabilidades
        idx   = np.argmax(preds)
        prob  = preds[idx]
        text  = f"{labels[idx]} ({prob*100:.1f}%)"
        # Mostrar texto en pantalla
        cv2.putText(frame, text, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Mostrar cámara
    cv2.imshow("LSP Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
