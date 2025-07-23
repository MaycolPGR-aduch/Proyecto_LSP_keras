import tensorflow as tf
import numpy as np
from model import create_sequence_model
from preprocess import load_data
import os

# ---------------- Configuración ----------------
BATCH_SIZE  = 32
EPOCHS      = 60
MODEL_NAME  = "lsp_sequence_modelv2.keras"   # carpeta .keras
CHECKPOINT  = "best_lsp_model.keras"       # guarda mejor época

# 1) Carga datos
print("Cargando datos…")
X, y, labels = load_data()
num_classes = len(labels)
print(f"Clases detectadas: {labels}")

# Partición train/val manual (ya hicimos stratify en preprocess)
# Si quieres usar .npz:
# data = np.load("../data/dataset_splits.npz")
# X_train, X_val = data["X_train"], data["X_val"]
# y_train, y_val = data["y_train"], data["y_val"]

# Re-particionamos aquí (opcional)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Construye modelo
print("Creando modelo…")
model = create_sequence_model(num_classes=num_classes)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3) Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT,
        monitor="val_loss",
        save_best_only=True
    )
]

# 4) Entrenamiento
print("Iniciando entrenamiento…")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# 5) Guarda el modelo final en formato .keras
print(f"Guardando modelo en {MODEL_NAME} …")
model.save(MODEL_NAME)
print("¡Entrenamiento completado!")
