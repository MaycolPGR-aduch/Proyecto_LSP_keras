# -*- coding: utf-8 -*-
"""
train_pose_unificado.py — Entrena modelo secuencial (126) alineado al nuevo flujo Pose+Hands

Flujo:
  1) Capturar muestras con capture_pose.py → coordenadas_pose/<ETIQUETA>/<ETIQUETA_###.npy
  2) Preprocesar con preprocess_pose.load_data() (re-muestreo + jitter opcional)
  3) Entrenar este script → guarda modelo + scaler + labels en models_pose/
  4) Usar modelo y scaler en inference_pose_estable.py

Guarda junto al modelo:
  - scaler_stats.npz  (mean, std)
  - labels.json       (lista de etiquetas)
  - history.json      (historial de entrenamiento)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocess_pose import load_data

# ---------------------------------------------------------------------
# Paths base (asumiendo estructura Proyecto_LSP_keras/)
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS_DIR = os.path.join(ROOT, "models_pose")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=80, help="Número de épocas")
parser.add_argument("--batch", type=int, default=32, help="Tamaño de batch")
parser.add_argument("--test_size", type=float, default=0.2, help="Proporción de validación")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate inicial")
parser.add_argument(
    "--model",
    type=str,
    default=os.path.join(MODELS_DIR, "lsp_pose_v1.keras"),
    help="Ruta donde guardar el modelo .keras",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=os.path.join(MODELS_DIR, "lsp_pose_v1_best.keras"),
    help="Ruta para guardar el mejor modelo (val_loss mínima)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------
# Semillas para reproducibilidad básica
# ---------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------
# Carga de datos (X: (N,T,F), y: (N,))
# ---------------------------------------------------------------------
print("[INFO] Cargando datos desde coordenadas_pose/ ...")
X, y, labels = load_data()   # definido en preprocess_pose.py

N, T, F = X.shape
num_classes = len(labels)

print(f"[INFO] Datos cargados: N={N}, T={T}, F={F}, clases={num_classes}")
print("[INFO] Etiquetas:", labels)

# ---------------------------------------------------------------------
# Split train/val
# ---------------------------------------------------------------------
print(f"[INFO] Dividiendo en train/val con test_size={args.test_size} ...")
X_tr, X_va, y_tr, y_va = train_test_split(
    X,
    y,
    test_size=args.test_size,
    random_state=SEED,
    stratify=y,
)
print("[INFO] Shapes ->")
print("  X_train:", X_tr.shape, " y_train:", y_tr.shape)
print("  X_val:  ", X_va.shape, " y_val:  ", y_va.shape)

# ---------------------------------------------------------------------
# Cálculo de scaler (mean/std) sobre TRAIN
# ---------------------------------------------------------------------
print("[INFO] Calculando mean/std (scaler) sobre TRAIN ...")
X_flat = X_tr.reshape(-1, F)      # (N*T, F)
mean = X_flat.mean(axis=0).astype(np.float32)
std = X_flat.std(axis=0).astype(np.float32)
std[std < 1e-6] = 1.0             # evitar divisiones por cero

scaler_path = os.path.join(os.path.dirname(args.model), "scaler_stats.npz")
np.savez(scaler_path, mean=mean, std=std)
print("[INFO] scaler_stats guardado en:", scaler_path)

# ---------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------
def make_ds(X, y, mean_vec, std_vec, batch_size, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)

    mean_tf = tf.constant(mean_vec, dtype=tf.float32)    # (F,)
    std_tf = tf.constant(std_vec, dtype=tf.float32)      # (F,)

    def _norm(x, y):
        # x: (T,F)
        x = (x - mean_tf) / std_tf
        return x, y

    ds = ds.map(_norm, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

ds_tr = make_ds(X_tr, y_tr, mean, std, args.batch, training=True)
ds_va = make_ds(X_va, y_va, mean, std, args.batch, training=False)

# ---------------------------------------------------------------------
# Modelo secuencial Conv1D
# ---------------------------------------------------------------------
def build_model(seq_len, feat_dim, num_classes):
    inp = tf.keras.Input(shape=(seq_len, feat_dim), name="seq")

    x = tf.keras.layers.Masking(mask_value=0.0)(inp)

    # Bloque 1
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Bloque 2
    x = tf.keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Bloque 3 (más ligero)
    x = tf.keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="lsp_pose_seq_model")
    return model

print("[INFO] Construyendo modelo ...")
model = build_model(T, F, num_classes)
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------
early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1,
)

ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=args.checkpoint,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

callbacks = [early, reduce, ckpt]

# ---------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------
print(f"[INFO] Entrenando… epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
hist = model.fit(
    ds_tr,
    validation_data=ds_va,
    epochs=args.epochs,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------
# Guardado del modelo y artefactos
# ---------------------------------------------------------------------
print("[INFO] Guardando modelo final en:", args.model)
model.save(args.model)

# labels.json
labels_path = os.path.join(os.path.dirname(args.model), "labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)
print("[INFO] labels.json guardado en:", labels_path)

# history.json
history_path = os.path.join(os.path.dirname(args.model), "history.json")
hist_dict = {k: [float(v) for v in hist.history[k]] for k in hist.history}
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(hist_dict, f, indent=2)
print("[INFO] history.json guardado en:", history_path)

print("[INFO] Entrenamiento completado correctamente.")
print("[INFO] Modelo listo para usar en inference_pose_estable.py")
