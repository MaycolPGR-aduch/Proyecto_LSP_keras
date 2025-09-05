# -*- coding: utf-8 -*-
"""
train2.py — Entrenamiento LSP (solo manos = 126 features)
- Lee splits y stats desde preprocess.build_datasets()
- Dataset tf.data con cache, shuffle y prefetch para máximo rendimiento
- Adam con clip de gradiente, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Soporte opcional de pesos de clase (auto desde preprocess o computados)
- Guardado de modelo .keras + history.json + labels.json

Uso (opcional):
    python train2.py --data-dir ../coordenadas --batch-size 32 --epochs 60 --lr 1e-3 --use-class-weights yes
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default=None,
                    help="Carpeta del dataset (por defecto la que usa preprocess.py)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--model-name", type=str, default="lsp_sequence_modelv3.keras")
parser.add_argument("--checkpoint", type=str, default="best_lsp_modelv3.keras")
parser.add_argument("--use-class-weights", type=str, default="auto",
                    choices=["auto", "yes", "no"],
                    help="auto: usa class_weights.json si existe; yes: siempre computa; no: desactiva")
parser.add_argument("--mixed", action="store_true",
                    help="Habilitar mixed-precision si hay GPU compatible (opcional)")
args = parser.parse_args()

# ---------------- Semillas / determinismo ----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# ---------------- Mixed precision (opcional y seguro) ----------------
if args.mixed and tf.config.list_physical_devices("GPU"):
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision ACTIVADA.")
    except Exception as e:
        print(f"[WARN] No se pudo activar mixed precision: {e}")

# ---------------- Importar preprocess y model ----------------
import preprocess as pp  # nuestro preprocess manos-only
from model import create_sequence_model

# Permitir cambiar BASE_DIR y/o SEQ_LENGTH de preprocess desde CLI
if args.data_dir:
    pp.BASE_DIR = os.path.abspath(args.data_dir)
    print(f"[INFO] BASE_DIR (preprocess) -> {pp.BASE_DIR}")

# Construir datasets + guardar artefactos (scaler, labels, class_weights)
print("Cargando y preprocesando datos…")
X_train, X_val, y_train, y_val, labels = pp.build_datasets(save_outputs=True)

num_classes = len(labels)
seq_len  = X_train.shape[1]
feat_dim = X_train.shape[2]
print(f"[INFO] Clases ({num_classes}): {labels}")
print(f"[INFO] Input shape: (seq_length={seq_len}, feat_dim={feat_dim})")

# Sanidad: esperamos 126 features
if feat_dim != 126:
    print(f"[WARN] feat_dim={feat_dim} (esperado 126). Continuando de todos modos.")

# ---------------- tf.data pipelines ----------------
AUTOTUNE = tf.data.AUTOTUNE

def make_ds(X, y, batch, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        # shuffle con un buffer razonable (hasta 10k o tamaño del dataset)
        buf = int(min(10000, len(y)))
        ds = ds.shuffle(buf, seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.cache()
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(X_train, y_train, args.batch_size, training=True)
val_ds   = make_ds(X_val,   y_val,   args.batch_size, training=False)

# ---------------- Modelo ----------------
model = create_sequence_model(
    num_classes=num_classes,
    seq_length=seq_len,
    feat_dim=feat_dim
)

# Optimizador con clip para estabilidad
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")
    ],
)

# ---------------- Class weights ----------------
class_weight = None
cw_path = os.path.join(pp.BASE_DIR, "class_weights.json")
if args.use_class_weights == "auto":
    if os.path.isfile(cw_path):
        try:
            with open(cw_path, "r", encoding="utf-8") as f:
                cw_raw = json.load(f)
            # keys podrían venir como str -> convertir a int
            class_weight = {int(k): float(v) for k, v in cw_raw.items()}
            print(f"[INFO] Usando class_weights.json -> {class_weight}")
        except Exception as e:
            print(f"[WARN] No se pudo leer class_weights.json: {e}")
            class_weight = None
elif args.use_class_weights == "yes":
    # Computar desde y_train
    uniq, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    class_weight = {int(k): float(total / (len(uniq) * c)) for k, c in zip(uniq, counts)}
    print(f"[INFO] class_weight computado -> {class_weight}")
else:
    class_weight = None
    print("[INFO] Entrenando sin class_weight.")

# ---------------- Callbacks ----------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,  # guarda modelo completo (.keras si ext .keras)
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger("training_log.csv", append=False)
]

# ---------------- Entrenamiento ----------------
print("Iniciando entrenamiento…")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks,
    class_weight=class_weight
)

# ---------------- Guardados ----------------
# Mejor checkpoint ya guardado por ModelCheckpoint
print(f"[INFO] Guardando modelo final en: {args.model_name}")
model.save(args.model_name)

# Guardar labels al lado del modelo (además de los que guardó preprocess)
labels_path = os.path.join(os.path.dirname(args.model_name) or ".", "labels.json")
try:
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Etiquetas guardadas en {labels_path}")
except Exception as e:
    print(f"[WARN] No se pudo guardar labels.json junto al modelo: {e}")

# Guardar history para análisis posterior
try:
    hist_path = os.path.join(os.path.dirname(args.model_name) or ".", "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    print(f"[INFO] Historial de entrenamiento guardado en {hist_path}")
except Exception as e:
    print(f"[WARN] No se pudo guardar history.json: {e}")

print("¡Entrenamiento completado!")
