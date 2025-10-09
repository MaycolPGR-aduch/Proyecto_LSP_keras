# -*- coding: utf-8 -*-
"""
train_pose.py — Entrena modelo secuencial (126) con ancla corporal y data augmentation
- Normaliza con mean/std (guardado en scaler_stats.npz).
- Aumentación on-the-fly:
  * Temporal: time-warp (estirar/comprimir 0.85–1.15), recorte/desplazamiento aleatorio,
              mask temporal corto (tipo SpecAugment temporal) opcional.
  * Espacial leve: jitter gaussiano pequeño, micro-scale y micro-shift por muestra.
- Pipeline eficiente con tf.data (shuffle → map(norm/aug) → batch → prefetch).
"""

import os, json, argparse, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess_pose import load_data

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch",  type=int, default=32)
parser.add_argument("--model",  type=str, default=os.path.join("..","..","models_pose","lsp_pose_v1.keras"))
parser.add_argument("--checkpoint", type=str, default=os.path.join("..","..","models_pose","best_lsp_pose_v1.keras"))
parser.add_argument("--aug", action="store_true", help="Activar data augmentation")
args = parser.parse_args()
os.makedirs(os.path.dirname(args.model), exist_ok=True)

# ---------- Datos ----------
X, y, labels = load_data()                   # X: (N, T, F) sin normalizar
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
T, F = Xtr.shape[1], Xtr.shape[2]
num_classes = len(labels)
print(f"[INFO] Shapes -> Xtr {Xtr.shape}  Xva {Xva.shape}  num_classes={num_classes}")

# ---------- Scaler (per-feature) ----------
mean = Xtr.reshape(-1, F).mean(axis=0).astype(np.float32)
std  = Xtr.reshape(-1, F).std(axis=0).astype(np.float32)
std[std < 1e-8] = 1.0
scaler_path = os.path.join(os.path.dirname(args.model), "scaler_stats.npz")
np.savez(scaler_path, mean=mean, std=std)
print("[INFO] Scaler guardado en:", scaler_path)

# ---------- Aumentación (tf ops, sin numpy) ----------
# Notas:
# - Trabajamos en el espacio NORMALIZADO (tras (x-mean)/std) para usar sigmas estables.
# - Mantener magnitudes pequeñas: jitter=0.02 aprox., micro-scale en [0.98, 1.02], micro-shift ±0.02.
# - Time-warp: factor 0.85–1.15 con resample lineal + recorte/pad a T.
# - Temporal shift pequeño: desplazar la ventana unos pocos frames.

@tf.function
def normalize(x):
    return (x - tf.constant(mean)[tf.newaxis, :]) / tf.constant(std)[tf.newaxis, :]

def _resample_linear(x, new_len):
    """x: (T,F) -> (new_len,F) por interpolación lineal."""
    t = tf.cast(tf.shape(x)[0], tf.float32)
    n = tf.cast(new_len, tf.float32)
    # posiciones destino en [0, T-1]
    idx_f = tf.linspace(0.0, t - 1.0, new_len)
    idx0 = tf.cast(tf.floor(idx_f), tf.int32)
    idx1 = tf.minimum(idx0 + 1, tf.shape(x)[0] - 1)
    w    = idx_f - tf.cast(idx0, tf.float32)
    x0 = tf.gather(x, idx0)            # (new_len, F)
    x1 = tf.gather(x, idx1)
    return x0 * (1.0 - w[:, None]) + x1 * (w[:, None])

def _time_warp(x, min_factor=0.85, max_factor=1.15):
    """Estira/encoge temporalmente y vuelve a tamaño T."""
    T = tf.shape(x)[0]
    factor = tf.random.uniform([], min_factor, max_factor)
    new_len = tf.cast(tf.round(tf.cast(T, tf.float32) * factor), tf.int32)
    new_len = tf.maximum(8, new_len)
    xw = _resample_linear(x, new_len)
    # recorte o pad para volver a T
    cur = tf.shape(xw)[0]
    def _crop():
        start = tf.random.uniform([], 0, cur - T + 1, dtype=tf.int32)
        return xw[start:start+T]
    def _pad():
        pad = tf.maximum(0, T - cur)
        return tf.pad(xw, [[0, pad], [0, 0]])[:T]
    return tf.cond(cur > T, _crop, _pad)

def _time_mask(x, max_frac=0.15):
    """Enmascara un segmento corto temporalmente (tipo SpecAugment temporal)."""
    T = tf.shape(x)[0]
    max_len = tf.cast(tf.round(tf.cast(T, tf.float32) * max_frac), tf.int32)
    max_len = tf.maximum(1, max_len)
    seg = tf.random.uniform([], 1, max_len + 1, dtype=tf.int32)
    start = tf.random.uniform([], 0, tf.maximum(1, T - seg + 1), dtype=tf.int32)
    mask = tf.concat([
        tf.ones((start, 1), tf.float32),
        tf.zeros((seg, 1), tf.float32),
        tf.ones((T - start - seg, 1), tf.float32)
    ], axis=0)
    return x * mask  # zero-out segmento

def _spatial_jitter(x, jitter_std=0.02, scale_min=0.98, scale_max=1.02, shift_max=0.02):
    """Ruido leve + micro-scale + micro-shift (aplicado por muestra)."""
    F = tf.shape(x)[1]
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=jitter_std)
    s = tf.random.uniform([1, F], scale_min, scale_max)
    sh = tf.random.uniform([1, F], -shift_max, shift_max)
    return x * s + sh + noise

def augment_sample(x, y):
    # x llega NORMALIZADO (T,F)
    # pipeline temporal (orden conservador):
    if tf.random.uniform([]) < 0.75:   # 75%: time-warp
        x = _time_warp(x)
    if tf.random.uniform([]) < 0.50:   # 50%: temporal mask corto
        x = _time_mask(x)
    # pequeño shift temporal (recorte + pad)
    if tf.random.uniform([]) < 0.50:
        T = tf.shape(x)[0]
        shift = tf.random.uniform([], -4, 5, dtype=tf.int32)  # [-4, 4]
        def _shift_right():
            padd = tf.zeros((shift, tf.shape(x)[1]), x.dtype)
            return tf.concat([padd, x[:-shift]], axis=0)
        def _shift_left():
            padd = tf.zeros((-shift, tf.shape(x)[1]), x.dtype)
            return tf.concat([x[-shift:], padd], axis=0)
        x = tf.cond(shift > 0, _shift_right, lambda: tf.cond(shift < 0, _shift_left, lambda: x))
    # espacial leve (después de temporal)
    x = _spatial_jitter(x)
    return x, y

def normalize_sample(x, y):
    return normalize(x), y

# ---------- tf.data ----------
def ds_from(X, y, train=True, do_aug=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(buffer_size=min(len(X), 1000), seed=42, reshuffle_each_iteration=True)
    # Normalización siempre dentro del grafo
    ds = ds.map(lambda a, b: (normalize(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    if train and do_aug:
        ds = ds.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)
    return ds

ds_tr = ds_from(Xtr, ytr, train=True,  do_aug=args.aug)
ds_va = ds_from(Xva, yva, train=False, do_aug=False)

# ---------- Modelo (TCN-lite estable/eficiente) ----------
inputs = tf.keras.Input(shape=(T, F), name="input_sequence")
x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(inputs)
x = tf.keras.layers.DepthwiseConv1D(kernel_size=5, padding="same")(x)
x = tf.keras.layers.Conv1D(128, kernel_size=1, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

for _ in range(2):
    res = x
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same")(x)
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")]
)

# ---------- Callbacks ----------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint, monitor="val_loss", save_best_only=True),
]

print(f"[INFO] Entrenando… (aug={'ON' if args.aug else 'OFF'})")
hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=callbacks)

# ---------- Guardados ----------
print("[INFO] Guardando modelo en:", args.model)
model.save(args.model)
json.dump({"labels": labels}, open(os.path.join(os.path.dirname(args.model), "labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump({k:[float(v) for v in hist.history[k]] for k in hist.history},
          open(os.path.join(os.path.dirname(args.model), "history.json"), "w"), indent=2)
print("OK")
