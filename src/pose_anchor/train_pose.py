# -*- coding: utf-8 -*-
"""
train_pose.py — Entrena modelo secuencial (126) para manos relativas al cuerpo
- Escala (mean/std) y guarda scaler_stats.npz junto al modelo.
- tf.data con cache/prefetch. EarlyStopping + ReduceLROnPlateau.
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
args = parser.parse_args()
os.makedirs(os.path.dirname(args.model), exist_ok=True)

# ---------- Datos ----------
X,y,labels = load_data()
Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
T,F = Xtr.shape[1], Xtr.shape[2]

# scaler (per-feature)
mean = Xtr.reshape(-1,F).mean(axis=0).astype(np.float32)
std  = Xtr.reshape(-1,F).std(axis=0).astype(np.float32); std[std<1e-8]=1.0
np.savez(os.path.join(os.path.dirname(args.model), "scaler_stats.npz"), mean=mean, std=std)

def scale(x): return (x-mean)/std
Xtr = scale(Xtr); Xva = scale(Xva)

# ---------- tf.data ----------
def ds_from(X,y,train=True):
    ds = tf.data.Dataset.from_tensor_slices((X,y))
    if train:
        ds = ds.shuffle(buffer_size=min(len(X), 1000), seed=42)
    ds = ds.batch(args.batch).cache().prefetch(tf.data.AUTOTUNE)
    return ds
ds_tr = ds_from(Xtr,ytr,True); ds_va = ds_from(Xva,yva,False)

# ---------- Modelo (ligero y estable) ----------
inputs = tf.keras.Input(shape=(T,F), name="input_sequence")
x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(inputs)
x = tf.keras.layers.DepthwiseConv1D(kernel_size=5, padding="same")(x)
x = tf.keras.layers.Conv1D(128, kernel_size=1, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Bloques TCN-lite
for _ in range(2):
    res = x
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same")(x)
    x = tf.keras.layers.Add()([x,res])
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(labels), activation="softmax", name="probs")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")])

# ---------- Callbacks ----------
cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(args.checkpoint, monitor="val_loss", save_best_only=True)
]

print("Entrenando…")
hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=cbs)

print("[INFO] Guardando modelo:", args.model)
model.save(args.model)

# guarda labels + history
json.dump({"labels": labels}, open(os.path.join(os.path.dirname(args.model),"labels.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump({k:[float(v) for v in hist.history[k]] for k in hist.history}, open(os.path.join(os.path.dirname(args.model),"history.json"),"w"), indent=2)
print("OK")
