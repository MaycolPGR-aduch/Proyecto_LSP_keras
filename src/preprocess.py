# -*- coding: utf-8 -*-
"""
preprocess.py — LSP manos-only (126) optimizado para rendimiento del modelo
- Carga .npy (L x 126) por clase desde ../coordenadas/<clase>/*.npy
- Arregla arrays 1D "object" antiguos
- Filtra muestras de baja calidad (frames sin manos)
- Rellena huecos (frames sin manos) por interpolación temporal
- Re-muestrea temporalmente a SEQ_LENGTH por interpolación (mejor que pad/trunc)
- Estandariza características con media/desv. estándar del split de entrenamiento
- Devuelve X_train/X_val/y_train/y_val y guarda:
    • dataset_splits.npz
    • scaler_stats.npz (mean, std)
    • labels.json  (mapeo etiqueta->índice)
    • class_weights.json (útil para entrenamiento balanceado)
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

# ================== Configuración ==================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.join(SCRIPT_DIR, "..", "coordenadas")
FEATURES_T   = 126           # SOLO MANOS (21*3*2)
SEQ_LENGTH   = int(os.environ.get("LSP_SEQ_LENGTH", "60"))   # ~2s a 30fps por defecto
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

# Calidad: proporción mínima de frames "válidos" (con alguna mano presente)
MIN_VALID_RATIO = float(os.environ.get("LSP_MIN_VALID_RATIO", "0.6"))

# Si un archivo trae 1530 features (dataset viejo), recortamos a las 126 primeras (manos)
ACCEPT_OLD_1530 = True

# ================== Utilidades ==================
def _fix_1d_object_array(seq_obj):
    """Repara .npy antiguos guardados como array 1D de listas/arrays -> (L, F) float32."""
    cleaned = []
    for elm in seq_obj:
        vals = list(elm) if hasattr(elm, "__len__") else []
        cleaned.append(vals)
    return np.array(cleaned, dtype=np.float32)

def _to_float32(a):
    return a.astype(np.float32, copy=False)

def _load_sequence(path):
    try:
        seq = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] No se pudo cargar {path}: {e}")
        return None

    if isinstance(seq, np.ndarray) and seq.ndim == 1:
        seq = _fix_1d_object_array(seq)

    if not isinstance(seq, np.ndarray) or seq.ndim != 2:
        print(f"[WARN] Saltando {os.path.basename(path)}: ndim inválido ({getattr(seq, 'ndim', None)})")
        return None

    L, F = seq.shape
    if F == 1530 and ACCEPT_OLD_1530:
        seq = seq[:, :FEATURES_T]  # manos al inicio
    elif F != FEATURES_T:
        if F < FEATURES_T:
            pad = np.zeros((L, FEATURES_T - F), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=1)
            print(f"[INFO] Pad features {F}->{FEATURES_T} en {os.path.basename(path)}")
        else:
            seq = seq[:, :FEATURES_T]
            print(f"[INFO] Trunc features {F}->{FEATURES_T} en {os.path.basename(path)}")

    return _to_float32(seq)

def _valid_frame_mask(seq):
    """True si el frame tiene alguna mano presente (no todos los 126 en cero)."""
    # suma absoluta por fila; si es 0, no hay manos
    return (np.abs(seq).sum(axis=1) > 0.0)

def _fill_gaps_interpolate(seq):
    """
    Rellena frames sin manos (filas en cero) interpolando temporalmente.
    Si todos los frames están vacíos -> devuelve None (descartar muestra).
    """
    L, F = seq.shape
    mask = _valid_frame_mask(seq)
    if not mask.any():
        return None

    # Índices válidos y todos
    t_all = np.arange(L, dtype=np.float32)
    t_ok  = t_all[mask]
    filled = np.empty_like(seq, dtype=np.float32)

    # Interpolación 1D por cada feature sobre el tiempo (vectorizado por columna)
    for j in range(F):
        y = seq[:, j]
        y_ok = y[mask]
        # np.interp requiere al menos 1 punto; si solo hay 1, extendemos constante
        if y_ok.shape[0] == 1:
            filled[:, j] = y_ok[0]
        else:
            filled[:, j] = np.interp(t_all, t_ok, y_ok)

    return filled

def _resample_time_interpolate(seq, target_len):
    """Re-muestrea temporalmente a target_len por interpolación lineal."""
    L, F = seq.shape
    if L == target_len:
        return seq.astype(np.float32, copy=False)
    t_old = np.linspace(0.0, 1.0, num=L, dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, F), dtype=np.float32)
    for j in range(F):
        out[:, j] = np.interp(t_new, t_old, seq[:, j])
    return out

def _sequence_quality_ok(seq):
    """Filtro de calidad: requiere proporción mínima de frames válidos."""
    mask = _valid_frame_mask(seq)
    ratio = mask.mean() if mask.size else 0.0
    return ratio >= MIN_VALID_RATIO, ratio

def _gather_files(base_dir):
    labels = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith("_")
    ])
    files = []
    for lab in labels:
        folder = os.path.join(base_dir, lab)
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(".npy"):
                files.append((lab, os.path.join(folder, filename)))
    return labels, files

# ================== Pipeline de carga ==================
def load_data():
    print(f"[preprocess] Leyendo desde: {BASE_DIR}")
    if not os.path.isdir(BASE_DIR):
        raise RuntimeError(f"No existe la carpeta de datos: {BASE_DIR}")

    labels, files = _gather_files(BASE_DIR)
    if not labels or not files:
        raise RuntimeError("No se encontraron clases o archivos .npy en 'coordenadas/'.")

    label2idx = {lab: i for i, lab in enumerate(labels)}
    kept, dropped = 0, 0
    qual_hist = []

    X_list, y_list = [], []

    for lab, path in files:
        seq = _load_sequence(path)
        if seq is None:
            dropped += 1
            continue

        # Rellenar huecos (frames sin manos) por interpolación temporal
        seq_filled = _fill_gaps_interpolate(seq)
        if seq_filled is None:
            print(f"[WARN] Muestra sin manos en todo el video: {os.path.basename(path)}")
            dropped += 1
            continue

        # Chequeo de calidad
        ok, ratio = _sequence_quality_ok(seq_filled)
        qual_hist.append(ratio)
        if not ok:
            print(f"[WARN] Baja calidad (valid_ratio={ratio:.2f}<{MIN_VALID_RATIO}) -> descarto {os.path.basename(path)}")
            dropped += 1
            continue

        # Re-muestreo temporal a longitud fija por interpolación
        seq_fixed = _resample_time_interpolate(seq_filled, SEQ_LENGTH)

        X_list.append(_to_float32(seq_fixed))
        y_list.append(label2idx[lab])
        kept += 1

    if kept == 0:
        raise RuntimeError("No quedaron muestras tras el filtrado de calidad. Revisa MIN_VALID_RATIO o tus capturas.")

    X = np.stack(X_list, axis=0).astype(np.float32)   # (N, SEQ_LENGTH, 126)
    y = np.array(y_list, dtype=np.int32)

    print(f"[preprocess] Total archivos: {len(files)} | Usados: {kept} | Desc. por calidad/errores: {dropped}")
    if qual_hist:
        print(f"[preprocess] Valid ratio medio: {np.mean(qual_hist):.2f}  (min={np.min(qual_hist):.2f}, max={np.max(qual_hist):.2f})")
    return X, y, labels

# ================== División, estandarización y guardado ==================
def build_datasets(save_outputs=True):
    X, y, labels = load_data()

    # Split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Estandarización por feature usando SOLO el split de entrenamiento
    # Evita fuga de información y mejora convergencia del modelo
    feat_mean = X_train.mean(axis=(0, 1), dtype=np.float64)  # (126,)
    feat_std  = X_train.std(axis=(0, 1), dtype=np.float64)   # (126,)
    # Evitar división por cero
    feat_std[feat_std < 1e-8] = 1.0

    def _standardize(a):
        return ((a - feat_mean) / feat_std).astype(np.float32, copy=False)

    X_train = _standardize(X_train)
    X_val   = _standardize(X_val)

    print(f"Clases ({len(labels)}): {labels}")
    print(f"Shapes -> X_train: {X_train.shape}  X_val: {X_val.shape}  (SEQ={SEQ_LENGTH}, FEAT={FEATURES_T})")

    if save_outputs:
        ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
        ensure_dir(BASE_DIR)

        # Splits
        out_npz = os.path.join(BASE_DIR, "dataset_splits.npz")
        np.savez_compressed(out_npz,
                            X_train=X_train, X_val=X_val,
                            y_train=y_train, y_val=y_val,
                            labels=np.array(labels))
        print(f"[preprocess] Splits guardados en {out_npz}")

        # Scaler
        scaler_path = os.path.join(BASE_DIR, "scaler_stats.npz")
        np.savez(scaler_path, mean=feat_mean.astype(np.float32), std=feat_std.astype(np.float32))
        print(f"[preprocess] Scaler guardado en {scaler_path}")

        # Labels mapping
        labels_path = os.path.join(BASE_DIR, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({lab: i for i, lab in enumerate(labels)}, f, ensure_ascii=False, indent=2)
        print(f"[preprocess] Labels guardados en {labels_path}")

        # Class weights (inversa de la frecuencia)
        uniq, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        class_weights = {int(k): float(total / (len(uniq) * c)) for k, c in zip(uniq, counts)}
        cw_path = os.path.join(BASE_DIR, "class_weights.json")
        with open(cw_path, "w", encoding="utf-8") as f:
            json.dump(class_weights, f, ensure_ascii=False, indent=2)
        print(f"[preprocess] Pesos de clase guardados en {cw_path}  -> {class_weights}")

    return X_train, X_val, y_train, y_val, labels

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, labels = build_datasets(save_outputs=True)

