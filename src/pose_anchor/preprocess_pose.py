# -*- coding: utf-8 -*-
"""
preprocess_pose.py — Preprocesamiento para coordenadas (L,126) relativas al cuerpo.

- Lee secuencias guardadas por capture_pose.py desde coordenadas_pose/<ETIQUETA>/<ETIQUETA_###.npy>
  (y sus .json opcionales de metadatos).
- Filtra secuencias inválidas o demasiado cortas.
- Re-muestrea temporalmente todas las secuencias a SEQ_LENGTH frames mediante interpolación lineal.
- (Opcional) Aplica un aumento de datos sencillo y seguro (jitter gaussiano leve).
- Devuelve X, y, labels compatibles con train_pose.py / train_pose2.py.
- Opcionalmente guarda los splits en un .npz (build_datasets()).

NOTA:
- La normalización (mean/std) y cualquier data augmentation más compleja
  se hace en los scripts de entrenamiento (train_pose*.py).
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------- Configuración --------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", "coordenadas_pose"))

FEATURES_T   = 126    # dimensión de características por frame (2 manos * 21 * (x,y,z))
SEQ_LENGTH   = 60     # nº de frames objetivo por secuencia
MIN_FRAMES   = 10     # mínimo de frames para aceptar una secuencia
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

# Data augmentation sencillo (puedes desactivarlo si quieres)
ENABLE_AUGMENT      = True   # si False, no se generan copias aumentadas
AUG_PER_SAMPLE      = 1      # nº de copias aumentadas por secuencia original
AUG_NOISE_STD       = 0.02   # desviación estándar del jitter gaussiano (en espacio ya normalizado al cuerpo)
AUG_CLIP_VALUE      = 3.0    # recorte de valores extremos tras augment

# -------------------- Utilidades internas --------------------
def _iter_label_dirs(base_dir):
    """Itera sobre carpetas de etiquetas dentro de BASE_DIR."""
    if not os.path.isdir(base_dir):
        return
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            yield name, full

def _load_sequence_with_meta(npy_path):
    """
    Carga una secuencia .npy y su .json de metadatos (si existe).
    Devuelve (seq: np.ndarray (L,FEATURES_T), meta: dict).
    Si la secuencia es inválida, devuelve (None, None).
    """
    try:
        arr = np.load(npy_path)
    except Exception as e:
        print(f"[WARN] No se pudo cargar {npy_path}: {e}")
        return None, None

    if arr.ndim != 2 or arr.shape[1] != FEATURES_T:
        print(f"[WARN] Forma inesperada en {npy_path}: {arr.shape}, se espera (L,{FEATURES_T})")
        return None, None

    L = arr.shape[0]
    if L < MIN_FRAMES:
        print(f"[WARN] Secuencia muy corta ({L} frames) en {npy_path}, se descarta.")
        return None, None

    # evitar NaNs / infs
    if not np.isfinite(arr).all():
        print(f"[WARN] Secuencia con NaNs/inf en {npy_path}, se descarta.")
        return None, None

    # recorte suave de valores extremos
    arr = np.clip(arr, -AUG_CLIP_VALUE, AUG_CLIP_VALUE).astype(np.float32)

    # metadatos
    meta_path = os.path.splitext(npy_path)[0] + ".json"
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[WARN] No se pudo leer meta {meta_path}: {e}")
            meta = {}
    else:
        meta = {}

    return arr, meta

def _resample_linear(seq, target_len):
    """
    Re-muestreo temporal lineal: (L,F) -> (target_len,F).
    Si L == target_len, devuelve copia.
    """
    seq = np.asarray(seq, dtype=np.float32)
    L, F = seq.shape
    if L == target_len:
        return seq.copy()

    # índices originales y nuevos (en eje temporal)
    old_idx = np.arange(L, dtype=np.float32)
    new_idx = np.linspace(0.0, float(L - 1), target_len, dtype=np.float32)

    out = np.empty((target_len, F), dtype=np.float32)
    for d in range(F):
        out[:, d] = np.interp(new_idx, old_idx, seq[:, d])

    return out

def _augment_sequence_jitter(seq):
    """
    Aumento muy simple: jitter gaussiano leve por frame:
      seq_aug = seq + N(0, sigma^2).
    Este tipo de augment es razonablemente seguro porque:
      - ya estamos en un espacio normalizado por el cuerpo,
      - el sigma es pequeño,
      - no cambia la estructura global de la seña.
    """
    noise = np.random.normal(loc=0.0, scale=AUG_NOISE_STD, size=seq.shape).astype(np.float32)
    aug = seq + noise
    aug = np.clip(aug, -AUG_CLIP_VALUE, AUG_CLIP_VALUE)
    return aug

# -------------------- API principal --------------------
def load_data():
    """
    Carga todos los .npy en coordenadas_pose/, los re-muestrea a SEQ_LENGTH,
    aplica (opcionalmente) un augmentation sencillo, y devuelve:

        X: np.ndarray (N, SEQ_LENGTH, FEATURES_T)
        y: np.ndarray (N,)
        labels: lista de nombres de etiqueta (en el orden de índices de y)

    Además:
    - Ignora secuencias ilegales o muy cortas.
    - Imprime un pequeño resumen con nº de muestras por clase.
    """
    all_X = []
    all_y = []
    all_meta = []   # por si luego quieres usar metadatos
    label_names = []

    # Descubrir etiquetas
    for label_name, folder in _iter_label_dirs(BASE_DIR):
        label_names.append(label_name)

    if not label_names:
        raise RuntimeError(f"No se encontraron carpetas de etiquetas en {BASE_DIR}")

    # Ordenar etiquetas para tener un índice estable
    label_names = sorted(label_names)
    label_to_idx = {lab: i for i, lab in enumerate(label_names)}

    print(f"[INFO] Etiquetas encontradas ({len(label_names)}): {label_names}")

    # Recorrer todas las secuencias
    n_raw = 0
    n_kept = 0
    n_aug = 0

    for lab in label_names:
        folder = os.path.join(BASE_DIR, lab)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".npy"):
                continue
            npy_path = os.path.join(folder, fname)
            n_raw += 1

            seq, meta = _load_sequence_with_meta(npy_path)
            if seq is None:
                continue

            # re-muestreo a longitud fija
            seq_rs = _resample_linear(seq, SEQ_LENGTH)
            all_X.append(seq_rs)
            all_y.append(label_to_idx[lab])
            all_meta.append(meta)
            n_kept += 1

            # data augmentation (solo sobre la secuencia ya re-muestreada)
            if ENABLE_AUGMENT and AUG_PER_SAMPLE > 0:
                for _ in range(AUG_PER_SAMPLE):
                    aug_seq = _augment_sequence_jitter(seq_rs)
                    all_X.append(aug_seq)
                    all_y.append(label_to_idx[lab])
                    # copiamos metadatos y marcamos como augmentado
                    meta_aug = dict(meta)
                    meta_aug["augmented"] = True
                    all_meta.append(meta_aug)
                    n_aug += 1

    if not all_X:
        raise RuntimeError("No se cargó ninguna secuencia válida desde coordenadas_pose/")

    X = np.stack(all_X, axis=0).astype(np.float32)  # (N, T, F)
    y = np.asarray(all_y, dtype=np.int64)

    print(f"[INFO] Secuencias crudas encontradas: {n_raw}")
    print(f"[INFO] Secuencias válidas (originales): {n_kept}")
    if ENABLE_AUGMENT:
        print(f"[INFO] Secuencias aumentadas añadidas: {n_aug}")
    print(f"[INFO] Shape final X: {X.shape}  y: {y.shape}")

    # pequeño resumen por clase
    counts = np.bincount(y, minlength=len(label_names))
    for i, lab in enumerate(label_names):
        print(f"   - {lab}: {int(counts[i])} muestras")

    # Guardar labels.json en la carpeta base del dataset (útil para inspección rápida)
    labels_json_path = os.path.join(BASE_DIR, "labels.json")
    try:
        with open(labels_json_path, "w", encoding="utf-8") as f:
            json.dump({"labels": label_names}, f, ensure_ascii=False, indent=2)
        print(f"[INFO] labels.json guardado en: {labels_json_path}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar labels.json en {labels_json_path}: {e}")

    return X, y, label_names

def build_datasets():
    """
    Construye splits de entrenamiento y validación a partir de los datos cargados
    y los guarda opcionalmente en un .npz dentro de coordenadas_pose/.
    Devuelve (X_train, X_val, y_train, y_val, labels).
    """
    X, y, labels = load_data()
    if len(X) == 0:
        raise RuntimeError("No hay muestras en coordenadas_pose/")

    Xtr, Xva, ytr, yva = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print("[INFO] Shapes ->")
    print("  X_train:", Xtr.shape, " y_train:", ytr.shape)
    print("  X_val:  ", Xva.shape, " y_val:  ", yva.shape)

    # Guardar splits opcional
    out_path = os.path.abspath(os.path.join(BASE_DIR, "dataset_splits.npz"))
    try:
        np.savez_compressed(
            out_path,
            X_train=Xtr,
            X_val=Xva,
            y_train=ytr,
            y_val=yva,
            labels=np.asarray(labels)
        )
        print("[INFO] Splits guardados en:", out_path)
    except Exception as e:
        print(f"[WARN] No se pudieron guardar los splits en {out_path}: {e}")

    return Xtr, Xva, ytr, yva, labels

if __name__ == "__main__":
    # Permite ejecutar: python preprocess_pose.py
    build_datasets()
