import os
import numpy as np
from sklearn.model_selection import train_test_split

# ------------- Configuración -------------
DATA_DIR   = os.path.join(os.getcwd(), "..", "data")
SEQ_LENGTH = 30     # Debe coincidir con capture.py
FEATURES   = 126    # 21 landmarks × 3 coords × 2 manos
TEST_SIZE  = 0.2    # 20% validación
RANDOM_SEED = 42

# ------------- Carga y preprocesamiento -------------
def load_data():
    X, y = [], []
    # Lee todas las subcarpetas (cada una es una clase o gesto)
    all_labels = sorted([d for d in os.listdir(DATA_DIR)
                         if os.path.isdir(os.path.join(DATA_DIR, d))])
    label2idx = {lab: i for i, lab in enumerate(all_labels)}

    for lab in all_labels:
        folder = os.path.join(DATA_DIR, lab)
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".npy"):
                continue

            # 1) Carga del archivo .npy
            seq = np.load(os.path.join(folder, filename), allow_pickle=True)

            # 2) Si vino como array 1D de objetos, reconstruirlo a (n_frames, FEATURES)
            if isinstance(seq, np.ndarray) and seq.ndim == 1:
                cleaned = []
                for elm in seq:
                    vals = list(elm) if hasattr(elm, "__len__") else []
                    # pad o trunca a FEATURES
                    if len(vals) < FEATURES:
                        vals = vals + [0.0] * (FEATURES - len(vals))
                    else:
                        vals = vals[:FEATURES]
                    cleaned.append(vals)
                seq = np.array(cleaned, dtype=np.float32)

            # 3) Pad o trunca la dimensión temporal a SEQ_LENGTH
            if seq.shape[0] < SEQ_LENGTH:
                padding = np.zeros((SEQ_LENGTH - seq.shape[0], FEATURES),
                                   dtype=np.float32)
                seq = np.vstack((seq, padding))
            else:
                seq = seq[:SEQ_LENGTH]

            # 4) Añade a X, y
            X.append(seq)                   # cada seq es (SEQ_LENGTH, FEATURES)
            y.append(label2idx[lab])

    X = np.array(X, dtype=np.float32)    # forma: (n_muestras, SEQ_LENGTH, FEATURES)
    y = np.array(y, dtype=np.int32)      # forma: (n_muestras,)
    return X, y, all_labels


# ------------- División en train/val -------------
def build_datasets():
    X, y, labels = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Classes: {labels}")
    print(f"Train samples: {X_train.shape[0]}  Val samples: {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val, labels

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, labels = build_datasets()
    # Opcional: guarda como .npz para cargas rápidas
    np.savez_compressed("../data/dataset_splits.npz",
                        X_train=X_train, X_val=X_val,
                        y_train=y_train, y_val=y_val,
                        labels=np.array(labels))
    print("Preprocesamiento completado y splits guardados en data/dataset_splits.npz")
