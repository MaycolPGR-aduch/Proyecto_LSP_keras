# -*- coding: utf-8 -*-
"""
preprocess_pose.py — Preprocesamiento para coordenadas (L,126) relativas al cuerpo
- Filtra secuencias vacías, re-muestrea a SEQ_LENGTH, y devuelve X,y,labels.
- Guarda splits opcionales en coordenadas_pose/.
"""

import os, numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", "coordenadas_pose"))
FEATURES_T = 126
SEQ_LENGTH = 60
TEST_SIZE  = 0.2
RANDOM_SEED = 42

def _fix_1d_object_array(seq_obj):
    cleaned=[]
    for elm in seq_obj:
        cleaned.append(list(elm) if hasattr(elm,"__len__") else [])
    return np.array(cleaned, dtype=np.float32)

def _pad_trunc_time(seq):
    L = seq.shape[0]
    if L==SEQ_LENGTH: return seq.astype(np.float32, copy=False)
    if L<SEQ_LENGTH:
        pad = np.zeros((SEQ_LENGTH-L, FEATURES_T), dtype=np.float32)
        return np.vstack([seq, pad])
    return seq[:SEQ_LENGTH].astype(np.float32, copy=False)

def load_data():
    print(f"[preprocess_pose] Leyendo {BASE_DIR}")
    X,y=[],[]
    labels = sorted([d for d in os.listdir(BASE_DIR)
                     if os.path.isdir(os.path.join(BASE_DIR,d)) and not d.startswith("_")])
    label2idx = {lab:i for i,lab in enumerate(labels)}
    for lab in labels:
        folder = os.path.join(BASE_DIR, lab)
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(".npy"): continue
            p = os.path.join(folder, f)
            try:
                seq = np.load(p, allow_pickle=True)
            except Exception as e:
                print("[WARN] no se pudo cargar", p, e); continue
            if isinstance(seq, np.ndarray) and seq.ndim==1:
                seq = _fix_1d_object_array(seq)
            if not isinstance(seq, np.ndarray) or seq.ndim!=2 or seq.shape[1]!=FEATURES_T:
                print(f"[WARN] shape inválida {p}: {getattr(seq,'shape',None)}"); continue
            seq = _pad_trunc_time(seq)
            # filtro de calidad: al menos 80% frames no cero
            valid = np.sum(np.any(np.abs(seq)>0, axis=1))
            if valid < int(0.8*SEQ_LENGTH): 
                print(f"[WARN] poca señal {p} ({valid}/{SEQ_LENGTH})"); 
                continue
            X.append(seq); y.append(label2idx[lab])
    X=np.array(X, np.float32); y=np.array(y, np.int32)
    return X,y,labels

def build_datasets():
    X,y,labels = load_data()
    if len(X)==0: raise RuntimeError("No hay .npy en coordenadas_pose/")
    Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_SEED,stratify=y)
    print("Clases:", labels)
    print("Shapes ->", Xtr.shape, Xva.shape)
    # guardar splits opcional
    out = os.path.abspath(os.path.join(BASE_DIR,"dataset_splits.npz"))
    np.savez_compressed(out, X_train=Xtr, X_val=Xva, y_train=ytr, y_val=yva, labels=np.array(labels))
    print("Splits guardados en:", out)
    return Xtr,Xva,ytr,yva,labels

if __name__=="__main__":
    build_datasets()
