import tensorflow as tf

def create_sequence_model(
    num_classes: int,
    seq_length: int = 60,   # Alineado con preprocess.py (60 frames)
    feat_dim: int = 126    # SOLO MANOS (21*3*2), alineado con pipeline
) -> tf.keras.Model:
    """
    Modelo de secuencias para landmarks (SOLO MANOS).
    - Entrada: (seq_length, feat_dim) p.ej. (60, 126)
    - Conv1D a lo largo del tiempo + pooling + GAP + Dense.
    """
    inp = tf.keras.Input(shape=(seq_length, feat_dim), name="input_sequence")

    # Bloque 1: Extracción de características temporales
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Bloque 2: Extracción de características más complejas
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Resumen temporal: Agrupa los patrones detectados
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Bloque de Clasificación (Cabeza)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x) # Regularización para evitar overfitting

    # Salida final
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = tf.keras.Model(inputs=inp, outputs=out, name="LSP_Sequence_Model")
    return model

if __name__ == "__main__":
    # Prueba rápida para verificar la arquitectura del modelo
    # Esto solo se ejecuta si corres: python model.py
    
    # Ejemplo: 5 clases, entrada (60, 126)
    m = create_sequence_model(num_classes=5)
    
    print("\n--- Resumen del Modelo (model.py) ---")
    m.summary()
    print("--------------------------------------\n")

