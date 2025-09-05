import tensorflow as tf

def create_sequence_model(
    num_classes: int,
    seq_length: int = 60,   # ahora 60 frames por defecto (≈2s)
    feat_dim: int = 1530    # manos(126) + rostro(1404)
) -> tf.keras.Model:
    """
    Modelo de secuencias para landmarks:
    - Entrada: (seq_length, feat_dim) p.ej. (60, 1530)
    - Conv1D a lo largo del tiempo + pooling + GAP + Dense.
    """
    inp = tf.keras.Input(shape=(seq_length, feat_dim), name="input_sequence")

    # Bloque 1
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Bloque 2
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Resumen temporal
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Cabeza
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="LSP_Sequence_Model")
    return model

if __name__ == "__main__":
    # Ejemplo: 5 clases, (60,1530)
    m = create_sequence_model(num_classes=5)
    m.summary()

