import tensorflow as tf

def create_sequence_model(
    num_classes: int,
    seq_length: int = 30,
    feat_dim: int = 126
) -> tf.keras.Model:
    """
    Crea un modelo de secuencias:
    - Entrada: (seq_length, feat_dim)
    - Capas Conv1D + pooling para extraer patrones temporales.
    - Capa densa final con softmax para num_classes.
    """
    inp = tf.keras.Input(shape=(seq_length, feat_dim), name="input_sequence")

    # Bloque 1: conv1D + pooling
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, activation="relu", padding="same"
    )(inp)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Bloque 2: conv1D + pooling
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Global pooling para colapsar dimensión temporal
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Capa intermedia
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Salida
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="LSP_Sequence_Model")
    return model

# Prueba sencilla de construcción
if __name__ == "__main__":
    # Suponiendo 5 clases (por ejemplo B, C, J, Y, Z)
    model = create_sequence_model(num_classes=5)
    model.summary()
