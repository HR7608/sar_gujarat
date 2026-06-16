import tensorflow as tf
from tensorflow.keras import layers, Model


def wind_direction_loss(y_true, y_pred):
    """
    Custom circular loss function for wind direction.
    L = 1 - cos²(θ_predicted - θ_target)

    Returns 0 for perfect prediction.
    Returns 0 for 180° error (handles aliasing naturally).
    Returns 1 for 90° error (worst case).
    """
    diff = y_pred - y_true
    return 1.0 - tf.square(tf.cos(diff))


def residual_block(x, filters=64):
    """
    One residual block (RNB) from Zanchetta & Zecchetto (2020).

    Structure:
        input x
            ↓
        Conv(3×3) → BN → ReLU
            ↓
        Conv(3×3) → BN
            ↓
        Add shortcut (x)
            ↓
        ReLU
            ↓
        MaxPool(2×2) → BN
            ↓
        output
    """
    shortcut = x

    # First conv layer
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second conv layer
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Shortcut projection if channel dimensions differ
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add shortcut connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    # MaxPool + BatchNorm
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    return x


def build_m64rn4():
    """
    M64RN4 ResNet architecture from Zanchetta & Zecchetto (2020).

    Input:  49×49×1 SAR patch (z-score normalised VV backscatter)
    Output: wind direction θ in radians (aliased, 180° ambiguity)

    Architecture:
        4 × RNB(64 channels)
        Flatten → 576
        Dense(512) → ReLU → BN
        Dense(128) → ReLU → BN
        Dense(32)  → ReLU → BN
        Dense(1)   → θ in radians

    Spatial progression:
        49×49 → 24×24 → 12×12 → 6×6 → 3×3 → flatten(576)

    Total parameters: ~630,529
    """
    inputs = tf.keras.Input(shape=(49, 49, 1), name="sar_patch")

    # 4 residual blocks — 64 channels each
    x = residual_block(inputs, filters=64)
    x = residual_block(x,      filters=64)
    x = residual_block(x,      filters=64)
    x = residual_block(x,      filters=64)

    # Flatten: 3×3×64 = 576
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(512)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(128)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # Output: single angle in radians
    outputs = layers.Dense(1, name="wind_direction_rad")(x)

    model = Model(inputs, outputs, name="M64RN4")
    return model


def load_trained_model(weights_path="best_model.keras"):
    """
    Load a trained M64RN4 model from saved weights.
    Used by api_service.py at startup.
    """
    from pathlib import Path
    if not Path(weights_path).exists():
        return None
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={"wind_direction_loss": wind_direction_loss}
    )
    return model


# ── Verify architecture when run directly ────────────────────────────────────
if __name__ == "__main__":
    print("Building M64RN4 ResNet...")
    model = build_m64rn4()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=wind_direction_loss,
    )
    model.summary()
    print(f"\nModel built successfully.")
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")