"""Model definitions and simple physics baselines."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seed(seed: int = 42) -> None:
    keras.utils.set_random_seed(seed)


def thesis_lstm(window: int, n_features: int) -> keras.Model:
    """The exact architecture from the thesis (2-layer LSTM, 118,626 params)."""
    model = keras.Sequential(
        [
            layers.Input((window, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(2),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def displacement_lstm(window: int, n_features: int) -> keras.Model:
    """Same LSTM family, but it predicts the next *displacement* in meters
    instead of an absolute latitude/longitude, so one model works for every
    vessel anywhere in the area."""
    model = keras.Sequential(
        [
            layers.Input((window, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(2),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber())
    return model


# --- Baselines --------------------------------------------------------------
# All take the input window of past positions with shape (n, window, 2) and
# return the predicted next position with shape (n, 2), in the same units.


def persistence(window_xy: np.ndarray) -> np.ndarray:
    """The ship stays where it is."""
    return window_xy[:, -1, :]


def constant_velocity(window_xy: np.ndarray) -> np.ndarray:
    """The ship repeats its last step."""
    return window_xy[:, -1, :] + (window_xy[:, -1, :] - window_xy[:, -2, :])


def gpu_summary() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    return f"TensorFlow {tf.__version__}, GPUs: {[g.name for g in gpus] or 'none'}"
