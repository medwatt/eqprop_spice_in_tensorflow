# imports <<<
import sys
import os

# Add the parent directory sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tensorflow as tf
import numpy as np

from src.layers.dense import DenseLayer
from src.layers.conv2d import Conv2DLayer
from src.layers.current import CurrentLayer
from src.layers.input_voltage import InputVoltageLayer
from src.voltage_analyzer import VoltageRangeAnalyzer
# >>>

tf.random.set_seed(42)
np.random.seed(42)

TF_DTYPE = tf.float64
NP_DTYPE = np.float64

# load dataset from file <<<
def load_dataset(dataset_size=1000, x_scale=10.0, y_scale=1.0, normalize=True, reshape=False, stack=False):
    dataset_directory = "/home/medwatt/jupyter/nn_models/mnist_full/"
    X_train = np.load(f"{dataset_directory}x_train.npy")[:dataset_size]
    Y_train = np.load(f"{dataset_directory}y_train.npy")[:dataset_size]
    X_test = np.load(f"{dataset_directory}x_test.npy")[:dataset_size]
    Y_test = np.load(f"{dataset_directory}y_test.npy")[:dataset_size]

    mean = 33.318421449829934
    std = 78.56748998339798
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = x_scale * X_train
    X_test = x_scale * X_test
    Y_train = y_scale * Y_train
    Y_test = y_scale * Y_test

    if reshape:
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        if stack:
            # Stack positive and negative as separate channels.
            X_train = np.stack([X_train, -X_train], axis=-1)
            X_test = np.stack([X_test, -X_test], axis=-1)
    elif stack:
        # Concatenate along feature dimension (as in the original code).
        X_train = np.concatenate([X_train, -X_train], axis=1)
        X_test = np.concatenate([X_test, -X_test], axis=1)

    X_train = X_train.astype(NP_DTYPE)
    X_test = X_test.astype(NP_DTYPE)
    Y_train = Y_train.astype(NP_DTYPE)
    Y_test = Y_test.astype(NP_DTYPE)

    return (X_train, Y_train), (X_test, Y_test)
# >>>

if __name__ == "__main__":

    batch_size = 100

    diode_params = {
        "vth_down": 1.16,
        "vth_up": 1.05,
        "ron": 1,
        "roff": 1e20
    }


    # x_dummy = np.random.rand(1, 28, 28, 1).astype(NP_DTYPE)
    (X_train, Y_train), (X_test, Y_test) = load_dataset(dataset_size=5000, reshape=True, stack=True)

    model = [
        InputVoltageLayer(shape=(batch_size, 28, 28, 2), dtype=TF_DTYPE),
        Conv2DLayer(10, kernel_size=(5, 5), strides=5, padding="valid", g_range=[1e-7, 1e-5], lr=1e-5, diodes_enabled=True, diode_params=diode_params, dtype=TF_DTYPE),
        DenseLayer(20, g_range=[1e-7, 1e-5], dtype=TF_DTYPE),
        CurrentLayer(current_range=[1e-7, 1e-5], enabled=False, dtype=TF_DTYPE),
    ]

    analyzer = VoltageRangeAnalyzer(model)
    analyzer.analyze(X_train, batch_size=batch_size, max_iteration=1000, tolerance=1e-6)
    analyzer.report()
