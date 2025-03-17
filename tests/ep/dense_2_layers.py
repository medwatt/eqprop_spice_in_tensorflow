# imports <<<
import sys
import os

# Add the parent directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tensorflow as tf
import numpy as np

from src.dc_solver import DCSolver
from src.layers.dense import DenseLayer
from src.layers.current import CurrentLayer
from src.layers.input_voltage import InputVoltageLayer
from utils.load_mnist_dataset import load_mnist_dataset
from src.ep_alg import EqProp
# >>>

tf.random.set_seed(42)
np.random.seed(42)

TF_DTYPE = tf.float64
NP_DTYPE = np.float64

# create dataset batches <<<
def create_dataset(x, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset
# >>>

if __name__ == "__main__":

    batch_size = 100

    diode_params = {
        "vth_down": 0.23,
        "vth_up": 0.23,
        "ron": 1,
        "roff": 1e20
    }

    (X_train, Y_train), (X_test, Y_test) = load_mnist_dataset(dataset_size=1000, reshape=False, stack=True, dtype=NP_DTYPE)
    dataset = create_dataset(X_train, Y_train, batch_size=batch_size)

    model = [
        InputVoltageLayer(shape=(batch_size, 28*28*2), dtype=TF_DTYPE),
        DenseLayer(100, g_range=[1e-7, 1e-5], trainable=True, diodes_enabled=True, diode_params=diode_params, dtype=TF_DTYPE),
        DenseLayer(20, g_range=[1e-7, 1e-5], trainable=True, dtype=TF_DTYPE),
        CurrentLayer(current_range=[1e-7, 1e-5], dtype=TF_DTYPE),
    ]

    ep = EqProp(model, beta=1e-7)
    ep.train(dataset, epochs=20, free_iter=100, nudge_iter=50, tolerance=1e-6)
