# imports <<<
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tensorflow as tf
import numpy as np

from src.dc_solver import DCSolver
from src.layers.dense import DenseLayer
from src.layers.conv2d import Conv2DLayer
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
    stack = True

    diode_params = {
        "vth_down": 1.0,
        "vth_up": 1.0,
        "ron": 1,
        "roff": 1e20
    }

    (X_train, Y_train), (X_test, Y_test) = load_mnist_dataset(dataset_size=1000, reshape=True, stack=stack, dtype=NP_DTYPE)
    dataset = create_dataset(X_train, Y_train, batch_size=batch_size)
    n_channel = 2 if stack else 1

    model = [
        InputVoltageLayer(shape=(batch_size, 28, 28, n_channel), dtype=TF_DTYPE),
        Conv2DLayer(16, kernel_size=(3, 3), strides=2, padding="valid", g_range=[1e-7, 1e-5], trainable=True, lr=1e-6, diodes_enabled=False, diode_params=diode_params, dtype=TF_DTYPE),
        DenseLayer(40, g_range=[1e-7, 1e-5], trainable=True, diodes_enabled=True, diode_params=diode_params, lr=1e-6, dtype=TF_DTYPE),
        DenseLayer(20, g_range=[1e-7, 1e-5], trainable=True, lr=1e-6, dtype=TF_DTYPE),
        CurrentLayer(current_range=[1e-7, 1e-5], enabled=False, dtype=TF_DTYPE),
    ]

    ep = EqProp(model, beta=1e-7)
    ep.train(dataset, epochs=20, free_iter=1000, nudge_iter=500, tolerance=1e-8)

