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
from src.layers.current import CurrentLayer
from src.layers.input_voltage import InputVoltageLayer
from utils.load_datasets import load_mnist_dataset
# >>>

tf.random.set_seed(42)
np.random.seed(42)

TF_DTYPE = tf.float64
NP_DTYPE = np.float64

if __name__ == "__main__":

    diode_params = {
        "vth_down": 0.2,
        "vth_up": 0.2,
        "ron": 1,
        "roff": 1e20
    }

    diode_params2 = {
        "vth_down": 0.05,
        "vth_up": 0.05,
        "ron": 1,
        "roff": 1e20
    }

    # x_dummy = tf.random.uniform((1, 28*28*2), minval=-2, maxval=2, dtype=TF_DTYPE)
    (X_train, Y_train), (X_test, Y_test) = load_mnist_dataset(dataset_size=1000, stack=True, dtype=NP_DTYPE)
    x_dummy = X_train[[100, 200, 1]]
    # x_dummy = X_train[1].reshape(1, -1)

    model = [
        InputVoltageLayer(shape=(3, 28*28*2), dtype=TF_DTYPE),
        DenseLayer(50, g_range=[1e-7, 1e-5], diodes_enabled=True, diode_params=diode_params, dtype=TF_DTYPE),
        DenseLayer(30, g_range=[1e-7, 1e-5], diodes_enabled=True, diode_params=diode_params2, dtype=TF_DTYPE),
        DenseLayer(20, g_range=[1e-7, 1e-5], dtype=TF_DTYPE),
        CurrentLayer(current_range=[1e-7, 1e-5], dtype=TF_DTYPE),
    ]
    solver = DCSolver(model, verbose=True)
    voltages = solver.solve_dc(x_dummy, max_iteration=1000, tolerance=1e-12)

    solver.write_netlist()
    print(voltages[-2])
