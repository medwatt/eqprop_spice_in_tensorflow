import tensorflow as tf
from .base import CircuitLayer


class InputVoltageLayer(CircuitLayer):
    def __init__(self, shape, dtype=tf.float64):
        self.shape = shape
        self.dtype = dtype

    def build(self, input_shape=None):
        self.output_shape = self.shape

    def initialize_voltage(self, x_batch=None):
        if x_batch is not None:
            self.voltage = tf.convert_to_tensor(x_batch, dtype=self.dtype)
        else:
            self.voltage = tf.random.uniform(self.shape, minval=-10.0, maxval=10.0, dtype=self.dtype)

    @staticmethod
    def forward_contribution():
        pass

    @staticmethod
    def backward_contribution():
        pass

    def jacobi_method(self, previous_layer, next_layer, tolerance):
        return True

    def get_netlist(self, nodes, x_batch=None, sample_idx=0):
        lines = []

        forced_voltage = self.voltage[sample_idx].numpy().flatten()
        for idx, node in enumerate(nodes):
            lines.append(f"V_in_{idx} {node} 0 DC {forced_voltage[idx]:.6e}")
        return lines
