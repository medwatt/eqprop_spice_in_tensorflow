import tensorflow as tf
from .base import CircuitLayer


class CurrentLayer(CircuitLayer):
    def __init__(self, current_range=[1e-7, 1e-5], enabled=True, dtype=tf.float32):
        self.current_range = current_range
        self.enabled = enabled
        self.dtype = dtype

    def build(self, input_shape):
        # This layer is expected to be preceded by a dense layer.
        # It has the same shape as the preceding dense layer.
        self.output_shape = input_shape

        # Layer's current
        if self.enabled:
            self.current = tf.convert_to_tensor(
                tf.random.uniform(
                    self.output_shape,
                    minval=self.current_range[0],
                    maxval=self.current_range[1],
                    dtype=self.dtype,
                )
            )
        else:
            self.current = tf.zeros(self.output_shape, dtype=self.dtype)

    def initialize_voltage(self):
        # No independent voltage variable.
        self.voltage = None

    def forward_contribution():
        "Not Implemented"
        pass

    def backward_contribution(self):
        return -self.current, 0

    def jacobi_method(self, previous_layer, next_layer, tolerance=1e-6):
        return True

    def get_netlist(self, prev_nodes, curr_nodes, layer_index, sample_idx=0):
        if not self.enabled:
            return []
        lines = []
        forced_current = self.current[sample_idx].numpy().flatten()
        for idx, node in enumerate(prev_nodes):
            lines.append(f"I_in_{idx} {node} 0 DC {forced_current[idx]:.10e}")
        return lines
