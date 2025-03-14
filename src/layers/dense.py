import tensorflow as tf
import numpy as np

from .base import CircuitLayer
from .current import CurrentLayer
from .conv2d import Conv2DLayer
from .input_voltage import InputVoltageLayer
from .diode import Diode


class DenseLayer(CircuitLayer):
    def __init__(
        self,
        units,
        g_range=[1e-7, 1e-5],
        trainable=False,
        lr=1e-7,
        diodes_enabled=False,
        diode_params=None,
        dtype=tf.float64,
    ):
        self.units = units
        self.g_range = g_range
        self.dtype = dtype
        self.trainable = trainable
        self.lr = lr
        self.diodes_enabled = diodes_enabled
        if self.diodes_enabled:
            self.diode = Diode(**(diode_params if diode_params is not None else {}), dtype=dtype)
            self.diode_active = False

    def build(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.input_dim = int(np.prod(input_shape[1:]))  # ignore batch_size
        self.output_shape = (self.batch_size, self.units)
        self.weights = tf.Variable(
            tf.random.uniform(
                (self.input_dim, self.units),
                minval=self.g_range[0],
                maxval=self.g_range[1],
                dtype=self.dtype,
            ),
            trainable=False,
        )
        if self.trainable:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def initialize_voltage(self):
        self.voltage = tf.Variable(tf.zeros(self.output_shape, dtype=self.dtype))

    @staticmethod
    def forward_contribution(v, g):
        numer = tf.linalg.matvec(g, v, transpose_a=True)
        denom = tf.reduce_sum(g, axis=0)
        return numer, denom

    @staticmethod
    def backward_contribution(v, g):
        numer = tf.linalg.matvec(g, v)
        denom = tf.reduce_sum(g, axis=1)
        return numer, denom

    def jacobi_method(self, previous_layer, next_layer, tolerance=1e-6):
        # Make copy of old voltages to check with the updated voltages for convergence.
        voltage_copy = tf.identity(self.voltage)

        # Contribution from previous layer.
        if isinstance(previous_layer, (InputVoltageLayer, Conv2DLayer)):
            v_flat = tf.reshape(previous_layer.voltage, [tf.shape(self.voltage)[0], -1])
            numer, denom = self.forward_contribution(v_flat, self.weights)
        else:
            numer, denom = previous_layer.forward_contribution(previous_layer.voltage, self.weights)

        # Contribution from next layer.
        if next_layer:
            if not isinstance(next_layer, CurrentLayer):
                numer_r, denom_r = next_layer.backward_contribution(next_layer.voltage, next_layer.weights)
            else:
                numer_r, denom_r = next_layer.backward_contribution()
            numer += numer_r
            denom += denom_r

        # Contribution from the diodes.
        if self.diodes_enabled:
            G_down, G_up = self.diode.linearize(self.voltage, self.diode_active)
            numer += G_down * self.diode.vth_down + G_up * (-self.diode.vth_up)
            denom += G_down + G_up

        # Update the voltages
        self.voltage = super().jacobi_update_step(numer, denom)

        # Compare old voltages to updated voltages
        return super().has_converged(self.voltage, voltage_copy, tolerance)

    def get_netlist(self, prev_nodes, curr_nodes, layer_index, **kwargs):
        lines = []
        weights = self.weights.numpy()
        n_prev, n_curr = weights.shape
        for j in range(n_curr):
            lines.append("*")
            for k in range(n_prev):
                G = weights[k, j]
                if G == 0:
                    continue
                R_val = 1.0 / G
                lines.append(
                    f"R_Dense_{layer_index}_{k}_{j} {prev_nodes[k]} {curr_nodes[j]} {R_val:.10e}"
                )
        if self.diodes_enabled:
            lines.extend(self.diode.get_netlist(curr_nodes, layer_index))
        return lines

    def update_weights(self, free_prev, free_curr, nudge_prev, nudge_curr, beta):
        if self.trainable:
            # Flatten input if from a conv layer.
            if len(free_prev.shape) > 2:
                batch_size = tf.shape(free_prev)[0]
                free_prev = tf.reshape(free_prev, [batch_size, -1])
                nudge_prev = tf.reshape(nudge_prev, [batch_size, -1])
            free_diff = tf.expand_dims(free_curr, 1) - tf.expand_dims(free_prev, 2)
            nudge_diff = tf.expand_dims(nudge_curr, 1) - tf.expand_dims(nudge_prev, 2)
            free_sq = tf.square(free_diff)
            nudge_sq = tf.square(nudge_diff)
            grad_estimate = tf.reduce_mean(nudge_sq - free_sq, axis=0) / beta
            self.optimizer.apply_gradients([(grad_estimate, self.weights)])
            self.weights.assign(tf.clip_by_value(self.weights, 1e-7, 1e-5))
