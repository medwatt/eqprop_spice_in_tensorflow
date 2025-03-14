import tensorflow as tf
import numpy as np
from .base import CircuitLayer
from .input_voltage import InputVoltageLayer
from .diode import Diode

# TODO: This code doesn't return correct results for batch sizes greater than 1

class Conv2DLayer(CircuitLayer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        g_range=[1e-7, 1e-5],
        diodes_enabled=False,
        diode_params=None,
        dtype=tf.float64,
        trainable=False,      # New parameter
        lr=1e-7,              # New parameter
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = [1, strides, strides, 1]
        self.padding = padding.upper()
        self.g_range = g_range
        self.dtype = dtype
        self.diodes_enabled = diodes_enabled
        self.trainable = trainable
        self.lr = lr
        if self.diodes_enabled:
            self.diode = Diode(**(diode_params if diode_params is not None else {}), dtype=dtype)
            self.diode_active = False

    def build(self, input_shape):
        self.input_shape = input_shape  # (batch_size, H, W, channels)
        self.batch_size = input_shape[0]
        in_channels = input_shape[-1]
        kh, kw = self.kernel_size
        self.kernel = tf.Variable(
            tf.random.uniform(
                (kh, kw, in_channels, self.filters),
                minval=self.g_range[0],
                maxval=self.g_range[1],
                dtype=self.dtype,
            ),
            trainable=False,
        )
        if self.trainable:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        if self.padding == "SAME":
            out_height = int(np.ceil(input_shape[1] / self.strides[1]))
            out_width = int(np.ceil(input_shape[2] / self.strides[2]))
        else:  # VALID
            out_height = int(np.ceil((input_shape[1] - kh + 1) / self.strides[1]))
            out_width = int(np.ceil((input_shape[2] - kw + 1) / self.strides[2]))
        self.output_shape = (self.batch_size, out_height, out_width, self.filters)

    def initialize_voltage(self):
        self.voltage = tf.Variable(tf.zeros(self.output_shape, dtype=self.dtype))

    @staticmethod
    def forward_contribution(voltage, kernel, strides, padding):
        numer = tf.nn.conv2d(input=voltage, filters=kernel, strides=strides, padding=padding)
        ones = tf.ones_like(voltage)
        denom = tf.nn.conv2d(input=ones, filters=kernel, strides=strides, padding=padding)
        return numer, denom

    @staticmethod
    def backward_contribution(voltage, kernel, strides, padding, output_shape):
        numer = tf.nn.conv2d_transpose(voltage, kernel, output_shape=output_shape, strides=strides, padding=padding)
        ones = tf.ones_like(voltage)
        denom = tf.nn.conv2d_transpose(ones, kernel, output_shape=output_shape, strides=strides, padding=padding)
        return numer, denom

    def jacobi_method(self, previous_layer, next_layer, tolerance=1e-6):
        # Make copy of old voltages to check with the updated voltages for convergence.
        voltage_copy = tf.identity(self.voltage)

        # Contribution from previous layer.
        if isinstance(previous_layer, (InputVoltageLayer, Conv2DLayer)):
            numer, denom = self.forward_contribution(previous_layer.voltage, self.kernel, self.strides, self.padding)
        else:
            numer, denom = previous_layer.forward_contribution(previous_layer.voltage, self.kernel)

        # Contribution from next layer.
        if next_layer:
            if isinstance(next_layer, Conv2DLayer):
                numer_r, denom_r = next_layer.backward_contribution(
                    next_layer.voltage,
                    next_layer.kernel,
                    next_layer.strides,
                    next_layer.padding,
                    tf.shape(self.voltage),
                )
            else:
                numer_r, denom_r = next_layer.backward_contribution(next_layer.voltage, next_layer.weights)
                numer_r = tf.reshape(numer_r, tf.shape(self.voltage))
                denom_r = tf.reshape(denom_r, (1, *self.output_shape[1:]))

            numer += numer_r
            denom += denom_r

        # Contribution from the diodes.
        if self.diodes_enabled:
            G_down, G_up = self.diode.linearize(self.voltage, self.diode_active)
            numer = numer + G_down * self.diode.vth_down + G_up * (-self.diode.vth_up)
            denom = denom + G_down + G_up

        # Update the voltages
        self.voltage = super().jacobi_update_step(numer, denom)

        # Compare old voltages to updated voltages
        return super().has_converged(self.voltage, voltage_copy, tolerance)

    def get_netlist(self, prev_nodes, curr_nodes, layer_index, **kwargs):
        lines = []
        kernel = self.kernel.numpy()
        kh, kw, in_channels, filters = kernel.shape
        stride_h = self.strides[1]
        stride_w = self.strides[2]
        pad_h = kh // 2 if self.padding == "SAME" else 0
        pad_w = kw // 2 if self.padding == "SAME" else 0
        H_in, W_in, _ = self.input_shape[1:]
        H_out, W_out, _ = self.output_shape[1:]
        for oh in range(H_out):
            for ow in range(W_out):
                for f in range(filters):
                    out_idx = (oh * W_out + ow) * filters + f
                    lines.append("*")
                    for i_k in range(kh):
                        for j_k in range(kw):
                            for c in range(in_channels):
                                in_h = oh * stride_h + i_k - pad_h
                                in_w = ow * stride_w + j_k - pad_w
                                if in_h < 0 or in_h >= H_in or in_w < 0 or in_w >= W_in:
                                    continue
                                in_idx = (in_h * W_in + in_w) * in_channels + c
                                G = kernel[i_k, j_k, c, f]
                                if G == 0:
                                    continue
                                R_val = 1.0 / G
                                lines.append(
                                    f"R_Conv_{layer_index}_{oh}_{ow}_{f}_{i_k}_{j_k}_{c} "
                                    f"{prev_nodes[in_idx]} {curr_nodes[out_idx]} {R_val:.10e}"
                                )
        if self.diodes_enabled:
            lines.extend(self.diode.get_netlist(curr_nodes, layer_index))
        return lines

    def update_weights(self, free_prev, free_curr, nudge_prev, nudge_curr, beta):
        if self.trainable:
            print("Not implemented yet!")
            exit()


