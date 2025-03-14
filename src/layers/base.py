import tensorflow as tf


class CircuitLayer:
    def build(self, input_shape):
        raise NotImplementedError

    def initialize_voltage(self, batch_size, x_batch=None):
        raise NotImplementedError

    @staticmethod
    def forward_contribution():
        pass

    @staticmethod
    def backward_contribution():
        pass

    @staticmethod
    def has_converged(new_voltage, old_voltage, tolerance):
        return tf.norm(new_voltage - old_voltage) < tolerance

    @staticmethod
    def jacobi_update_step(numer, denom):
        return tf.where(denom != 0, numer/denom, tf.zeros_like(numer))

    def jacobi_method(self, previous_layer, next_layer, tolerance):
        raise NotImplementedError

    def get_netlist(self, prev_nodes, curr_nodes, layer_index, **kwargs):
        return []
