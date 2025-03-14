import tensorflow as tf

class Diode:
    def __init__(self, vth_down=0.7, vth_up=0.7, ron=1e-3, roff=1e20, dtype=tf.float32):
        self.vth_down = tf.constant(vth_down, dtype=dtype)
        self.vth_up = tf.constant(vth_up, dtype=dtype)
        self.ron = tf.constant(ron, dtype=dtype)
        self.roff = tf.constant(roff, dtype=dtype)
        self.gon = 1 / tf.constant(ron, dtype=dtype)
        self.goff = 1 / tf.constant(roff, dtype=dtype)

    def linearize(self, voltage, active):
        if active:
            G_down = tf.where(voltage > self.vth_down, self.gon, self.goff)
            G_up   = tf.where(voltage < -self.vth_up, self.gon, self.goff)
        else:
            G_down = tf.ones_like(voltage) * self.goff
            G_up   = tf.ones_like(voltage) * self.goff
        return G_down, G_up

    def get_netlist(self, nodes, layer_index):
        lines = []
        for j, node in enumerate(nodes):
            # Diode pointing down
            x_expr = f"(V({node}) - {self.vth_down})"
            on_expr = f"({x_expr}/{self.ron})"
            off_expr = f"({x_expr}/{self.roff})"
            current_expr = f"{{ max({off_expr}, {on_expr}) }}"
            lines.append(f"Bd{layer_index}d_{j} {node} 0 I={current_expr}")
            # Diode pointing up
            x_expr = f"(V({node}) + {self.vth_up})"
            on_expr = f"({x_expr}/{self.ron})"
            off_expr = f"({x_expr}/{self.roff})"
            current_expr = f"{{ min({off_expr}, {on_expr}) }}"
            lines.append(f"Bd{layer_index}u_{j} {node} 0 I={current_expr}")
        return lines

