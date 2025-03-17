import numpy as np
from .layers import CurrentLayer

class DCSolver:
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose
        self.built = False

    def build(self, input_shape):
        shape = input_shape
        for layer in self.model:
            layer.build(shape)
            shape = layer.output_shape
        self.built = True

    def initialize_voltages(self, x_batch):
        self.model[0].initialize_voltage(x_batch)
        # First layer is assumed to be the input voltage layer.
        for layer in self.model[1:]:
            layer.initialize_voltage()

    def initialize_circuit_solver(self, x_batch):
        if not self.built:
            self.build(x_batch.shape)
        self.initialize_voltages(x_batch)

    def _solve_phase(self, max_iteration, tolerance):
        for n_iter in range(max_iteration):
            converged = True
            for idx, layer in enumerate(self.model[1:], start=1):
                previous_layer = self.model[idx - 1]
                next_layer = self.model[idx + 1] if idx < len(self.model) - 1 else None
                if not layer.jacobi_method(previous_layer, next_layer, tolerance):
                    converged = False
            if converged:
                if self.verbose:
                    print(f"Converged in {n_iter + 1} iterations.")
                break

    def solve_dc(self, x_batch, max_iteration=1000, diode_iterations=1, tolerance=1e-6):
        if x_batch is not None:
            self.initialize_circuit_solver(x_batch)

        # Crude way for checking if non-linearities are present.
        # TODO: There should be a cleaner way to do this.
        diode_layers = [layer for layer in self.model if getattr(layer, "diodes_enabled", False)]

        if diode_layers:
            # Initial pass: force diodes off.
            for layer in diode_layers:
                layer.diode_active = False
            self._solve_phase(max_iteration, tolerance)

            # Refinement iterations with diode model enabled.
            for _ in range(diode_iterations):
                for layer in diode_layers:
                    layer.diode_active = True
                self._solve_phase(max_iteration, tolerance)
        else:

            self._solve_phase(max_iteration, tolerance)

        return [layer.voltage for layer in self.model]

    def export_spice_netlist(self, x_batch=None, sample_idx=0):
        if not self.built:
            raise ValueError("Model not built. Run solve_dc or build the model first.")
        netlist = ["* SPICE netlist generated from TensorFlow model"]
        nodes_by_layer = []
        for idx, layer in enumerate(self.model):
            num_nodes = int(np.prod(layer.output_shape[1:]))
            nodes = [f"L{idx}_N{n}" for n in range(num_nodes)]
            nodes_by_layer.append(nodes)
        netlist.extend(self.model[0].get_netlist(nodes_by_layer[0], x_batch=x_batch, sample_idx=sample_idx))
        for i in range(1, len(self.model)):
            prev_nodes = nodes_by_layer[i - 1]
            curr_nodes = nodes_by_layer[i]
            netlist.extend(self.model[i].get_netlist(prev_nodes, curr_nodes, i, sample_idx=sample_idx))
        last_layer = self.model[-1]
        last_layer_index = len(self.model) - 1
        if isinstance(self.model[-1], CurrentLayer):
            last_layer_index -= 1
        output_voltages = [f"V(L{last_layer_index}_N{node})" for node in range(last_layer.output_shape[1])]
        control_block = [".control", "op", "print " + " ".join(output_voltages), ".endc"]
        netlist.extend(control_block)
        netlist.append(".end")
        return "\n".join(netlist)

    def write_netlist(self, sample_idx=0, filename="netlist.sp"):
        netlist_str = self.export_spice_netlist(sample_idx=sample_idx)
        with open(filename, "w") as f:
            f.write(netlist_str)
        print(f"Netlist written to {filename}")

