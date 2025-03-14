import numpy as np
from .dc_solver import DCSolver


class VoltageRangeAnalyzer:
    def __init__(self, model):
        self.model = model
        self.stats = {}

    def analyze(
        self,
        X,
        batch_size=100,
        max_iteration=1000,
        tolerance=1e-6,
        diode_iterations=1,
        verbose=False,
    ):
        solver = DCSolver(self.model, verbose=verbose)
        collected = {}  # key: layer index, value: list of 1D numpy arrays

        num_samples = X.shape[0]
        for i in range(0, num_samples, batch_size):

            x_batch = X[i : i + batch_size]

            voltages = solver.solve_dc(
                x_batch,
                max_iteration=max_iteration,
                tolerance=tolerance,
                diode_iterations=diode_iterations,
            )

            for layer_idx, voltage in enumerate(voltages):
                # Skip layers with no voltage output (e.g. CurrentLayer)
                if voltage is None:
                    continue
                try:
                    v_np = voltage.numpy()
                except AttributeError:
                    v_np = np.array(voltage)

                # Assume first dimension is the batch dimension.
                # Flatten each sample's voltage and store them.
                v_flat = v_np.reshape(v_np.shape[0], -1)
                for sample_voltage in v_flat:
                    collected.setdefault(layer_idx, []).append(sample_voltage)

        self.stats = {}
        for layer_idx, v_list in collected.items():
            all_v = np.concatenate(v_list)

            # Overall statistics
            stat = {
                "num_points": all_v.size,
                "mean": np.mean(all_v),
                "std": np.std(all_v),
            }

            # Separate positive and negative values.
            pos = all_v[all_v > 0]
            neg = all_v[all_v < 0]

            if pos.size > 0:
                stat.update(
                    {
                        "pos_count": pos.size,
                        "pos_mean": np.mean(pos),
                        "pos_std": np.std(pos),
                        "pos_min": np.min(pos),
                        "pos_25th": np.percentile(pos, 25),
                        "pos_median": np.median(pos),
                        "pos_75th": np.percentile(pos, 75),
                        "pos_90th": np.percentile(pos, 90),
                        "pos_max": np.max(pos),
                    }
                )
            else:
                stat.update({"pos_count": 0})

            if neg.size > 0:
                stat.update(
                    {
                        "neg_count": neg.size,
                        "neg_mean": np.mean(neg),
                        "neg_std": np.std(neg),
                        "neg_min": np.min(neg),
                        "neg_25th": np.percentile(neg, 25),
                        "neg_median": np.median(neg),
                        "neg_75th": np.percentile(neg, 75),
                        "neg_90th": np.percentile(neg, 10),
                        "neg_max": np.max(neg),
                    }
                )
            else:
                stat.update({"neg_count": 0})

            self.stats[layer_idx] = stat

    def report(self):
        if not self.stats:
            print("No statistics available. Run analyze() first.")
            return

        for layer_idx in sorted(self.stats.keys()):
            layer_name = self.model[layer_idx].__class__.__name__
            stat = self.stats[layer_idx]

            print(f"{layer_name} (Layer {layer_idx}) statistics:")
            print(f"  Total points: {stat['num_points']}")
            print(f"  Overall Mean: {stat['mean']:.6g}")
            print(f"  Overall Std: {stat['std']:.6g}")

            if stat.get("pos_count", 0) > 0:
                print("  Positive values:")
                print(f"    Count: {stat['pos_count']}")
                print(f"    Mean: {stat['pos_mean']:.6g}")
                print(f"    Std: {stat['pos_std']:.6g}")
                print(f"    Min: {stat['pos_min']:.6g}")
                print(f"    25th Percentile: {stat['pos_25th']:.6g}")
                print(f"    Median: {stat['pos_median']:.6g}")
                print(f"    75th Percentile: {stat['pos_75th']:.6g}")
                print(f"    90th Percentile: {stat['pos_90th']:.6g}")
                print(f"    Max: {stat['pos_max']:.6g}")
            else:
                print("  No positive values.")

            if stat.get("neg_count", 0) > 0:
                print("  Negative values:")
                print(f"    Count: {stat['neg_count']}")
                print(f"    Mean: {stat['neg_mean']:.6g}")
                print(f"    Std: {stat['neg_std']:.6g}")
                print(f"    Min: {stat['neg_min']:.6g}")
                print(f"    25th Percentile: {stat['neg_25th']:.6g}")
                print(f"    Median: {stat['neg_median']:.6g}")
                print(f"    75th Percentile: {stat['neg_75th']:.6g}")
                print(f"    90th Percentile: {stat['neg_90th']:.6g}")
                print(f"    Max: {stat['neg_max']:.6g}")

            else:
                print("  No negative values.")

            print("")
