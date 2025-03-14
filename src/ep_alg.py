import tensorflow as tf
import numpy as np

from .dc_solver import DCSolver
from .layers import CurrentLayer
from .layers import DenseLayer
from .layers import Conv2DLayer

class EqProp:

    def __init__(self, model, beta):
        self.model = model
        self.dc_solver = DCSolver(model)
        self.beta = beta

    def train(self, dataset, epochs, free_iter=1000, nudge_iter=500, tolerance=1e-6, verbose=True):

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            batches = 0

            for batch_idx, (x_batch, y_batch) in enumerate(dataset):

                # --- Free Phase ---
                # print(f"Batch {batch_idx+1}: Free Phase")

                # Set nudging currents to zero during free phase.
                for layer in self.model:
                    if isinstance(layer, CurrentLayer):
                        layer.enabled = False

                _ = self.dc_solver.solve_dc(x_batch=x_batch, max_iteration=free_iter, tolerance=tolerance)

                # Make a copy of free phase voltages.
                free_dict = {}
                for i, layer in enumerate(self.model):
                    if hasattr(layer, "voltage") and layer.voltage is not None:
                        free_dict[i] = tf.identity(layer.voltage)

                # Get output of last dense layer
                if isinstance(self.model[-1], CurrentLayer):
                    raw_output = self.model[-2].voltage
                else:
                    raw_output = self.model[-1].voltage

                # Compute gradient of loss with respect to free output.
                with tf.GradientTape() as tape:
                    tape.watch(raw_output)
                    # Compute prediction as the difference of each consecutive pair.
                    free_output = raw_output[:, 0::2] - raw_output[:, 1::2]
                    loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=free_output))
                grad_output = tape.gradient(loss_val, raw_output)

                preds = tf.argmax(free_output, axis=1)
                targets = tf.argmax(y_batch, axis=1)
                acc = tf.reduce_mean(tf.cast(tf.equal(preds, targets), tf.float32))
                total_loss += loss_val.numpy()
                total_acc += acc.numpy()
                # print("Batch accuracy", acc.numpy())

                # --- Nudge Phase ---
                # print("Nudging Phase")

                # Enable current layer and inject current = beta * grad_output.
                for layer in self.model:
                    if isinstance(layer, CurrentLayer):
                        layer.enabled = True
                        layer.current = self.beta * grad_output

                _ = self.dc_solver.solve_dc(x_batch=None, max_iteration=nudge_iter, tolerance=tolerance)

                # Save nudge-phase voltages.
                nudge_dict = {}
                for i, layer in enumerate(self.model):
                    if hasattr(layer, 'voltage') and layer.voltage is not None:
                        nudge_dict[i] = tf.identity(layer.voltage)

                # --- Weight Update for each learnable layer ---
                for i, layer in enumerate(self.model):
                    if isinstance(layer, (DenseLayer, Conv2DLayer)):
                        free_prev = free_dict[i-1]
                        free_curr = free_dict[i]
                        nudge_prev = nudge_dict[i-1]
                        nudge_curr = nudge_dict[i]
                        layer.update_weights(free_prev, free_curr, nudge_prev, nudge_curr, self.beta)


                batches += 1

            print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss/batches:.4f}, Accuracy = {total_acc/batches:.4f}")

