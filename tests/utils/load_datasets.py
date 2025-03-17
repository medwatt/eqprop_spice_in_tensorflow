import numpy as np
import tensorflow as tf

# save mnist dataset to file <<<
def save_mnist_dataset(path="./"):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    np.save(f'{path}x_train.npy', x_train)
    np.save(f'{path}y_train.npy', y_train)
    np.save(f'{path}x_test.npy', x_test)
    np.save(f'{path}y_test.npy', y_test)
# >>>

# load mnist dataset from file <<<
def load_mnist_dataset(dataset_size=1000, x_scale=10.0, y_scale=1.0, normalize=True, reshape=False, stack=False, dtype=np.float32):
    dataset_directory = "/home/medwatt/jupyter/nn_models/mnist_full/"
    X_train = np.load(f"{dataset_directory}x_train.npy")[:dataset_size]
    Y_train = np.load(f"{dataset_directory}y_train.npy")[:dataset_size]
    X_test = np.load(f"{dataset_directory}x_test.npy")[:dataset_size]
    Y_test = np.load(f"{dataset_directory}y_test.npy")[:dataset_size]

    mean = 33.318421449829934
    std = 78.56748998339798
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = x_scale * X_train
    X_test = x_scale * X_test
    Y_train = y_scale * Y_train
    Y_test = y_scale * Y_test

    if reshape:
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        if stack:
            # Stack positive and negative as separate channels.
            X_train = np.stack([X_train, -X_train], axis=-1)
            X_test = np.stack([X_test, -X_test], axis=-1)
        else:
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

    elif stack:
        # Concatenate along feature dimension.
        X_train = np.concatenate([X_train, -X_train], axis=1)
        X_test = np.concatenate([X_test, -X_test], axis=1)

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    Y_train = Y_train.astype(dtype)
    Y_test = Y_test.astype(dtype)

    return (X_train, Y_train), (X_test, Y_test)
# >>>
