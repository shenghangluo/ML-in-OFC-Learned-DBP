import numpy as np

class DatasetMNIST(object):
    def __init__(self, val_size):
        self._val_size = val_size

    def load_data(self):
        """
        :return:
        """

        y_train_real = np.reshape(np.ones(10, dtype=float), (1, 10))
        y_train_image = y_train_real
        x_train_real = np.reshape(np.random.normal(0, 1, 10), (1, 10))
        x_train_image = np.reshape(np.random.normal(0, 1, 10), (1, 10))

        x_test_real = x_train_real
        np.random.shuffle(x_test_real)
        x_valid_real = x_train_real
        np.random.shuffle(x_valid_real)

        x_test_image = x_train_image
        np.random.shuffle(x_test_image)
        x_valid_image = x_train_image
        np.random.shuffle(x_valid_image)

        y_test_real = y_train_real
        np.random.shuffle(y_test_real)
        y_valid_real = y_train_real
        np.random.shuffle(y_valid_real)

        y_test_image = y_train_image
        np.random.shuffle(y_test_image)
        y_valid_image = y_train_image
        np.random.shuffle(y_valid_image)

        return {
                'X_train_real': x_train_real,
                'X_train_image': x_train_image,
                'y_train_real': y_train_real,
                'y_train_image': y_train_image,
                'X_test_real': x_test_real,
                'X_test_image': x_test_image,
                'y_test_real': y_test_real,
                'y_test_image': y_test_image,
                'X_valid_real': x_valid_real,
                'X_valid_image': x_valid_image,
                'y_valid_real': y_valid_real,
                'y_valid_image': y_valid_image
                }

    @staticmethod
    def shuffle_batch(X_real, y_real, X_image, y_image, batch_size):
        rnd_idx = np.random.permutation(len(X_real))
        n_batches = len(X_real) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch_real, y_batch_real = X_real[batch_idx], y_real[batch_idx]
            X_batch_image, y_batch_image = X_image, y_image
            yield X_batch_real, y_batch_real, X_batch_image, y_batch_image
