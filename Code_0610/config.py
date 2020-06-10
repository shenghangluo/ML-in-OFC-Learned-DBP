import tensorflow as tf


def create_params():
    n_epochs = 15
    tr_batch_size = 1
    optimizer_params = {'lr': 1e-3}

    return {'n_epochs': n_epochs,
            'tr_batch_size': tr_batch_size,
            'optimizer_params': optimizer_params}
