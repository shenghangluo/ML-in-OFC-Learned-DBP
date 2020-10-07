import tensorflow as tf


def create_params():
    n_epochs = 200
    tr_batch_size = 30
    optimizer_params = {'lr': 1e-2}

    return {'n_epochs': n_epochs,
            'tr_batch_size': tr_batch_size,
            'optimizer_params': optimizer_params}
