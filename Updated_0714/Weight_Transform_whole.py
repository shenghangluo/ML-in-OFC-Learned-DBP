import tensorflow as tf
import keras.backend as K


def Weight_Transform_whole(w_vector, n):
    weight = tf.reshape(w_vector, [1, n])
    out = weight
    for ii in range(n - 1):
        A = tf.roll(weight, shift=ii + 1, axis=1)
        out = tf.concat([out, A], axis=0)

    return out