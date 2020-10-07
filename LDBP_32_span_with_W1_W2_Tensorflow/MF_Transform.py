import tensorflow as tf
import keras.backend as K


def MF_Transform(w_vector, k, n):
    w_sliced = tf.slice(w_vector, [0], [k])
    paddings = tf.constant([[0, k]])
    w_pad = tf.pad(w_vector, paddings, "CONSTANT")
    weight = w_pad
    paddings = tf.constant([[0, n - 2 * k - 1]])
    weight = tf.pad(weight, paddings, "CONSTANT")
    weight = tf.reshape(weight, [1, n])

    out = weight
    for ii in range(n-k-1):
        A = tf.roll(weight, shift=ii + 1, axis=1)
        out = tf.concat([out, A], axis=0)

    return out