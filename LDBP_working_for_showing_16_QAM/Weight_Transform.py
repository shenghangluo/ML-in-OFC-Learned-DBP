import tensorflow as tf
import keras.backend as K


def Weight_Transform(w_vector, k, n, m):
    w_sliced = tf.slice(w_vector, [0], [k])
    paddings = tf.constant([[0, k]])
    w_pad = tf.pad(w_vector, paddings, "CONSTANT")
    # print("k", k)
    # print("w vector shape", w_vector.shape)
    # print("w_pad shape: ", w_pad.shape)
    paddings = tf.constant([[0, k + 1]])
    w_sliced_pad = tf.pad(w_sliced, paddings, "CONSTANT")
    w_sliced_pad = K.reverse(w_sliced_pad, axes=0)
    # print("w_sliced_pad shape:", w_sliced_pad.shape)
    weight = tf.math.add(w_sliced_pad, w_pad)

    paddings = tf.constant([[0, m - 2 * k - 1]])
    weight = tf.pad(weight, paddings, "CONSTANT")
    weight = tf.reshape(weight, [1, m])

    out = weight
    for ii in range(n - 1):
        A = tf.roll(weight, shift=ii + 1, axis=1)
        out = tf.concat([out, A], axis=0)

    return out