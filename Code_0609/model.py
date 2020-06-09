import tensorflow as tf


class Model(object):
    def __init__(self):
        self.X_real = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="X_real")
        self.X_image = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="X_image")

        '''one layer'''
        # Linear
        # name_real = 'weight_real_1'
        # name_image = 'weight_image_1'
        # self.L1 = Linear_OP(X_real=self.X_real, X_image=self.X_image, name_real=name_real, name_image=name_image)
        # self.Si, self.Sr = self.L1.get_linear_out()

        self.rr = tf.layers.dense(inputs=self.X_real, units=10, activation=None, name='weight_real_1')
        self.ri = tf.layers.dense(inputs=self.X_real, units=10, activation=None, name='weight_image_1')

        self.ir = tf.layers.dense(inputs=self.X_image, units=10, activation=None, name='weight_real_1', reuse=True)
        self.ii = tf.layers.dense(inputs=self.X_image, units=10, activation=None, name='weight_image_1', reuse=True)

        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

        # Nonlinear
        self.alph = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='alph')
        self.S_power = tf.math.square(self.Sr) + tf.math.square(self.Si)
        self.S_power = tf.math.scalar_mul(self.alph, self.S_power)
        self.sin = tf.math.sin(self.S_power)
        self.cos = tf.math.cos(self.S_power)

        self.y_real = tf.math.add(tf.math.multiply(self.Sr, self.cos), tf.math.multiply(self.Si, self.sin))
        self.y_image = tf.math.subtract(tf.math.multiply(self.Si, self.cos), tf.math.multiply(self.Sr, self.sin))




    def get_reconstruction(self):
        return self.y_real, self.y_image

    def get_para(self):
        return self.alph



#
# class Linear_OP(object):
#     def __init__(self, X_real, X_image, name_real, name_image):
#         self.in_real = X_real
#         self.in_image = X_image
#
#         self.rr = tf.layers.dense(inputs=self.in_real, units=10, activation=None, name=name_real)
#         self.ri = tf.layers.dense(inputs=self.in_real, units=10, activation=None, name=name_image)
#
#         self.ir = tf.layers.dense(inputs=self.in_image, units=10, activation=None, name=name_real, reuse=True)
#         self.ii = tf.layers.dense(inputs=self.in_image, units=10, activation=None, name=name_image, reuse=True)
#
#         self.Si = tf.math.add(self.ri, self.ir)
#         self.Sr = tf.math.subtract(self.rr, self.ii)
#
#     def get_linear_out(self):
#         return self.Sr, self.Si
