import tensorflow as tf
import numpy as np
from weight_init import W1_init_Real, W1_init_Imag, W2_init_Real, W2_init_Imag
N = 100
M = 50
Lsp = 100000
M_step = 2
delta = Lsp/M_step

class Model(object):
    def __init__(self):
        self.X_real = tf.compat.v1.placeholder(tf.float32, shape=(None, N), name="X_real")
        self.X_image = tf.compat.v1.placeholder(tf.float32, shape=(None, N), name="X_image")
        self.y_real = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="y_real")
        self.y_image = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="y_image")

        self.X_OP_real = self.X_real        # Since placeholder cannot assigned by another tensor
        self.X_OP_image = self.X_image

        self.layerlist = []
        for ii in range(1, 16):              # for loop ready for large network
            self.layerlist.append(Layer(X_real=self.X_OP_real, X_image=self.X_OP_image, layer_number=ii))
            self.out_real, self.out_image = self.layerlist[ii-1].get_layer_output()
            self.X_OP_real = self.out_real
            self.X_OP_image = self.out_image

        # MF
        self.rr = tf.contrib.layers.fully_connected(self.out_real, num_outputs=M, activation_fn=None, biases_initializer=None,
                                                    trainable=True, scope='MF_Real')
        self.ri = tf.contrib.layers.fully_connected(self.out_real, num_outputs=M, activation_fn=None, biases_initializer=None,
                                                    trainable=True, scope='MF_Image')
        self.ir = tf.contrib.layers.fully_connected(self.out_image, num_outputs=M, activation_fn=None, biases_initializer=None,
                                                    trainable=True, reuse=True, scope='MF_Real')
        self.ii = tf.contrib.layers.fully_connected(self.out_image, num_outputs=M, activation_fn=None, biases_initializer=None,
                                                    trainable=True, reuse=True, scope='MF_Image')
        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

        self.out_MF_real = self.Sr
        self.out_MF_image = self.Si

    def get_reconstruction(self):
        return self.out_MF_real, self.out_MF_image

    def get_middel_para(self):
        return self.layerlist[13].get_mid_para()

class Layer(object):
    def __init__(self, X_real, X_image, layer_number):
        self.X_real = X_real
        self.X_image = X_image

        # Linear W1
        self.Linear1 = Linear_W1(X_real=self.X_real, X_image=self.X_image, layer_number=layer_number)
        self.Sr, self.Si = self.Linear1.get_linear_out()

        # Nonlinear
        self.Nonlinear = NonLinear_OP(S_real=self.Sr, S_image=self.Si, layernumber=layer_number)
        self.y_real, self.y_image = self.Nonlinear.get_nonlinear_out()

        # Linear W2
        self.Linear2 = Linear_W2(X_real=self.y_real, X_image=self.y_image, layer_number=layer_number)
        self.out_real, self.out_image = self.Linear2.get_linear_out()

    def get_layer_output(self):
        return self.out_real, self.out_image

    def get_mid_para(self):
        return self.out_real


class Linear_W1(object):
    def __init__(self, X_real, X_image, layer_number):
        self.in_real = X_real
        self.in_image = X_image

        self.name_real = 'weight_real_' + str(layer_number) + '1'
        self.name_image = 'weight_image_' + str(layer_number) + '1'

        self.rr = tf.contrib.layers.fully_connected(self.in_real, num_outputs=N, activation_fn=None,
                                          weights_initializer=W1_init_Real, biases_initializer=None, trainable=True, scope=self.name_real)
        self.ri = tf.contrib.layers.fully_connected(self.in_real, num_outputs=N, activation_fn=None,
                                          weights_initializer=W1_init_Imag, biases_initializer=None, trainable=True, scope=self.name_image)

        self.ir = tf.contrib.layers.fully_connected(self.in_image, num_outputs=N, activation_fn=None,
                                          weights_initializer=W1_init_Real, biases_initializer=None, trainable=True, reuse=True, scope=self.name_real)
        self.ii = tf.contrib.layers.fully_connected(self.in_image, num_outputs=N, activation_fn=None,
                                          weights_initializer=W1_init_Imag, biases_initializer=None, trainable=True, reuse=True, scope=self.name_image)


        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

    def get_linear_out(self):
        return self.Sr, self.Si


class Linear_W2(object):
    def __init__(self, X_real, X_image, layer_number):
        self.in_real = X_real
        self.in_image = X_image

        self.name_real = 'weight_real_' + str(layer_number) + '2'
        self.name_image = 'weight_image_' + str(layer_number) + '2'

        self.rr = tf.contrib.layers.fully_connected(self.in_real, num_outputs=N, activation_fn=None,
                                          weights_initializer=W2_init_Real, biases_initializer=None, trainable=True, scope=self.name_real)
        self.ri = tf.contrib.layers.fully_connected(self.in_real, num_outputs=N, activation_fn=None,
                                          weights_initializer=W2_init_Imag, biases_initializer=None, trainable=True, scope=self.name_image)

        self.ir = tf.contrib.layers.fully_connected(self.in_image, num_outputs=N, activation_fn=None,
                                          weights_initializer=W2_init_Real, biases_initializer=None, trainable=True, reuse=True, scope=self.name_real)
        self.ii = tf.contrib.layers.fully_connected(self.in_image, num_outputs=N, activation_fn=None,
                                          weights_initializer=W2_init_Imag, biases_initializer=None, trainable=True, reuse=True, scope=self.name_image)


        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

    def get_linear_out(self):
        return self.Sr, self.Si



class NonLinear_OP(object):
    def __init__(self, S_real, S_image, layernumber):
        self.Sr = S_real
        self.Si = S_image
        self.name = 'alph_' + str(layernumber) + '_alph'

        self.alph = tf.Variable(25.4, trainable=True, dtype=tf.float32, name=self.name)              # Alpha Initialization
        self.S_power = tf.math.add(tf.math.square(self.Sr), tf.math.square(self.Si))
        self.S_power = tf.math.scalar_mul(self.alph, self.S_power)
        self.sin = tf.math.sin(self.S_power)
        self.cos = tf.math.cos(self.S_power)

        self.y_real = tf.math.add(tf.math.multiply(self.Sr, self.cos), tf.math.multiply(self.Si, self.sin))
        self.y_image = tf.math.subtract(tf.math.multiply(self.Si, self.cos), tf.math.multiply(self.Sr, self.sin))

    def get_nonlinear_out(self):
        return self.y_real, self.y_image

