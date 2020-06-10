import tensorflow as tf


class Model(object):
    def __init__(self):
        self.X_real = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="X_real")
        self.X_image = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name="X_image")

        '''one layer'''#---------------- with twp Linear and one Nonlinear Operations
        self.layer1 = Layer(X_real=self.X_real, X_image=self.X_image, layer_number=1)
        self.out_real, self.out_image = self.layer1.get_layer_output()

    def get_reconstruction(self):
        return self.out_real, self.out_image


class Layer(object):
    def __init__(self, X_real, X_image, layer_number):
        self.X_real = X_real
        self.X_image = X_image

        # Linear W1
        self.name_real = 'weight_real_' + str(layer_number) + '1'
        self.name_image = 'weight_image_' + str(layer_number) + '1'
        self.Linear11 = Linear_OP(X_real=self.X_real, X_image=self.X_image, name_real=self.name_real, name_image=self.name_image)
        self.Sr, self.Si = self.Linear11.get_linear_out()

        # Nonlinear
        self.Nonlinear1 = NonLinear_OP(S_real=self.Sr, S_image=self.Si)
        self.y_real, self.y_image = self.Nonlinear1.get_nonlinear_out()

        # Linear W2
        self.name_real = 'weight_real_' + str(layer_number) + '2'
        self.name_image = 'weight_image_' + str(layer_number) + '2'
        self.Linear12 = Linear_OP(X_real=self.y_real, X_image=self.y_image, name_real=self.name_real, name_image=self.name_image)
        self.out_real, self.out_image = self.Linear12.get_linear_out()

    def get_layer_output(self):
        return self.out_real, self.out_image



class Linear_OP(object):
    def __init__(self, X_real, X_image, name_real, name_image):
        self.in_real = X_real
        self.in_image = X_image

        self.rr = tf.layers.dense(inputs=self.in_real, units=10, activation=None, name=name_real)
        self.ri = tf.layers.dense(inputs=self.in_real, units=10, activation=None, name=name_image)

        self.ir = tf.layers.dense(inputs=self.in_image, units=10, activation=None, name=name_real, reuse=True)
        self.ii = tf.layers.dense(inputs=self.in_image, units=10, activation=None, name=name_image, reuse=True)

        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

    def get_linear_out(self):
        return self.Sr, self.Si




class NonLinear_OP(object):
    def __init__(self, S_real, S_image):
        self.Sr = S_real
        self.Si = S_image

        self.alph = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        self.S_power = tf.math.square(self.Sr) + tf.math.square(self.Si)
        self.S_power = tf.math.scalar_mul(self.alph, self.S_power)
        self.sin = tf.math.sin(self.S_power)
        self.cos = tf.math.cos(self.S_power)

        self.y_real = tf.math.add(tf.math.multiply(self.Sr, self.cos), tf.math.multiply(self.Si, self.sin))
        self.y_image = tf.math.subtract(tf.math.multiply(self.Si, self.cos), tf.math.multiply(self.Sr, self.sin))

    def get_nonlinear_out(self):
        return self.y_real, self.y_image
