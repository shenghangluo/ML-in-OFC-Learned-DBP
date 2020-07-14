import tensorflow as tf
import numpy as np
from Weight_Transform import Weight_Transform
from Weight_Transform_whole import Weight_Transform_whole
from scipy.io import loadmat
N = 2048
M = 1024
k = 6
Lsp = 100000
Nsp = 1
M_step = 2
STPS = 1   # Step per Span
delta = Lsp/M_step

A = loadmat('init.mat')['real']
A = np.array(A)
real_init = np.squeeze(A)
A = loadmat('init.mat')['image']
A = np.array(A)
imag_init = np.squeeze(A)

A = loadmat('init_MF.mat')['real_MF']
A = np.array(A)
MF_real_init = np.squeeze(A)

A = loadmat('init_MF.mat')['image_MF']
A = np.array(A)
MF_image_init = np.squeeze(A)


class Model(object):
    def __init__(self):
        self.X_real = tf.compat.v1.placeholder(tf.float32, shape=(None, N), name="X_real")
        self.X_image = tf.compat.v1.placeholder(tf.float32, shape=(None, N), name="X_image")
        self.y_real = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="y_real")
        self.y_image = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="y_image")

        self.X_OP_real = tf.math.scalar_mul(0.1, self.X_real)        # Since placeholder cannot assigned by another tensor
        self.X_OP_image = tf.math.scalar_mul(0.1, self.X_image)

        self.spanlist = []
        for ii in range(1, Nsp+1):              # for loop ready for large network
            self.spanlist.append(Span(X_real=self.X_OP_real, X_image=self.X_OP_image, span_number=ii))
            self.out_real, self.out_image = self.spanlist[ii-1].get_span_out()
            self.X_OP_real = self.out_real
            self.X_OP_image = self.out_image

        # MF
        # self.rr = tf.contrib.layers.fully_connected(self.out_real, num_outputs=M, activation_fn=None, biases_initializer=None,
        #                                             trainable=True, scope='MF_Real')
        # self.ri = tf.contrib.layers.fully_connected(self.out_real, num_outputs=M, activation_fn=None, biases_initializer=None,
        #                                             trainable=True, scope='MF_Image')
        # self.ir = tf.contrib.layers.fully_connected(self.out_image, num_outputs=M, activation_fn=None, biases_initializer=None,
        #                                             trainable=True, reuse=True, scope='MF_Real')
        # self.ii = tf.contrib.layers.fully_connected(self.out_image, num_outputs=M, activation_fn=None, biases_initializer=None,
        #                                             trainable=True, reuse=True, scope='MF_Image')
        self.MF_real = tf.Variable(MF_real_init, trainable=True, dtype=tf.float32, name='MF_Real')  # Filter Initialization Needed
        self.MF_Processed_real = Weight_Transform(self.MF_real, k=10, n=N, m=M)

        self.MF_image = tf.Variable(MF_image_init, trainable=True, dtype=tf.float32, name='MF_Image')  # Filter Initialization Needed
        self.MF_Processed_image = Weight_Transform(self.MF_image, k=10, n=N, m=M)

        self.rr = tf.matmul(self.out_real, self.MF_Processed_real)
        self.ri = tf.matmul(self.out_real, self.MF_Processed_image)
        self.ir = tf.matmul(self.out_image, self.MF_Processed_real)
        self.ii = tf.matmul(self.out_image, self.MF_Processed_image)

        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

        self.out_MF_real = self.Sr
        self.out_MF_image = self.Si

    def get_reconstruction(self):
        return self.out_MF_real, self.out_MF_image

    def get_middle_para(self):
        return self.out_real, self.out_image

class Span(object):
    def __init__(self, X_real, X_image, span_number):
        self.X_real = X_real
        self.X_image = X_image
        self.span_number = span_number

        self.layerlist = []
        for ii in range(1, STPS+1):              # for loop ready for large network
            self.layerlist.append(Layer(X_real=self.X_real, X_image=self.X_image, layer_number=ii, span_number=self.span_number))
            self.out_real, self.out_image = self.layerlist[ii-1].get_layer_output()
            self.X_real = self.out_real
            self.X_image = self.out_image

    def get_span_out(self):
        return self.out_real, self.out_image

    def get_w1(self):
        return self.layerlist[1].get_w1()

class Layer(object):
    def __init__(self, X_real, X_image, layer_number, span_number):
        self.X_real = X_real
        self.X_image = X_image
        self.layer_number = layer_number
        self.span_number = span_number

        # Linear W1
        self.Linear1 = Linear_W1(X_real=self.X_real, X_image=self.X_image, layer_number=self.layer_number, span_number=self.span_number)
        self.Sr, self.Si = self.Linear1.get_linear_W1_out()

        # Nonlinear
        self.Nonlinear = NonLinear_OP(S_real=self.Sr, S_image=self.Si, layer_number=self.layer_number, span_number=self.span_number)
        self.out_real, self.out_image = self.Nonlinear.get_nonlinear_out()

        # # Linear W2
        # self.Linear2 = Linear_W2(X_real=self.y_real, X_image=self.y_image, layer_number=self.layer_number, span_number=self.span_number)
        # self.out_real, self.out_image = self.Linear2.get_linear_W2_out()

    def get_layer_output(self):
        return self.out_real, self.out_image

    def get_w1(self):
        return self.Linear1.get_w1()

class Linear_W1(object):
    def __init__(self, X_real, X_image, layer_number, span_number):
        self.X_real = X_real
        self.X_image = X_image

        self.name_real = 'weight_real_' + str(span_number) + '_' + str(layer_number) + '_' + '1'
        self.name_image = 'weight_image_' + str(span_number) + '_' + str(layer_number) + '_' + '1'

        x = 1
        self.W1_real = tf.Variable(real_init, trainable=True, dtype=tf.float32, name=self.name_real)        # Filter Initialization Needed
        # f1 = lambda: self.W1_real                                                                                  # Filter Initialization Needed
        # f2 = lambda: self.W1_real*0.1                                                                               # Filter Initialization Needed
        # self.W1_real = tf.case([(tf.less(x, layer_number), f1)], default=f2)
        self.W1_Processed_real = Weight_Transform(self.W1_real, k=69, n=N, m=N)

        self.W1_image = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name=self.name_image)
        # f1 = lambda: self.W1_image                                                                                  # Filter Initialization Needed
        # f2 = lambda: self.W1_image*0.1                                                                               # Filter Initialization Needed
        # self.W1_image = tf.case([(tf.less(x, layer_number), f1)], default=f2)
        self.W1_Processed_image = Weight_Transform(self.W1_image,k=69, n=N, m=N)

        self.rr = tf.matmul(self.X_real, self.W1_Processed_real)
        self.ri = tf.matmul(self.X_real,  self.W1_Processed_image)
        self.ir = tf.matmul(self.X_image, self.W1_Processed_real)
        self.ii = tf.matmul(self.X_image, self.W1_Processed_image)

        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

    def get_linear_W1_out(self):
        return self.Sr, self.Si

    def get_w1(self):
        return self.W1_real


class Linear_W2(object):
    def __init__(self, X_real, X_image, layer_number, span_number):
        self.X_real = X_real
        self.X_image = X_image

        self.name_real = 'weight_real_' + str(span_number) + '_' + str(layer_number) + '_' + '2'
        self.name_image = 'weight_image_' + str(span_number) + '_' + str(layer_number) + '_' + '2'

        self.W2_real = tf.Variable([0.3,0.3,0.3,0.3,0.3,0.3,0.3], trainable=True, dtype=tf.float32, name=self.name_real)
        self.W2_Processed_real = Weight_Transform(self.W2_real, k=k, n=N)

        self.W2_image = tf.Variable([0.3,0.3,0.3,0.3,0.3,0.3,0.3], trainable=True, dtype=tf.float32, name=self.name_image)
        self.W2_Processed_image = Weight_Transform(self.W2_image, k=k, n=N)

        self.rr = tf.matmul(self.X_real, tf.transpose(self.W2_Processed_real))
        self.ri = tf.matmul(self.X_real,  tf.transpose(self.W2_Processed_image))
        self.ir = tf.matmul(self.X_image, tf.transpose(self.W2_Processed_real))
        self.ii = tf.matmul(self.X_image, tf.transpose(self.W2_Processed_image))

        self.Si = tf.math.add(self.ri, self.ir)
        self.Sr = tf.math.subtract(self.rr, self.ii)

    def get_linear_W2_out(self):
        return self.Sr, self.Si



class NonLinear_OP(object):
    def __init__(self, S_real, S_image, layer_number, span_number):
        self.Sr = S_real
        self.Si = S_image
        self.name = 'alph_' + str(span_number) + '_' + str(layer_number)

        self.alph = tf.Variable(-1.0*84.8, trainable=True, dtype=tf.float32, name=self.name)              # Alpha Initialization
        self.S_power = tf.math.add(tf.math.square(self.Sr), tf.math.square(self.Si))
        self.S_power = tf.math.scalar_mul(self.alph, self.S_power)
        self.sin = tf.math.sin(self.S_power)
        self.cos = tf.math.cos(self.S_power)

        self.y_real = tf.math.subtract(tf.math.multiply(self.Sr, self.cos), tf.math.multiply(self.Si, self.sin))
        self.y_image = tf.math.add(tf.math.multiply(self.Si, self.cos), tf.math.multiply(self.Sr, self.sin))

    def get_nonlinear_out(self):
        return self.y_real, self.y_image

