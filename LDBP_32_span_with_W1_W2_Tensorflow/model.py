import tensorflow as tf
import numpy as np
from Weight_Transform import Weight_Transform
from Layers import layers
from scipy.io import loadmat
import keras.backend as K

N = 16384
M = 8192
MF_M = M+500
k = 12
Lsp = 100000
Nsp = 32
CD_Overlap = np.floor(k*Nsp).astype(int)
alpha = -1.0*23.4845
MF_length = 501
M_step = 2
STPS = 1   # Step per Span
delta = Lsp/M_step



A = loadmat('MF_init.mat')['MF_real']
A = np.array(A)
MF_real_init = np.squeeze(A)
A = loadmat('MF_init.mat')['MF_image']
A = np.array(A)
MF_image_init = np.squeeze(A)

class Model(object):
    def __init__(self):
        self.X_real_in = tf.compat.v1.placeholder(tf.float32, shape=(None, N), name="X_real")
        self.X_image_in = tf.compat.v1.placeholder(tf.float32, shape=(None, N), name="X_image")
        self.y_real = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="y_real")
        self.y_image = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="y_image")
        self.mask = tf.compat.v1.placeholder(tf.int32, shape=(), name="mask")

        self.X_real = self.X_real_in        # Since placeholder cannot assigned by another tensor
        self.X_image = self.X_image_in

        self.X_real, self.X_image = layers(self.X_real, self.X_image, ii=self.mask)


        self.after_CD_real = self.X_real
        self.after_CD_image = self.X_image
        self.out_real = self.X_real
        self.out_image = self.X_image

        # MF
        # self.MF_real = tf.Variable(MF_real_init, trainable=False, dtype=tf.float32, name='MF_real')        # Filter Initialization Needed
        # self.MF_Processed_real = Weight_Transform(self.MF_real, k=MF_length-1, n=N, m=N+2*MF_length-2)

        paddings = tf.constant([[0, 0], [MF_length - 1, MF_length - 1]])
        self.out_real = tf.pad(self.out_real, paddings, "CONSTANT")
        # print("out_real_1", self.out_real.shape)
        self.out_real = tf.reshape(self.out_real, [-1, 1, N + 2 * MF_length - 2])
        # print("out_real_2", self.out_real.shape)
        self.out_real = tf.transpose(self.out_real, [0, 2, 1])
        # print("out_real_3", self.out_real.shape)

        self.out_image = tf.pad(self.out_image, paddings, "CONSTANT")
        self.out_image = tf.reshape(self.out_image, [-1, 1, N + 2 * MF_length - 2])
        self.out_image = tf.transpose(self.out_image, [0, 2, 1])

        self.MF_real = tf.constant(MF_real_init, dtype=tf.float32, name='MF_real')
        self.MF_real = K.reverse(self.MF_real, axes=0)
        self.MF_real = tf.reshape(self.MF_real, [2 * MF_length - 1, 1, 1])

        self.MF_image = tf.constant(MF_image_init, dtype=tf.float32, name='MF_image')
        self.MF_image = K.reverse(self.MF_image, axes=0)
        self.MF_image = tf.reshape(self.MF_image, [2 * MF_length - 1, 1, 1])

        self.rr = tf.nn.conv1d(self.out_real, filters=self.MF_real, padding='SAME')
        self.rr = tf.transpose(self.rr, [0, 2, 1])
        # print("out_real is: ", self.out_real.shape)
        self.rr = tf.reshape(self.rr, [-1, N + 2 * MF_length - 2])
        print("rr is: ", self.rr.shape)

        self.ri = tf.nn.conv1d(self.out_real, filters=self.MF_image, padding='SAME')
        self.ri = tf.transpose(self.ri, [0, 2, 1])
        self.ri = tf.reshape(self.ri, [-1, N + 2 * MF_length - 2])

        self.ir = tf.nn.conv1d(self.out_image, filters=self.MF_real, padding='SAME')
        self.ir = tf.transpose(self.ir, [0, 2, 1])
        self.ir = tf.reshape(self.ir, [-1, N + 2 * MF_length - 2])

        self.ii = tf.nn.conv1d(self.out_image, filters=self.MF_image, padding='SAME')
        self.ii = tf.transpose(self.ii, [0, 2, 1])
        self.ii = tf.reshape(self.ii, [-1, N + 2 * MF_length - 2])

        # self.MF_image = tf.Variable(MF_image_init, trainable=False, dtype=tf.float32, name='MF_image')        # Filter Initialization Needed
        # self.MF_Processed_image = Weight_Transform(self.MF_image, k=MF_length-1, n=N, m=N+2*MF_length-2)
        #
        # self.rr = tf.matmul(self.out_real, self.MF_Processed_real)
        # print("rr", self.rr.shape)
        # self.ri = tf.matmul(self.out_real, self.MF_Processed_image)
        # self.ir = tf.matmul(self.out_image, self.MF_Processed_real)
        # self.ii = tf.matmul(self.out_image, self.MF_Processed_image)

        self.out_image = tf.math.add(self.ri, self.ir)
        self.out_real = tf.math.subtract(self.rr, self.ii)
        # print("out_image:", self.out_image.shape)
        # self.out_real = tf.matmul(self.out_real, self.MF)
        # self.out_image = tf.matmul(self.out_image, self.MF)
        self.after_MF_real, self.after_MF_image = self.out_real, self.out_image

        # Downsample
        self.out_real = tf.contrib.layers.fully_connected(self.out_real, num_outputs=MF_M, activation_fn=None,
                                                              weights_initializer=my_init, biases_initializer=None,
                                                              trainable=False)
        self.out_image = tf.contrib.layers.fully_connected(self.out_image, num_outputs=MF_M, activation_fn=None,
                                                                  weights_initializer=my_init, biases_initializer=None,
                                                                  trainable=False)
        print("outreal shape ",self.out_real.shape)
        #self.after_down_real = self.out_real
        #self.after_down_image = self.out_image

        # Slice
        self.out_real = tf.slice(self.out_real, [0, 250+CD_Overlap], [-1, M-CD_Overlap])
        self.out_image = tf.slice(self.out_image, [0, 250+CD_Overlap], [-1, M-CD_Overlap])
        #self.after_down_real = self.out_real
        #self.after_down_image = self.out_image
        print("sliced shape ",self.out_real.shape)
        self.out_real = tf.math.scalar_mul(22.3872, self.out_real)        # Since placeholder cannot assigned by another tensor
        self.out_image = tf.math.scalar_mul(22.3872, self.out_image)

        self.after_overlap_real = self.out_real
        self.after_overlap_image = self.out_image

        self.final_out_real = tf.slice(self.out_real, [0, 25], [-1, M-CD_Overlap-50])
        self.final_out_image = tf.slice(self.out_image, [0, 25], [-1, M-CD_Overlap-50])

        # Slice Trans
        self.y_real_sliced = tf.slice(self.y_real, [0, 0], [-1, M-CD_Overlap])
        self.y_image_sliced = tf.slice(self.y_image, [0, 0], [-1, M-CD_Overlap])
        print("sliced Trans shape ",self.y_real_sliced.shape)

        self.final_y_real = tf.slice(self.y_real_sliced, [0, 25], [-1, M-CD_Overlap-50])
        self.final_y_image = tf.slice(self.y_image_sliced, [0, 25], [-1, M-CD_Overlap-50])

        # Phase Shift
        self.out_real_sliced = tf.slice(self.out_real, [1, 0], [1, 3000])
        self.out_image_sliced = tf.slice(self.out_image, [1, 0], [1, 3000])

        self.trans_real_sliced = tf.slice(self.y_real, [1, 0], [1, 3000])
        self.trans_image_sliced = tf.slice(self.y_image, [1, 0], [1, 3000])

        paddings = tf.constant([[0, 0], [0, 2999]])
        self.out_real_sliced = tf.pad(self.out_real_sliced, paddings, "CONSTANT")
        self.out_image_sliced = tf.pad(self.out_image_sliced, paddings, "CONSTANT")
        self.trans_real_sliced = tf.pad(self.trans_real_sliced, paddings, "CONSTANT")
        self.trans_image_sliced = tf.pad(self.trans_image_sliced, paddings, "CONSTANT")

        self.out_complex = tf.fft(tf.complex(self.out_real_sliced, self.out_image_sliced))
        self.trans_complex = tf.fft(tf.complex(self.trans_real_sliced, self.trans_image_sliced))
        self.xcorr = tf.squeeze(tf.ifft(tf.math.conj(self.out_complex) * (self.trans_complex)))

        self.abs = tf.math.abs(self.xcorr)
        self.max = tf.math.argmax(self.abs)

        self.max_value = self.xcorr[self.max]

        self.max_value_real = tf.real(self.max_value)
        self.max_value_image = tf.imag(self.max_value)

        self.Phi = tf.math.angle(self.xcorr[self.max])
        self.Phi = tf.complex(0.0, self.Phi)
        self.Phi = tf.math.exp(self.Phi)

        self.phi_x_real = tf.real(self.Phi)
        self.phi_x_image = tf.imag(self.Phi)
        self.rr = tf.math.scalar_mul(self.phi_x_real, self.final_out_real)
        self.ii = tf.math.scalar_mul(self.phi_x_image, self.final_out_image)
        self.ir = tf.math.scalar_mul(self.phi_x_image, self.final_out_real)
        self.ri = tf.math.scalar_mul(self.phi_x_real, self.final_out_image)

        self.final_out_image = tf.math.add(self.ri, self.ir)
        self.final_out_real = tf.math.subtract(self.rr, self.ii)



    def get_reconstruction(self):
        return self.final_out_real, self.final_out_image

    def get_Transmit(self):
        return self.final_y_real, self.final_y_image
    def after_CD(self):
        return self.after_CD_real, self.after_CD_image
    def after_MF(self):
        return self.after_MF_real, self.after_MF_image

def my_init(shape, dtype=None, partition_info=None):
    val = np.zeros((2*MF_M, MF_M))
    row_index = np.arange(0,2*MF_M,2)
    colum_index = np.arange(MF_M)
    val[row_index, colum_index] = 1.0

    return K.variable(value=val, dtype=dtype)