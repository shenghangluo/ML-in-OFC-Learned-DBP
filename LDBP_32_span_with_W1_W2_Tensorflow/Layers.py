import tensorflow as tf
import numpy as np
from Weight_Transform import Weight_Transform
from MF_Transform import MF_Transform
from scipy.io import loadmat

N = 16384
M = 8192
MF_M = M+500
k = 12
Lsp = 100000
Nsp = 32
CD_Overlap = np.floor(k*Nsp).astype(int)
alpha = -1.0*23.4845
MF_length = 501

A = loadmat('init.mat')['real']
A = np.array(A)
real_init = np.squeeze(A)
A = loadmat('init.mat')['image']
A = np.array(A)
imag_init = np.squeeze(A)


def one_layer(X_real, X_image, W1_real, W1_image, W2_real, W2_image, alph, ii):
        ## High memory consuming
        all_zeros = tf.zeros([k+1], tf.float32)
        ones = tf.ones([k+1], tf.bool)
        zeros = tf.zeros([k+1], tf.bool)
        sliced_zeros = tf.slice(zeros, [ii], [k+1 - ii])
        sliced_ones = tf.slice(ones, [0], [ii])

        mask = tf.concat((sliced_ones, sliced_zeros), axis=0)
        W1_real = tf.where(mask, all_zeros, W1_real)
        W1_image = tf.where(mask, all_zeros, W1_image)

        W1_real = Weight_Transform(W1_real, k=k, n=N)
        W1_image = Weight_Transform(W1_image, k=k, n=N)

        W2_real = tf.where(mask, all_zeros, W2_real)
        W2_image = tf.where(mask, all_zeros, W2_image)

        W2_real = Weight_Transform(W2_real, k=k, n=N)
        W2_image = Weight_Transform(W2_image, k=k, n=N)

        X_complex_fft = tf.fft(tf.complex(X_real, X_image))
        W1_complex_fft = tf.fft(tf.complex(W1_real, W1_image))
        X_complex = tf.ifft(X_complex_fft * W1_complex_fft)
        X_real = tf.math.real(X_complex)
        X_image = tf.math.imag(X_complex)

        # None-linear
        S_power = tf.math.add(tf.math.square(X_real), tf.math.square(X_image))
        S_power = tf.math.scalar_mul(alph, S_power)
        sin = tf.math.sin(S_power)
        cos = tf.math.cos(S_power)

        X_real = tf.math.subtract(tf.math.multiply(X_real, cos), tf.math.multiply(X_image, sin))
        X_image = tf.math.add(tf.math.multiply(X_image, cos), tf.math.multiply(X_real, sin))

        # W2
        X_complex_fft = tf.fft(tf.complex(X_real, X_image))
        W2_complex_fft = tf.fft(tf.complex(W2_real, W2_image))
        X_complex = tf.ifft(X_complex_fft * W2_complex_fft)
        out_real = tf.math.real(X_complex)
        out_image = tf.math.imag(X_complex)

        return out_real, out_image


def layers(X_real, X_image, ii):
        # Layer_1
        W1_real_1 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_1_real')
        W1_image_1 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_1_image')
        alph_1 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha1')
        W2_real_1 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_1_real_2')
        W2_image_1 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_1_image_2')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_1, W1_image=W1_image_1, W2_real=W2_real_1, W2_image=W2_image_1, alph=alph_1, ii=ii)

        # Layer_2
        W1_real_2 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real')
        W1_image_2 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image')
        alph_2 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha2')
        W2_real_2 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_2')
        W2_image_2 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_2')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_2, W1_image=W1_image_2,  W2_real=W2_real_2, W2_image=W2_image_2, alph=alph_2, ii=ii)

        # Layer_3
        W1_real_3 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_3_real')
        W1_image_3 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_3_image')
        alph_3 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha3')
        W2_real_3 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_3')
        W2_image_3 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_3')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_3, W1_image=W1_image_3,  W2_real=W2_real_3, W2_image=W2_image_3, alph=alph_3, ii=ii)

        # Layer_4
        W1_real_4 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_4_real')
        W1_image_4 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_4_image')
        alph_4 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha4')
        W2_real_4 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_2')
        W2_image_4 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_2')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_4, W1_image=W1_image_4,  W2_real=W2_real_4, W2_image=W2_image_4, alph=alph_4, ii=ii)

        # Layer_5
        W1_real_5 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_5_real')
        W1_image_5 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_5_image')
        alph_5 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha5')
        W2_real_5 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_5')
        W2_image_5 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_5')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_5, W1_image=W1_image_5,  W2_real=W2_real_5, W2_image=W2_image_5, alph=alph_5, ii=ii)

        # Layer_6
        W1_real_6 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_6_real')
        W1_image_6 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_6_image')
        alph_6 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha6')
        W2_real_6 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_6')
        W2_image_6 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_6')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_6, W1_image=W1_image_6,  W2_real=W2_real_6, W2_image=W2_image_6, alph=alph_6, ii=ii)

        # Layer_7
        W1_real_7 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_7_real')
        W1_image_7 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_7_image')
        alph_7 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha7')
        W2_real_7 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_7')
        W2_image_7 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_7')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_7, W1_image=W1_image_7,  W2_real=W2_real_7, W2_image=W2_image_7, alph=alph_7, ii=ii)

        # Layer_8
        W1_real_8 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_8_real')
        W1_image_8 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_8_image')
        alph_8 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha8')
        W2_real_8 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_8')
        W2_image_8 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_8')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_8, W1_image=W1_image_8,  W2_real=W2_real_8, W2_image=W2_image_8, alph=alph_8, ii=ii)

        # Layer_9
        W1_real_9 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_9_real')
        W1_image_9 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_9_image')
        alph_9 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha9')
        W2_real_9 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_9')
        W2_image_9 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_9')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_9, W1_image=W1_image_9,  W2_real=W2_real_9, W2_image=W2_image_9, alph=alph_9, ii=ii)

        # Layer_10
        W1_real_10 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_10_real')
        W1_image_10 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_10_image')
        alph_10 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha10')
        W2_real_10 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_10')
        W2_image_10 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_10')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_10, W1_image=W1_image_10,  W2_real=W2_real_10, W2_image=W2_image_10, alph=alph_10, ii=ii)

        # Layer_11
        W1_real_11 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_11_real')
        W1_image_11 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_11_image')
        alph_11 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha11')
        W2_real_11 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_11')
        W2_image_11 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_11')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_11, W1_image=W1_image_11,  W2_real=W2_real_11, W2_image=W2_image_11, alph=alph_11, ii=ii)

        # Layer_12
        W1_real_12 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_12_real')
        W1_image_12 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_12_image')
        alph_12 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha12')
        W2_real_12 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_12')
        W2_image_12 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_12')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_12, W1_image=W1_image_12,  W2_real=W2_real_12, W2_image=W2_image_12, alph=alph_12, ii=ii)

        # Layer_13
        W1_real_13 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_13_real')
        W1_image_13 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_13_image')
        alph_13 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha13')
        W2_real_13 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_13')
        W2_image_13 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_13')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_13, W1_image=W1_image_13,  W2_real=W2_real_13, W2_image=W2_image_13, alph=alph_13, ii=ii)

        # Layer_14
        W1_real_14 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_14_real')
        W1_image_14 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_14_image')
        alph_14 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha14')
        W2_real_14 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_14')
        W2_image_14 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_14')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_14, W1_image=W1_image_14,  W2_real=W2_real_14, W2_image=W2_image_14, alph=alph_14, ii=ii)

        # Layer_15
        W1_real_15 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_15_real')
        W1_image_15 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_15_image')
        alph_15 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha15')
        W2_real_15 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_15')
        W2_image_15 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_15')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_15, W1_image=W1_image_15,  W2_real=W2_real_15, W2_image=W2_image_15, alph=alph_15, ii=ii)

        # Layer_16
        W1_real_16 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_16_real')
        W1_image_16 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_16_image')
        alph_16 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha16')
        W2_real_16 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_16')
        W2_image_16 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_16')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_16, W1_image=W1_image_16,  W2_real=W2_real_16, W2_image=W2_image_16, alph=alph_16, ii=ii)

        # Layer_17
        W1_real_17 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_17_real')
        W1_image_17 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_17_image')
        alph_17 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha17')
        W2_real_17 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_17')
        W2_image_17 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_17')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_17, W1_image=W1_image_17,  W2_real=W2_real_17, W2_image=W2_image_17, alph=alph_17, ii=ii)

        #Layer_18
        W1_real_18 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_18_real')
        W1_image_18 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_18_image')
        alph_18 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha18')
        W2_real_18 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_18')
        W2_image_18 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_18')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_18, W1_image=W1_image_18,  W2_real=W2_real_18, W2_image=W2_image_18, alph=alph_18, ii=ii)

        # Layer_19
        W1_real_19 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_19_real')
        W1_image_19 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_19_image')
        alph_19 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha19')
        W2_real_19 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_19')
        W2_image_19 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_19')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_19, W1_image=W1_image_19,  W2_real=W2_real_19, W2_image=W2_image_19, alph=alph_19, ii=ii)

        # Layer_20
        W1_real_20 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_20_real')
        W1_image_20 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_20_image')
        alph_20 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha20')
        W2_real_20 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_20')
        W2_image_20 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_20')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_20, W1_image=W1_image_20,  W2_real=W2_real_20, W2_image=W2_image_20, alph=alph_20, ii=ii)

        # Layer_21
        W1_real_21 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_21_real')
        W1_image_21 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_21_image')
        alph_21 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha21')
        W2_real_21 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_21')
        W2_image_21 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_21')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_21, W1_image=W1_image_21,  W2_real=W2_real_21, W2_image=W2_image_21, alph=alph_21, ii=ii)

        # Layer_22
        W1_real_22 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_22_real')
        W1_image_22 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_22_image')
        alph_22 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha22')
        W2_real_22 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_22')
        W2_image_22 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_22')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_22, W1_image=W1_image_22,  W2_real=W2_real_22, W2_image=W2_image_22, alph=alph_22, ii=ii)

        # Layer_23
        W1_real_23 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_23_real')
        W1_image_23 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_23_image')
        alph_23 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha23')
        W2_real_23 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_23')
        W2_image_23 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_23')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_23, W1_image=W1_image_23,  W2_real=W2_real_23, W2_image=W2_image_23, alph=alph_23, ii=ii)

        # Layer_24
        W1_real_24 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_24_real')
        W1_image_24 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_24_image')
        alph_24 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha24')
        W2_real_24 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_24')
        W2_image_24 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_24')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_24, W1_image=W1_image_24,  W2_real=W2_real_24, W2_image=W2_image_24, alph=alph_24, ii=ii)

        # Layer_25
        W1_real_25 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_25_real')
        W1_image_25 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_25_image')
        alph_25 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha25')
        W2_real_25 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_25')
        W2_image_25 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_25')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_25, W1_image=W1_image_25,  W2_real=W2_real_25, W2_image=W2_image_25, alph=alph_25, ii=ii)

        # Layer_26
        W1_real_26 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_26_real')
        W1_image_26 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_26_image')
        alph_26 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha26')
        W2_real_26 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_26')
        W2_image_26 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_26')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_26, W1_image=W1_image_26,  W2_real=W2_real_26, W2_image=W2_image_26, alph=alph_26, ii=ii)

        # Layer_27
        W1_real_27 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_27_real')
        W1_image_27 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_27_image')
        alph_27 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha27')
        W2_real_27 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_27')
        W2_image_27 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_27')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_27, W1_image=W1_image_27,  W2_real=W2_real_27, W2_image=W2_image_27, alph=alph_27, ii=ii)

        # Layer_28
        W1_real_28 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_28_real')
        W1_image_28 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_28_image')
        alph_28 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha28')
        W2_real_28 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_28')
        W2_image_28 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_28')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_28, W1_image=W1_image_28,  W2_real=W2_real_28, W2_image=W2_image_28, alph=alph_28, ii=ii)

        # Layer_29
        W1_real_29 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_29_real')
        W1_image_29 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_29_image')
        alph_29 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha29')
        W2_real_29 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_29')
        W2_image_29 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_29')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_29, W1_image=W1_image_29,  W2_real=W2_real_29, W2_image=W2_image_29, alph=alph_29, ii=ii)

        # Layer_30
        W1_real_30 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_30_real')
        W1_image_30 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_30_image')
        alph_30 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha30')
        W2_real_30 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_30')
        W2_image_30 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_30')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_30, W1_image=W1_image_30,  W2_real=W2_real_30, W2_image=W2_image_30, alph=alph_30, ii=ii)

        # Layer_31
        W1_real_31 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_31_real')
        W1_image_31 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_31_image')
        alph_31 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha31')
        W2_real_31 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_31')
        W2_image_31 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_31')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_31, W1_image=W1_image_31,  W2_real=W2_real_31, W2_image=W2_image_31, alph=alph_31, ii=ii)

        # Layer_32
        W1_real_32 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_32_real')
        W1_image_32 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_32_image')
        alph_32 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha32')
        W2_real_32 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real_32')
        W2_image_32 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image_32')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_32, W1_image=W1_image_32,  W2_real=W2_real_32, W2_image=W2_image_32, alph=alph_32, ii=ii)

        return X_real, X_image
