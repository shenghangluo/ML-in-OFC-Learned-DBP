import tensorflow as tf
import numpy as np
from Weight_Transform import Weight_Transform
from MF_Transform import MF_Transform
from scipy.io import loadmat

N = 16384
M = 8192
MF_M = M+500
k = 80
Lsp = 100000
Nsp = 32
CD_Overlap = np.floor(k*Nsp/2).astype(int)
alpha = -1.0*23.4845
MF_length = 501

A = loadmat('init.mat')['real']
A = np.array(A)
real_init = np.squeeze(A)
A = loadmat('init.mat')['image']
A = np.array(A)
imag_init = np.squeeze(A)


def one_layer(X_real, X_image, W1_real, W1_image, alph):
        ## High memory consuming
        W1_real = Weight_Transform(W1_real, k=k, n=N)
        W1_image = Weight_Transform(W1_image, k=k, n=N)

        X_complex_fft = tf.fft(tf.complex(X_real, X_image))
        W_complex_fft = tf.fft(tf.complex(W1_real, W1_image))
        X_complex = tf.ifft(X_complex_fft * W_complex_fft)
        X_real = tf.math.real(X_complex)
        X_image = tf.math.imag(X_complex)

        # None-linear
        S_power = tf.math.add(tf.math.square(X_real), tf.math.square(X_image))
        S_power = tf.math.scalar_mul(alph, S_power)
        sin = tf.math.sin(S_power)
        cos = tf.math.cos(S_power)

        out_real = tf.math.subtract(tf.math.multiply(X_real, cos), tf.math.multiply(X_image, sin))
        out_image = tf.math.add(tf.math.multiply(X_image, cos), tf.math.multiply(X_real, sin))
        return out_real, out_image


def layers(X_real, X_image):
        # Layer_1
        W1_real_1 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_1_real')
        W1_image_1 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_1_image')
        alph_1 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha1')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_1, W1_image=W1_image_1, alph=alph_1)

        # Layer_2
        W1_real_2 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_2_real')
        W1_image_2 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_2_image')
        alph_2 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha2')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_2, W1_image=W1_image_2, alph=alph_2)

        # Layer_3
        W1_real_3 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_3_real')
        W1_image_3 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_3_image')
        alph_3 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha3')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_3, W1_image=W1_image_3, alph=alph_3)

        # Layer_4
        W1_real_4 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_4_real')
        W1_image_4 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_4_image')
        alph_4 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha4')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_4, W1_image=W1_image_4, alph=alph_4)

        # Layer_5
        W1_real_5 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_5_real')
        W1_image_5 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_5_image')
        alph_5 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha5')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_5, W1_image=W1_image_5, alph=alph_5)

        # Layer_6
        W1_real_6 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_6_real')
        W1_image_6 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_6_image')
        alph_6 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha6')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_6, W1_image=W1_image_6, alph=alph_6)

        # Layer_7
        W1_real_7 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_7_real')
        W1_image_7 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_7_image')
        alph_7 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha7')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_7, W1_image=W1_image_7, alph=alph_7)

        # Layer_8
        W1_real_8 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_8_real')
        W1_image_8 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_8_image')
        alph_8 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha8')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_8, W1_image=W1_image_8, alph=alph_8)

        # Layer_9
        W1_real_9 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_9_real')
        W1_image_9 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_9_image')
        alph_9 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha9')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_9, W1_image=W1_image_9, alph=alph_9)

        # Layer_10
        W1_real_10 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_10_real')
        W1_image_10 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_10_image')
        alph_10 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha10')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_10, W1_image=W1_image_10, alph=alph_10)

        # Layer_11
        W1_real_11 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_11_real')
        W1_image_11 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_11_image')
        alph_11 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha11')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_11, W1_image=W1_image_11, alph=alph_11)

        # Layer_12
        W1_real_12 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_12_real')
        W1_image_12 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_12_image')
        alph_12 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha12')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_12, W1_image=W1_image_12, alph=alph_12)

        # Layer_13
        W1_real_13 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_13_real')
        W1_image_13 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_13_image')
        alph_13 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha13')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_13, W1_image=W1_image_13, alph=alph_13)

        # Layer_14
        W1_real_14 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_14_real')
        W1_image_14 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_14_image')
        alph_14 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha14')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_14, W1_image=W1_image_14, alph=alph_14)

        # Layer_15
        W1_real_15 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_15_real')
        W1_image_15 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_15_image')
        alph_15 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha15')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_15, W1_image=W1_image_15, alph=alph_15)

        # Layer_16
        W1_real_16 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_16_real')
        W1_image_16 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_16_image')
        alph_16 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha16')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_16, W1_image=W1_image_16, alph=alph_16)

        # Layer_17
        W1_real_17 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_17_real')
        W1_image_17 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_17_image')
        alph_17 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha17')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_17, W1_image=W1_image_17, alph=alph_17)

        #Layer_18
        W1_real_18 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_18_real')
        W1_image_18 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_18_image')
        alph_18 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha18')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_18, W1_image=W1_image_18, alph=alph_18)

        # Layer_19
        W1_real_19 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_19_real')
        W1_image_19 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_19_image')
        alph_19 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha19')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_19, W1_image=W1_image_19, alph=alph_19)

        # Layer_20
        W1_real_20 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_20_real')
        W1_image_20 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_20_image')
        alph_20 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha20')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_20, W1_image=W1_image_20, alph=alph_20)

        # Layer_21
        W1_real_21 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_21_real')
        W1_image_21 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_21_image')
        alph_21 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha21')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_21, W1_image=W1_image_21, alph=alph_21)

        # Layer_22
        W1_real_22 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_22_real')
        W1_image_22 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_22_image')
        alph_22 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha22')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_22, W1_image=W1_image_22, alph=alph_22)

        # Layer_23
        W1_real_23 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_23_real')
        W1_image_23 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_23_image')
        alph_23 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha23')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_23, W1_image=W1_image_23, alph=alph_23)

        # Layer_24
        W1_real_24 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_24_real')
        W1_image_24 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_24_image')
        alph_24 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha24')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_24, W1_image=W1_image_24, alph=alph_24)

        # Layer_25
        W1_real_25 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_25_real')
        W1_image_25 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_25_image')
        alph_25 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha25')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_25, W1_image=W1_image_25, alph=alph_25)

        # Layer_26
        W1_real_26 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_26_real')
        W1_image_26 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_26_image')
        alph_26 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha26')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_26, W1_image=W1_image_26, alph=alph_26)

        # Layer_27
        W1_real_27 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_27_real')
        W1_image_27 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_27_image')
        alph_27 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha27')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_27, W1_image=W1_image_27, alph=alph_27)

        # Layer_28
        W1_real_28 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_28_real')
        W1_image_28 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_28_image')
        alph_28 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha28')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_28, W1_image=W1_image_28, alph=alph_28)

        # Layer_29
        W1_real_29 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_29_real')
        W1_image_29 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_29_image')
        alph_29 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha29')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_29, W1_image=W1_image_29, alph=alph_29)

        # Layer_30
        W1_real_30 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_30_real')
        W1_image_30 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_30_image')
        alph_30 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha30')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_30, W1_image=W1_image_30, alph=alph_30)

        # Layer_31
        W1_real_31 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_31_real')
        W1_image_31 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_31_image')
        alph_31 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha31')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_31, W1_image=W1_image_31, alph=alph_31)

        # Layer_32
        W1_real_32 = tf.Variable(real_init, trainable=True, dtype=tf.float32, name='W_32_real')
        W1_image_32 = tf.Variable(imag_init, trainable=True, dtype=tf.float32, name='W_32_image')
        alph_32 = tf.Variable(alpha, trainable=True, dtype=tf.float32, name='alpha32')
        X_real, X_image = one_layer(X_real=X_real, X_image=X_image, W1_real=W1_real_32, W1_image=W1_image_32, alph=alph_32)

        return X_real, X_image
