import numpy as np
import keras.backend as K
import scipy.linalg as linalg
from scipy.linalg import dft

N = 100
M = 50

alp = 0.2           #attenuation
Lsp = 100
M_step = 2
delta = Lsp/M_step
beta2 = 2.175*1e-26


def W1_init_Real(shape, dtype=None, partition_info=None):                # both W1 and W2 need initialization
    # fk = np.arange(N)
    # fk = 2.0 * np.pi * fk / N
    # fk = np.square(fk)
    # Hk = -1.0 * 1j * fk * beta2 / 2.0 + alp / 2.0
    # Hk = Hk * delta / 2.0
    # Hk = np.exp(Hk)
    Hk = np.ones(N)
    Hk = Hk*3.0
    dia_Hk = np.diag(Hk)

    w = dft(N)
    w_invert = linalg.inv(w)

    A = np.matmul(w_invert, dia_Hk)
    A = np.matmul(A, w)
    A_real = np.real(A)
    A_real = A_real*0.1

    return K.variable(value=A_real, dtype=dtype)


def W1_init_Imag(shape, dtype=None, partition_info=None):  # both W1 and W2 need initialization
    # fk = np.arange(N)
    # fk = 2.0 * np.pi * fk / N
    # fk = np.square(fk)
    # Hk = -1.0 * 1j * fk * beta2 / 2.0 + alp / 2.0
    # Hk = Hk * delta / 2.0
    # Hk = np.exp(Hk)
    Hk = np.ones(N)
    Hk = Hk*3.0
    dia_Hk = np.diag(Hk)

    w = dft(N)
    w_invert = linalg.inv(w)

    A = np.matmul(w_invert, dia_Hk)
    A = np.matmul(A, w)
    A_imag = np.imag(A)
    A_imag = A_imag * 0.1

    return K.variable(value=A_imag, dtype=dtype)


def W2_init_Real(shape, dtype=None, partition_info=None):  # both W1 and W2 need initialization
    # fk = np.arange(N)
    # fk = 2.0 * np.pi * fk / N
    # fk = np.square(fk)
    # Hk = -1.0 * 1j * fk * beta2 / 2.0 + alp / 2.0
    # Hk = Hk * delta / 2.0
    # Hk = np.exp(Hk)
    Hk = np.ones(N)
    Hk = Hk*3.0
    dia_Hk = np.diag(Hk)

    w = dft(N)
    w_invert = linalg.inv(w)

    A = np.matmul(w_invert, dia_Hk)
    A = np.matmul(A, w)
    A_real = np.real(A)

    return K.variable(value=A_real, dtype=dtype)


def W2_init_Imag(shape, dtype=None, partition_info=None):  # both W1 and W2 need initialization
    # fk = np.arange(N)
    # fk = 2.0 * np.pi * fk / N
    # fk = np.square(fk)
    # Hk = -1.0 * 1j * fk * beta2 / 2.0 + alp / 2.0
    # Hk = Hk * delta / 2.0
    # Hk = np.exp(Hk)
    Hk = np.ones(N)
    Hk = Hk*3.0
    dia_Hk = np.diag(Hk)

    w = dft(N)
    w_invert = linalg.inv(w)

    A = np.matmul(w_invert, dia_Hk)
    A = np.matmul(A, w)
    A_imag = np.imag(A)

    return K.variable(value=A_imag, dtype=dtype)



#
# def MF_init(shape, dtype=None, partition_info=None):
#     val = np.ones((N, M))
#     val[1:-1, 1:-1] = 0
#     return K.variable(value=val, dtype=dtype)