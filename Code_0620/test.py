import tensorflow
import numpy as np
#
# #Trainable Parameters
# W = tensorflow.Variable([0.3], dtype=tensorflow.float32)
# b = tensorflow.Variable([-0.2], dtype=tensorflow.float32)
#
# #Training Data (inputs/outputs)
# x = tensorflow.placeholder(dtype=tensorflow.float32)
# y = tensorflow.placeholder(dtype=tensorflow.float32)
#
# x_train = [1, 2, 3, 4]
# y_train = [0, 1, 2, 3]
#
# #Linear Model
# linear_model = W * x + b
#
# #Linear Regression Loss Function - sum of the squares
# squared_deltas = tensorflow.square(linear_model - y_train)
# loss = tensorflow.reduce_sum(squared_deltas)
#
# #Gradient descent optimizer
# optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss=loss)
#
# #Creating a session
# sess = tensorflow.Session()
#
# writer = tensorflow.summary.FileWriter("/tmp/log/", sess.graph)
#
# #Initializing variables
# init = tensorflow.global_variables_initializer()
# sess.run(init)
#
# #Optimizing the parameters
# for i in range(1000):
#     sess.run(train, feed_dict={x: x_train, y: y_train})
#
# #Print the parameters and loss
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
# print("W : ", curr_W, ", b : ", curr_b, ", loss : ", curr_loss)
#
# writer.close()
#
# sess.close()


""" for load data
data = np.loadtxt(fname = "C:/MASc/Learned_LDBP/Fiber_optic_transmission_system/Dataset_for_Training_DNN/Dataset_for_Training_DNN/Dataset_y_X_pol_Imag.txt")
data = np.asarray(data)
num = np. floor_divide(4*len(data), 5)
print(num)
"""

"""#Weight Initialization
alp = 0.2           #attenuation
Lsp = 100
M = 2
delta = Lsp/M
beta2 = 2.175*1e-26
n = 40


# fk = np.arange(n)
# fk = 2.0*np.pi*fk/n
# fk = np.square(fk)
# Hk = -1.0*1j*fk*beta2/2.0+alp/2.0
# Hk = Hk*delta/2.0
# Hk = np.exp(Hk)
Hk = np.ones(n)
Hk = Hk*10.0
dia_Hk = np.diag(Hk)


import scipy.linalg as linalg
from scipy.linalg import dft
w = dft(n)
w_invert = linalg.inv(w)

A = np.matmul(w_invert, dia_Hk)
A = np.matmul(A, w)
print("A", A)
A_real = np.real(A)
A_real = A_real*0.1
#
# real = np.real(A)
# comp = np.imag(A)
print("A becomes", A_real)
"""

#
# N = 100
# M = 50
# import matplotlib.pyplot as plt
# # x_X_pol_Real = np.loadtxt(
# #      fname="C:/MASc/Learned_LDBP/Fiber_optic_transmission_system/Dataset_for_Training_DNN/Dataset_for_Training_DNN/Dataset_x_X_pol_Real.txt")
# # x_X_pol_Image = np.loadtxt(
# #     fname="C:/MASc/Learned_LDBP/Fiber_optic_transmission_system/Dataset_for_Training_DNN/Dataset_for_Training_DNN/Dataset_x_X_pol_Imag.txt")
# x_X_pol_Real = np.loadtxt(
#             fname="C:/MASc/Learned_LDBP/Fiber_optic_transmission_system/Dataset_for_Training_DNN/Testing_data/1_Dataset_x_X_pol_Imag.txt")[:65000]
# x_X_pol_Image = np.loadtxt(
#             fname="C:/MASc/Learned_LDBP/Fiber_optic_transmission_system/Dataset_for_Training_DNN/Testing_data/1_Dataset_x_X_pol_Imag.txt")[:65000]
# y_X_pol_Real = np.loadtxt(
#             fname="C:/MASc/Learned_LDBP/Fiber_optic_transmission_system/Dataset_for_Training_DNN/Testing_data/1_Dataset_y_X_pol_Real.txt")[:130000]
# y_X_pol_Image = np.loadtxt(
#             fname="C:/MASc/Learned_LDBP/Fiber_optic_transmission_system/Dataset_for_Training_DNN/Testing_data/1_Dataset_y_X_pol_Imag.txt")[:130000]
# print("y_X_pol_Real ", y_X_pol_Real.shape)
# print("middel", len(y_X_pol_Real))
# print("reshape", np.floor_divide(4*len(x_X_pol_Real), 5))
# # Split the Dataset to Training : Validation : Test = 8 : 1 : 1
# y_train_real = np.reshape(x_X_pol_Real[:np.floor_divide(4*len(x_X_pol_Real), 5)], (-1, M))
# y_train_image = np.reshape(x_X_pol_Image[:np.floor_divide(4*len(x_X_pol_Image), 5)], (-1, M))
# x_train_real = np.reshape(y_X_pol_Real[:np.floor_divide(4*len(y_X_pol_Real), 5)], (-1, N))
# x_train_image = np.reshape(y_X_pol_Image[:np.floor_divide(4*len(y_X_pol_Image), 5)], (-1, N))
# print("x_train_real ", x_train_real.shape)
# print("x_train_image ", x_train_image.shape)
# print("y_train_real ", y_train_real.shape)
# print("y_train_image ", y_train_image.shape)

# plt.scatter(x_train_real,x_train_image, color='red')
# plt.show()


""" Weight Matrix
import tensorflow as tf
import keras.backend as K
n = 15
k = 5

w_vector = tf.constant([1,2,3,4,5,6])
w_sliced = tf.slice(w_vector,[0], [k])
paddings = tf.constant([[0, k]])
w_pad = tf.pad(w_vector, paddings, "CONSTANT")
paddings = tf.constant([[0, k+1]])
w_sliced_pad = tf.pad(w_sliced, paddings, "CONSTANT")
w_sliced_pad = K.reverse(w_sliced_pad, axes=0)
weight = tf.math.add(w_sliced_pad, w_pad)

paddings = tf.constant([[0, n-2*k-1]])
weight = tf.pad(weight, paddings, "CONSTANT")
weight = tf.reshape(weight, [1, n])
print("weight shape", weight.shape)

# weight = tf.constant([[1,2,3,4,5,0,0,0,0,0]])
# print("weight shape", weight.shape)
out = weight
for ii in range(n-1):
    A = tf.roll(weight, shift=ii+1, axis=1)
    out = tf.concat([out, A], axis=0)



with tf.Session() as sess:
    print("haha", sess.run(out))
"""