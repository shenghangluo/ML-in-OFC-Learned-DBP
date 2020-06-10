# import tensorflow
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

number = 1
name = 'halle' +  str(number) +'2'
print(name)
