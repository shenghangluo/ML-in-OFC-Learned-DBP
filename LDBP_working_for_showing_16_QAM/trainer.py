import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, optimizer_params):
        self._learning_rate = tf.placeholder(tf.float32, shape=[], name="learn_rate")

        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        self._dataset = DatasetMNIST(val_size=10000)
        self._model = Model()
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
#AdaBoundOptimizer(learning_rate=0.01, final_lr=0.01, beta1=0.9, beta2=0.999, amsbound=False)
        self._writer = tf.summary.FileWriter('./summary')

        self.out_real = self._model.get_reconstruction()[0]
        self.out_image = self._model.get_reconstruction()[1]
        self.loss_real = tf.math.subtract(self._model.y_real, self.out_real)
        self.loss_image = tf.math.subtract(self._model.y_image, self.out_image)
        print("self._model.y_real",self._model.y_real.shape)
        print("self.out_real",self.out_real.shape)

    def train_model(self):
        data = self._dataset.load_data()
        # loss function
        loss = tf.reduce_sum(tf.math.add(tf.math.square(self.loss_real), tf.math.square(self.loss_image)))

        # Q function
        x_mag = tf.reduce_sum(tf.math.add(tf.math.square(self._model.y_real), tf.math.square(self._model.y_image)))
        Q = tf.math.divide(x_mag, loss)

        tf.summary.scalar('Loss', loss)
        training_op = self._optimizer.minimize(loss)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            print("Training Start")
            # variables_names = [v.name for v in tf.trainable_variables()]
            # values = sess.run(variables_names)
            # for k, v in zip(variables_names, values):
            #     print("Variable: ", k)
            #     print("Shape: ", v.shape)
            #     print(v)

            # tvars = tf.trainable_variables()
            # tvars_vals = sess.run(tvars)
            #
            # for var, val in zip(tvars, tvars_vals):
            #     print(var.name, val)

            # w1 = self._model.get_w1().eval(feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
            #                self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})
            # print("w1 is:", w1[0,1:100])

            # real = self._model.get_middle_para()[0].eval(
            #     feed_dict={self._model.X_real: data['X_train_real'], self._model.X_image: data['X_train_image'],
            #                self._model.y_real: data['y_train_real'], self._model.y_image: data['y_train_image']})
            #
            # # print("predict:", self._model.get_middle_para()[1].eval(
            # #     feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
            # #                self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # image = self._model.get_middle_para()[1].eval(
            #     feed_dict={self._model.X_real: data['X_train_real'], self._model.X_image: data['X_train_image'],
            #                self._model.y_real: data['y_train_real'], self._model.y_image: data['y_train_image']})
            # print("real shape is", real.shape)
            #
            # # np.savetxt('data_real_x.csv', real, delimiter=',')
            # # np.savetxt('data_image_x.csv', image, delimiter=',')
            # plt.scatter(real[1, :1024], image[1, :1024])
            # plt.show()

            real = self._model.get_middle_para()[0].eval(
                feed_dict={self._model.X_real: data['X_train_real'], self._model.X_image: data['X_train_image'],
                           self._model.y_real: data['y_train_real'], self._model.y_image: data['y_train_image']})

            # print("predict:", self._model.get_middle_para()[1].eval(
            #     feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
            #                self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            image = self._model.get_middle_para()[1].eval(
                feed_dict={self._model.X_real: data['X_train_real'], self._model.X_image: data['X_train_image'],
                           self._model.y_real: data['y_train_real'], self._model.y_image: data['y_train_image']})
            print("real shape is", real.shape)

            np.savetxt('data_real_y.csv', real, delimiter=',')
            np.savetxt('data_image_y.csv', image, delimiter=',')
            plt.scatter(real[0, :1024], image[0, :1024])
            plt.show()



'''
                    _, loss_batch = sess.run([training_op, loss], feed_dict={self._learning_rate: learning_rate, self._model.X_real: X_batch_real, self._model.X_image: X_batch_image, self._model.y_real: y_batch_real, self._model.y_image: y_batch_image})
                    train_loss += loss_batch

                #summary = sess.run(summary_op, feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_train_real'], self._model.X_image: data['X_train_image'], self._model.y_real: data['y_train_real'], self._model.y_image: data['y_train_image']})
                #self._writer.add_summary(summary=summary, global_step=epoch)
                print(epoch, "Training Loss:", train_loss,#, "Validation Loss",
                 #loss.eval(feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_train_real'], self._model.X_image: data['X_train_image'], self._model.y_real: data['y_train_real'], self._model.y_image: data['y_train_image']}),
                       "Q factor:", Q.eval(feed_dict={self._learning_rate: learning_rate, self._model.X_real: X_batch_real, self._model.X_image: X_batch_image, self._model.y_real: y_batch_real, self._model.y_image: y_batch_image}))

            print("Testing Start")
            train_loss=0
            Q_total=0
            ii=1
            for X_batch_real, y_batch_real, X_batch_image, y_batch_image in self._dataset.shuffle_batch(
                    data['X_train_real'], data['y_train_real'], data['X_train_image'], data['y_train_image'],
                    self._batch_size):
                loss_batch = loss.eval(feed_dict={self._learning_rate: learning_rate, self._model.X_real: X_batch_real,
                                                  self._model.X_image: X_batch_image, self._model.y_real: y_batch_real,
                                                  self._model.y_image: y_batch_image})
                train_loss += loss_batch
            print("Testing Loss", train_loss)

            for X_batch_real, y_batch_real, X_batch_image, y_batch_image in self._dataset.shuffle_batch(
                    data['X_train_real'], data['y_train_real'], data['X_train_image'], data['y_train_image'],
                    self._batch_size):
                Q_batch = Q.eval(feed_dict={self._learning_rate: learning_rate, self._model.X_real: X_batch_real,
                                            self._model.X_image: X_batch_image, self._model.y_real: y_batch_real,
                                            self._model.y_image: y_batch_image})
                Q_total += Q_batch
                ii += 1

            print("Testing Q", Q_total/ii)

            ii=0
            Out_real = np.empty((256,32), np.float32)
            Out_image = np.empty((256,32), np.float32)
            for X_batch_real, y_batch_real, X_batch_image, y_batch_image in self._dataset.shuffle_batch(
                    data['X_train_real'], data['y_train_real'], data['X_train_image'], data['y_train_image'],
                    self._batch_size):

                real = self._model.get_reconstruction()[0].eval(
                	feed_dict={self._learning_rate: learning_rate, self._model.X_real: X_batch_real,
                                            self._model.X_image: X_batch_image, self._model.y_real: y_batch_real,
                                            self._model.y_image: y_batch_image})
                image = self._model.get_reconstruction()[1].eval(
                	feed_dict={self._learning_rate: learning_rate, self._model.X_real: X_batch_real,
                                            self._model.X_image: X_batch_image, self._model.y_real: y_batch_real,
                                            self._model.y_image: y_batch_image})
                Out_real[ii:ii+8,:] = real
                Out_image[ii:ii+8,:] = image
                ii += 8

            np.savetxt('data_real.csv', Out_real, delimiter=',')
            np.savetxt('data_image_.csv', Out_image, delimiter=',')
'''
            # # Power = 1
            # real = self._model.get_reconstruction()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})
            #
            # print("power = 1:", loss.eval(
            #     feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
            #                self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # image = self._model.get_reconstruction()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})
            #
            # np.savetxt('data_real_1.csv', real, delimiter=',')
            # np.savetxt('data_image_1.csv', image, delimiter=',')
            #
            # # Power = 2
            # real = self._model.get_reconstruction()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'], self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']})
            #
            # print("power = 2:", loss.eval(
            #     feed_dict={self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'],
            #                self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']}))
            # image = self._model.get_reconstruction()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'], self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']})
            # # print("real shape is", real.shape)
            #
            # np.savetxt('data_real_2.csv', real, delimiter=',')
            # np.savetxt('data_image_2.csv', image, delimiter=',')
            #
            # # Power = 3
            # real = self._model.get_reconstruction()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_3'], self._model.X_image: data['X_test_image_3'], self._model.y_real: data['y_test_real_3'], self._model.y_image: data['y_test_image_3']})
            #
            # print("power = 3:", loss.eval(
            #     feed_dict={self._model.X_real: data['X_test_real_3'], self._model.X_image: data['X_test_image_3'],
            #                self._model.y_real: data['y_test_real_3'], self._model.y_image: data['y_test_image_3']}))
            # image = self._model.get_reconstruction()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_3'], self._model.X_image: data['X_test_image_3'], self._model.y_real: data['y_test_real_3'], self._model.y_image: data['y_test_image_3']})
            # # print("real shape is", real.shape)
            #
            # np.savetxt('data_real_3.csv', real, delimiter=',')
            # np.savetxt('data_image_3.csv', image, delimiter=',')
            #
            # # Power = 4
            # real = self._model.get_reconstruction()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_4'], self._model.X_image: data['X_test_image_4'], self._model.y_real: data['y_test_real_4'], self._model.y_image: data['y_test_image_4']})
            #
            # print("power = 4:", loss.eval(
            #     feed_dict={self._model.X_real: data['X_test_real_4'], self._model.X_image: data['X_test_image_4'],
            #                self._model.y_real: data['y_test_real_4'], self._model.y_image: data['y_test_image_4']}))
            # image = self._model.get_reconstruction()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_4'], self._model.X_image: data['X_test_image_4'], self._model.y_real: data['y_test_real_4'], self._model.y_image: data['y_test_image_4']})
            # # print("real shape is", real.shape)
            #
            # np.savetxt('data_real_4.csv', real, delimiter=',')
            # np.savetxt('data_image_4.csv', image, delimiter=',')
            #
            # # Power = 5
            # real = self._model.get_reconstruction()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_5'], self._model.X_image: data['X_test_image_5'],
            #                self._model.y_real: data['y_test_real_5'], self._model.y_image: data['y_test_image_5']})
            #
            # print("power = 5:", loss.eval(
            #     feed_dict={self._model.X_real: data['X_test_real_5'], self._model.X_image: data['X_test_image_5'],
            #                self._model.y_real: data['y_test_real_5'], self._model.y_image: data['y_test_image_5']}))
            # image = self._model.get_reconstruction()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real: data['X_test_real_5'], self._model.X_image: data['X_test_image_5'],
            #                self._model.y_real: data['y_test_real_5'], self._model.y_image: data['y_test_image_5']})
            # # print("real shape is", real.shape)
            #
            # np.savetxt('data_real_5.csv', real, delimiter=',')
            # np.savetxt('data_image_5.csv', image, delimiter=',')
            # plt.scatter(real[1, :1024], image[1, :1024])
            # plt.show()
            # print("Testing")
            # print("Power = 1", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # print("Power = 2", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'], self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']}))
            # print("Power = 3", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_3'], self._model.X_image: data['X_test_image_3'], self._model.y_real: data['y_test_real_3'], self._model.y_image: data['y_test_image_3']}))
            # print("Power = 4", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_4'], self._model.X_image: data['X_test_image_4'], self._model.y_real: data['y_test_real_4'], self._model.y_image: data['y_test_image_4']}))
            # print("Power = 5", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_5'], self._model.X_image: data['X_test_image_5'], self._model.y_real: data['y_test_real_5'], self._model.y_image: data['y_test_image_5']}))




