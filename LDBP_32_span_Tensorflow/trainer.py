import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import numpy as np
import time
import matplotlib.pyplot as plt
from keras import backend as K
#learning_rate = 0.001
# phi_x_real = 0.9994
# phi_x_image = -1.0*0.0332
class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, optimizer_params):
        self._learning_rate = tf.placeholder(tf.float32, shape=[], name="learn_rate")

        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        self._dataset = DatasetMNIST(val_size=10000)
        self._model = Model()
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self._writer = tf.summary.FileWriter('./summary')

        self.out_real = self._model.get_reconstruction()[0]
        self.out_image = self._model.get_reconstruction()[1]
        #self.output = tf.concat([self.out_real, self.out_image], axis=1)


        self.Trans_real = self._model.get_Transmit()[0]
        self.Trans_image = self._model.get_Transmit()[1]

        self.loss_real = tf.math.subtract(self.Trans_real, self.out_real)
        self.loss_image = tf.math.subtract(self.Trans_image, self.out_image)
        #self.Trans = tf.concat([self.Trans_real, self.Trans_image], axis=1)

    def train_model(self):
        data = self._dataset.load_data()
        # loss function
        loss = tf.reduce_mean(tf.math.add(tf.math.square(self.loss_real), tf.math.square(self.loss_image)))

        # Q function
        x_mag = tf.reduce_mean(tf.math.add(tf.math.square(self.Trans_real), tf.math.square(self.Trans_image)))
        Q = tf.math.divide(x_mag, loss)

        # loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.Trans-self.output), axis=0))
        # # Q function
        # x_mag = tf.reduce_sum(tf.reduce_mean(tf.square(self.Trans), axis=0))
        # Q = tf.math.divide(x_mag, loss)

        tf.summary.scalar('Loss', loss)
        training_op = self._optimizer.minimize(loss)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
#        K.clear_session()
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        #sess = tf.Session()
        with tf.Session(config=tf_config) as sess:
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

            # real = self._model.get_middle_para()[0].eval(
            #     feed_dict={self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'],
            #                self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']})
            #
            # # print("predict:", self._model.get_middle_para()[1].eval(
            # #     feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
            # #                self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # image = self._model.get_middle_para()[1].eval(
            #     feed_dict={self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'],
            #                self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']})
            # print("real shape is", real.shape)
            #
            # # np.savetxt('data_real_y.csv', real, delimiter=',')
            # # np.savetxt('data_image_y.csv', image, delimiter=',')
            # plt.scatter(real[1, :1024], image[1, :1024])
            # plt.show()
            #
            #saver.restore(sess, "/data/temp/checkpoint/model.ckpt")
            #print("Model restored.")
            sess.graph.finalize()
            for epoch in range(0, self._n_epochs):
                train_loss = 0
                if epoch < 6:
                    learning_rate = 5e-4
                else:
                    learning_rate = 5e-4
#                for X_batch_real, y_batch_real, X_batch_image, y_batch_image in self._dataset.shuffle_batch(data['X_train_real'], data['y_train_real'], data['X_train_image'], data['y_train_image'], self._batch_size):
#                    _, loss_batch = sess.run([training_op, loss], feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: X_batch_real, self._model.X_image_in: X_batch_image, self._model.y_real: y_batch_real,
#                                                                             self._model.y_image: y_batch_image})
#                    train_loss += loss_batch
#                
#                summary = sess.run(summary_op, feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: X_batch_real, self._model.X_image_in: X_batch_image, self._model.y_real: y_batch_real,
#                                                                             self._model.y_image: y_batch_image})
#                self._writer.add_summary(summary=summary, global_step=epoch)
#                print(epoch, "Training Loss:", train_loss, "Validation Loss",
#                loss.eval(feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}),
#                     "Q factor:", Q.eval(feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
#                save_path = saver.save(sess, "/data/temp/checkpoint/model.ckpt")
#                print("Model saved")

            # Power
            real = self._model.get_reconstruction()[0].eval(
                feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})

            print("power = 1:", loss.eval(
                feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            image = self._model.get_reconstruction()[1].eval(
                feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})

            np.savetxt('out_real.csv', real, delimiter=',')
            np.savetxt('out_image.csv', image, delimiter=',')

            #real = self._model.after_MF()[0].eval(
            #    feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})

            #print("power = 1:", loss.eval(
            #    feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            #image = self._model.after_MF()[1].eval(
            #    feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})

#            np.savetxt('after_MF_real.csv', real, delimiter=',')
#            np.savetxt('after_MF_image.csv', image, delimiter=',')

            # real = self._model.after_sclar()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            #
            # # print("power = 1:", loss.eval(
            # #    feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # image = self._model.after_sclar()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            #
            # np.savetxt('after_sclar_real.csv', real, delimiter=',')
            # np.savetxt('after_sclar_image.csv', image, delimiter=',')
            #
            #real = self._model.Phix().eval(
            #    feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #               self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #               self._model.y_image: data['y_test_image_1']})
            #print('Phix is: ', real)
            # real = self._model.after_MF()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            # image = self._model.after_MF()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            # np.savetxt('afterMF_real.csv', real, delimiter=',')
            # np.savetxt('afterMF_image.csv', image, delimiter=',')
            #
            # real = self._model.after_down()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            # image = self._model.after_down()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            # np.savetxt('afterdown_real.csv', real, delimiter=',')
            # np.savetxt('afterdown_image.csv', image, delimiter=',')
            # real = self._model.get_w()[0].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            #
            # # print("power = 1:", loss.eval(
            # #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'], self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # image = self._model.get_w()[1].eval(
            #     feed_dict={self._learning_rate: learning_rate, self._model.X_real_in: data['X_test_real_1'],
            #                self._model.X_image_in: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'],
            #                self._model.y_image: data['y_test_image_1']})
            #
            # np.savetxt('w_real.csv', real, delimiter=',')
            # np.savetxt('w_image.csv', image, delimiter=',')


            #print("real shape:::" , real.shape)
            #plt.scatter(real[0:15, :], image[0:15, :])
            #plt.show()

            # print("Testing")
            # print("Power = 1", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # print("Power = 2", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'], self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']}))
            # print("Power = 3", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_3'], self._model.X_image: data['X_test_image_3'], self._model.y_real: data['y_test_real_3'], self._model.y_image: data['y_test_image_3']}))
            # print("Power = 4", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_4'], self._model.X_image: data['X_test_image_4'], self._model.y_real: data['y_test_real_4'], self._model.y_image: data['y_test_image_4']}))
            # print("Power = 5", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_5'], self._model.X_image: data['X_test_image_5'], self._model.y_real: data['y_test_real_5'], self._model.y_image: data['y_test_image_5']}))




