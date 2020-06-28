import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time


class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, optimizer_params):
        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        self._dataset = DatasetMNIST(val_size=10000)
        self._model = Model()
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params['lr'])
        # self._writer = tf.summary.FileWriter('./summary')

        # self.out_real = self._model.get_reconstruction()[0]
        # self.out_image = self._model.get_reconstruction()[1]
        # self.loss_real = tf.math.subtract(self._model.y_real, self.out_real)
        # self.loss_image = tf.math.subtract(self._model.y_image, self.out_image)

    def train_model(self):
        data = self._dataset.load_data()
        # # loss function
        # loss = tf.reduce_sum(tf.math.add(tf.math.square(self.loss_real), tf.math.square(self.loss_image)))
        #
        # # Q function
        # x_mag = tf.reduce_sum(tf.math.add(tf.math.square(self._model.y_real), tf.math.square(self._model.y_image)))
        # Q = tf.math.divide(x_mag, loss)
        #
        # tf.summary.scalar('Loss', loss)
        # training_op = self._optimizer.minimize(loss)
        # saver = tf.train.Saver()
        # summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            print("Training Start")
            writer = tf.summary.FileWriter('./graphs', sess.graph)
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

            # print("predict:", self._model.get_middle_para()[0].eval(
            #     feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
            #                self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            real = self._model.get_middle_para()[0].eval(
                feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
                           self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})

            # print("predict:", self._model.get_middle_para()[1].eval(
            #     feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
            #                self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            image = self._model.get_middle_para()[1].eval(
                feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'],
                           self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']})

            np.savetxt('data_real.csv', real, delimiter=',')
            np.savetxt('data_image.csv', image, delimiter=',')

            # for epoch in range(0, self._n_epochs):
            #     train_loss = 0
            #     for X_batch_real, y_batch_real, X_batch_image, y_batch_image in self._dataset.shuffle_batch(data['X_train_real'], data['y_train_real'], data['X_train_image'], data['y_train_image'], self._batch_size):
            #         _, loss_batch = sess.run([training_op, loss], feed_dict={self._model.X_real: X_batch_real, self._model.X_image: X_batch_image, self._model.y_real: y_batch_real, self._model.y_image: y_batch_image})
            #         train_loss += loss_batch
            #
            #     summary = sess.run(summary_op, feed_dict={self._model.X_real: data['X_valid_real_1'], self._model.X_image: data['X_valid_image_1'], self._model.y_real: data['y_valid_real_1'], self._model.y_image: data['y_valid_image_1']})
            #     self._writer.add_summary(summary=summary, global_step=epoch)
            #     print(epoch, "Training Loss:", train_loss, "Validation Loss",
            #     loss.eval(feed_dict={self._model.X_real: data['X_valid_real_2'], self._model.X_image: data['X_valid_image_2'], self._model.y_real: data['y_valid_real_2'], self._model.y_image: data['y_valid_image_2']}),
            #           "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_valid_real_1'], self._model.X_image: data['X_valid_image_1'], self._model.y_real: data['y_valid_real_1'], self._model.y_image: data['y_valid_image_1']}),
            #           "Valid Q factor", Q.eval(feed_dict={self._model.X_real: data['X_valid_real_2'], self._model.X_image: data['X_valid_image_2'], self._model.y_real: data['y_valid_real_2'], self._model.y_image: data['y_valid_image_2']}))
            #
            #     # "predict:", self._model.get_middel_para()[0].eval(
            #     #     feed_dict={self._model.X_real: data['X_valid_real'], self._model.X_image: data['X_valid_image'],
            #     #                self._model.y_real: data['y_valid_real'], self._model.y_image: data['y_valid_image']})
            #
            # print("Testing")
            # print("Power = 1", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_1'], self._model.X_image: data['X_test_image_1'], self._model.y_real: data['y_test_real_1'], self._model.y_image: data['y_test_image_1']}))
            # print("Power = 2", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_2'], self._model.X_image: data['X_test_image_2'], self._model.y_real: data['y_test_real_2'], self._model.y_image: data['y_test_image_2']}))
            # print("Power = 3", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_3'], self._model.X_image: data['X_test_image_3'], self._model.y_real: data['y_test_real_3'], self._model.y_image: data['y_test_image_3']}))
            # print("Power = 4", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_4'], self._model.X_image: data['X_test_image_4'], self._model.y_real: data['y_test_real_4'], self._model.y_image: data['y_test_image_4']}))
            # print("Power = 5", "Q factor:", Q.eval(feed_dict={self._model.X_real: data['X_test_real_5'], self._model.X_image: data['X_test_image_5'], self._model.y_real: data['y_test_real_5'], self._model.y_image: data['y_test_image_5']}))




