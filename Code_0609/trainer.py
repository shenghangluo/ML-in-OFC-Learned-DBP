import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import numpy as np


class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, optimizer_params):
        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        self._dataset = DatasetMNIST(val_size=10000)
        self._model = Model()
        self._optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params['lr'])
        self._writer = tf.summary.FileWriter('./summary')

    def train_model(self):
        data = self._dataset.load_data()

        # loss function     ----right now just a simple mean function
        loss = tf.reduce_mean(self._model.get_reconstruction()[0])

        tf.summary.scalar('Loss', loss)
        training_op = self._optimizer.minimize(loss)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            print("Training Start")
            for epoch in range(0, self._n_epochs):
                train_loss = 0
                for X_batch_real, y_batch_real, X_batch_image, y_batch_image in self._dataset.shuffle_batch(data['X_train_real'], data['y_train_real'], data['X_train_image'], data['y_train_image'], self._batch_size):
                    _, loss_batch = sess.run([training_op, loss], feed_dict={self._model.X_real: X_batch_real, self._model.X_image: X_batch_image})
                    train_loss += loss_batch
                    #print("alph is: ", self._model.get_para().eval())

                summary = sess.run(summary_op, feed_dict={self._model.X_real: data['X_valid_real'], self._model.X_image: data['X_valid_image']})
                self._writer.add_summary(summary=summary, global_step=epoch)
                print(epoch, "Training Loss:", train_loss, "Validation Loss",
                loss.eval(feed_dict={self._model.X_real: data['X_valid_real'], self._model.X_image: data['X_valid_image']}))







