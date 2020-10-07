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