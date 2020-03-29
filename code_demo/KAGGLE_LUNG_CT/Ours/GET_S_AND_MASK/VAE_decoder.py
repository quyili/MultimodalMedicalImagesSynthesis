# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class VDecoder:
    def __init__(self, name, ngf=64, slice_stride=2, keep_prob=1.0, output_channl=1, units=8192):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.slice_stride = slice_stride
        self.keep_prob = keep_prob
        self.output_channl = output_channl
        self.units = units

    def __call__(self, DC_input):
        """
        Args:
          input: batch_size x width x height x N
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope("dense0", reuse=self.reuse):
                dense0 = tf.layers.dense(DC_input, units=self.units, name="dense0")
            with tf.variable_scope("dense1", reuse=self.reuse):
                dense1 = tf.layers.dense(dense0, units=self.units, name="dense1")
                dense1 = tf.reshape(dense1, shape=[DC_input.get_shape().as_list()[0], 8, 8, -1])
            # 6,5
            with tf.variable_scope("conv0_1", reuse=self.reuse):
                conv0_1 = tf.layers.conv2d(inputs=dense1, filters=8 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=0.02, dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_1')
                norm0_1 = ops.norm(conv0_1)
                relu0_1 = ops.relu(norm0_1)
            # 6,5
            with tf.variable_scope("deconv0_1_r", reuse=self.reuse):
                resize0_1 = ops.uk_resize(relu0_1, reuse=self.reuse, name='resize')
                deconv0_1_r = tf.layers.conv2d(inputs=resize0_1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=0.0, stddev=0.02, dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_1_r')
                deconv0_1norm1_r = ops.norm(deconv0_1_r)
                deconv0_1_relu1 = ops.relu(deconv0_1norm1_r)
            # 12,9
            with tf.variable_scope("conv0_2", reuse=self.reuse):
                conv0_2 = tf.layers.conv2d(inputs=deconv0_1_relu1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=0.02, dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_2')
                norm0_2 = ops.norm(conv0_2)
                relu0_2 = ops.relu(norm0_2)
            # 12,9
            with tf.variable_scope("deconv0_2_r", reuse=self.reuse):
                resize0_2 = ops.uk_resize(relu0_2, reuse=self.reuse, name='resize')
                deconv0_2_r = tf.layers.conv2d(inputs=resize0_2, filters=4 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=0.0, stddev=0.02, dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_2_r')
                deconv0_2norm1_r = ops.norm(deconv0_2_r)
                deconv0_2_relu1 = ops.relu(deconv0_2norm1_r)
            # 23, 18
            with tf.variable_scope("conv0_3", reuse=self.reuse):
                conv0_3 = tf.layers.conv2d(inputs=deconv0_2_relu1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=0.02, dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_3')
                norm0_3 = ops.norm(conv0_3)
                relu0_3 = ops.relu(norm0_3)
            # 23, 18
            with tf.variable_scope("deconv0_3_r", reuse=self.reuse):
                resize0_3 = ops.uk_resize(relu0_3, reuse=self.reuse, name='resize')
                deconv0_3_r = tf.layers.conv2d(inputs=resize0_3, filters=2 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=0.0, stddev=0.02, dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_3_r')
                deconv0_3norm1_r = ops.norm(deconv0_3_r)
                add0 = ops.relu(deconv0_3norm1_r)
            # 46, 36
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=add0, filters=2 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops.norm(conv1)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("deconv1_r", reuse=self.reuse):
                resize1 = ops.uk_resize(relu1, reuse=self.reuse, name='resize')
                deconv1_r = tf.layers.conv2d(inputs=resize1, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=0.0, stddev=0.02, dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv1_r')
                deconv1norm1_r = ops.norm(deconv1_r)
                add1 = ops.relu(deconv1norm1_r)
            with tf.variable_scope("add1_conv1", reuse=self.reuse):
                add1_conv1 = tf.layers.conv2d(inputs=add1, filters=2 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=0.0, stddev=0.02, dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv1')
                add1norm1 = ops.norm(add1_conv1)
                add1_relu1 = ops.relu(add1norm1)
            with tf.variable_scope("deconv2_r", reuse=self.reuse):
                resize2 = ops.uk_resize(add1_relu1, reuse=self.reuse, name='resize')
                deconv2_r = tf.layers.conv2d(inputs=resize2, filters=self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=0.0, stddev=0.02, dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv2_r')
                deconv2norm1_r = ops.norm(deconv2_r)
                add2 = ops.relu(deconv2norm1_r)
            with tf.variable_scope("add2_conv1", reuse=self.reuse):
                add2_conv1 = tf.layers.conv2d(inputs=add2, filters=self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=0.0, stddev=0.02, dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add2_conv1')
                add2norm1 = ops.norm(add2_conv1)
                add2_relu1 = ops.relu(add2norm1)
            with tf.variable_scope("conv2", reuse=self.reuse):
                resize3 = ops.uk_resize(add2_relu1, reuse=self.reuse, name='resize')
                conv2 = tf.layers.conv2d(inputs=resize3, filters=self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops.norm(conv2)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("add3_conv1", reuse=self.reuse):
                add3_conv1 = tf.layers.conv2d(inputs=relu2, filters=self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=0.0, stddev=0.02, dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add3_conv1')
                add3norm1 = ops.norm(add3_conv1)
                add3_relu1 = ops.relu(add3norm1)
            with tf.variable_scope("lastconv", reuse=self.reuse):
                lastconv = tf.layers.conv2d(inputs=add3_relu1, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=0.0, stddev=0.02, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv')
                lastnorm = ops.norm(lastconv)
                output = tf.nn.sigmoid(lastnorm)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
