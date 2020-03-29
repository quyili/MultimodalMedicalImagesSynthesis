# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class VEncoder:
    def __init__(self, name, ngf=64, slice_stride=2, keep_prob=1.0, units=8192):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.slice_stride = slice_stride
        self.keep_prob = keep_prob
        self.units = units

    def __call__(self, EC_input):
        """
        Args:
          input: batch_size x width x height x C
        Returns:
          output: same size as input
        """

        with tf.variable_scope(self.name):
            EC_input = tf.nn.dropout(EC_input, keep_prob=self.keep_prob)
            with tf.variable_scope("conv0", reuse=self.reuse):
                conv0 = tf.layers.conv2d(inputs=EC_input, filters=self.ngf, kernel_size=5,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv0')
                norm0 = ops.norm(conv0)
                relu0 = ops.relu(norm0)
            # pool1
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=relu0, filters=self.ngf, kernel_size=5,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops.norm(conv1)
                relu1 = ops.relu(norm1)
            # w/2,h/2
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=2 * self.ngf, kernel_size=3,
                                         strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops.norm(conv2)
                relu2 = ops.relu(norm2)
            # pool2
            with tf.variable_scope("conv3", reuse=self.reuse):
                conv3 = tf.layers.conv2d(inputs=relu2, filters=4 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv3')
                norm3 = ops.norm(conv3)
                relu3 = ops.relu(norm3)
            # w/4,h/4
            with tf.variable_scope("conv4", reuse=self.reuse):
                conv4 = tf.layers.conv2d(inputs=relu3, filters=4 * self.ngf, kernel_size=3,
                                         strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv4')
                norm4 = ops.norm(conv4)
                relu4 = ops.relu(norm4)
            with tf.variable_scope("conv5", reuse=self.reuse):
                conv5 = tf.layers.conv2d(inputs=relu4, filters=4 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv5')
                norm5 = ops.norm(conv5)
                relu5 = ops.relu(norm5)
            # pool3
            with tf.variable_scope("conv6", reuse=self.reuse):
                conv6 = tf.layers.conv2d(inputs=relu5, filters=4 * self.ngf, kernel_size=3,
                                         strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv6')
                norm6 = ops.norm(conv6)
                relu6 = ops.relu(norm6)
            # w/8,h/8 18 23
            with tf.variable_scope("conv7", reuse=self.reuse):
                conv7 = tf.layers.conv2d(inputs=relu6, filters=8 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv7')
                norm7 = ops.norm(conv7)
                relu7 = ops.relu(norm7)
            # pool4
            with tf.variable_scope("conv8", reuse=self.reuse):
                conv8 = tf.layers.conv2d(inputs=relu7, filters=8 * self.ngf, kernel_size=3,
                                         strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv8')
                norm8 = ops.norm(conv8)
                relu8 = ops.relu(norm8)
            # 9 12
            with tf.variable_scope("conv9", reuse=self.reuse):
                conv9 = tf.layers.conv2d(inputs=relu8, filters=8 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv9')
                norm9 = ops.norm(conv9)
                relu9 = ops.relu(norm9)
            # pool5
            with tf.variable_scope("conv10", reuse=self.reuse):
                conv10 = tf.layers.conv2d(inputs=relu9, filters=2 * self.ngf, kernel_size=3,
                                          strides=1,
                                          padding="SAME",
                                          activation=None,
                                          kernel_initializer=tf.random_normal_initializer(
                                              mean=0.0, stddev=0.02, dtype=tf.float32),
                                          bias_initializer=tf.constant_initializer(0.0), name='conv10')
                norm10 = ops.norm(conv10)
                relu10 = ops.relu(norm10)
                conv_output = tf.layers.flatten(relu10)
            # 5 6
            with tf.variable_scope("dense1", reuse=self.reuse):
                mean = tf.layers.dense(conv_output, units=self.units, name="dense1")
            with tf.variable_scope("dense2", reuse=self.reuse):
                log_var = tf.layers.dense(conv_output, units=self.units, name="dense2")

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return mean, log_var
