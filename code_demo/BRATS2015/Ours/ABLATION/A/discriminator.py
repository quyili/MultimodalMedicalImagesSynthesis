# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class Discriminator:
    def __init__(self, name, ngf=64, keep_prob=1.0, output=1,output_channl=1):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.keep_prob = keep_prob
        self.output = output
        self.output_channl =output_channl

    def __call__(self, D_input):
        """
        Args:
          input: batch_size x image_size x image_size x c

        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            D_input = tf.nn.dropout(D_input, keep_prob=self.keep_prob)
            with tf.variable_scope("conv0", reuse=self.reuse):
                conv0 = tf.layers.conv2d(inputs=D_input, filters=self.ngf, kernel_size=5,
                                         strides=2,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv0')
                norm0 = ops.norm(conv0)
                relu0 = ops.relu(norm0)
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=relu0, filters=self.ngf, kernel_size=5,
                                         strides=2,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops.norm(conv1)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=2 * self.ngf, kernel_size=3,
                                         strides=2,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops.norm(conv2)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("conv3", reuse=self.reuse):
                conv3 = tf.layers.conv2d(inputs=relu2, filters=2 * self.ngf, kernel_size=3,
                                         strides=2,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv3')
                norm3 = ops.norm(conv3)
                relu3 = ops.relu(norm3)
            with tf.variable_scope("conv4_1", reuse=self.reuse):
                conv4_1 = tf.layers.conv2d(inputs=relu3, filters=self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=0.02, dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv4_1')
                norm4_1 = ops.norm(conv4_1)
                relu4_1 = ops.relu(norm4_1)
            with tf.variable_scope("conv5_1", reuse=self.reuse):
                output_1 = tf.layers.conv2d(inputs=relu4_1, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=0.0, stddev=0.02, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='conv5_1')
            if self.output == 2:
                with tf.variable_scope("conv4_2", reuse=self.reuse):
                    conv4_2 = tf.layers.conv2d(inputs=relu3, filters=self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=0.0, stddev=0.02, dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0), name='conv4_2')
                    norm4_2 = ops.norm(conv4_2)
                    relu4_2 = ops.relu(norm4_2)
                with tf.variable_scope("conv5_2", reuse=self.reuse):
                    output_2 = tf.layers.conv2d(inputs=relu4_2, filters=self.output_channl, kernel_size=3, strides=1,
                                                padding="SAME",
                                                activation=None,
                                                kernel_initializer=tf.random_normal_initializer(
                                                    mean=0.0, stddev=0.02, dtype=tf.float32),
                                                bias_initializer=tf.constant_initializer(0.0), name='conv5_2')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        if self.output == 2:
            return output_1, output_2
        else:
            return output_1
