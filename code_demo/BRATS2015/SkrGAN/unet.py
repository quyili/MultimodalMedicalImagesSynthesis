# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class Unet:
    def __init__(self, name, ngf=64, slice_stride=2, keep_prob=1.0, output_channl=1):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.slice_stride = slice_stride
        self.keep_prob = keep_prob
        self.output_channl = output_channl

    def __call__(self, input1, input2=None):
        """
        Args:
          input: batch_size x width x height x c
        Returns:
          output: same size as input
        """

        with tf.variable_scope(self.name):
            input1 = tf.nn.dropout(input1, keep_prob=self.keep_prob)
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=input1, filters=self.ngf, kernel_size=3, strides=1, padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 1), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops.norm(conv1)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=self.ngf, kernel_size=3, strides=1, padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops.norm(conv2)
                relu2 = ops.relu(norm2)
            # pool1
            with tf.variable_scope("conv3", reuse=self.reuse):
                conv3 = tf.layers.conv2d(inputs=relu2, filters=self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv3')
                norm3 = ops.norm(conv3)
                relu3 = ops.relu(norm3)
            with tf.variable_scope("conv4", reuse=self.reuse):
                conv4 = tf.layers.conv2d(inputs=relu3, filters=2 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv4')
                norm4 = ops.norm(conv4)
                relu4 = ops.relu(norm4)
            with tf.variable_scope("conv5", reuse=self.reuse):
                conv5 = tf.layers.conv2d(inputs=relu4, filters=2 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv5')
                norm5 = ops.norm(conv5)
                relu5 = tf.nn.relu(norm5)
            # pool2
            with tf.variable_scope("conv6", reuse=self.reuse):
                conv6 = tf.layers.conv2d(inputs=relu5, filters=2 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv6')
                norm6 = ops.norm(conv6)
                relu6 = ops.relu(norm6)
            with tf.variable_scope("conv7", reuse=self.reuse):
                conv7 = tf.layers.conv2d(inputs=relu6, filters=4 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv7')
                norm7 = ops.norm(conv7)
                relu7 = ops.relu(norm7)
            with tf.variable_scope("conv8", reuse=self.reuse):
                conv8 = tf.layers.conv2d(inputs=relu7, filters=4 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv8')
                norm8 = ops.norm(conv8)
                relu8 = ops.relu(norm8)
            # pool3
            with tf.variable_scope("conv9", reuse=self.reuse):
                conv9 = tf.layers.conv2d(inputs=relu8, filters=4 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv9')
            # DC
            if input2 != None:
                conv9 = tf.concat([conv9, input2], axis=-1)
            with tf.variable_scope("conv10", reuse=self.reuse):
                conv10 = tf.layers.conv2d(inputs=conv9, filters=4 * self.ngf, kernel_size=3, strides=1,
                                          padding="SAME",
                                          activation=None,
                                          kernel_initializer=tf.random_normal_initializer(
                                              mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                              dtype=tf.float32),
                                          bias_initializer=tf.constant_initializer(0.0), name='conv10')
                norm10 = ops.norm(conv10)
                relu10 = ops.relu(norm10)
            with tf.variable_scope("deconv1_r", reuse=self.reuse):
                resize1 = ops.uk_resize(relu10, reuse=self.reuse, name='resize1')
                deconv1_r = tf.layers.conv2d(inputs=resize1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv1_r')
                deconv1_norm1_r = ops.norm(deconv1_r)
                deconv1 = ops.relu(deconv1_norm1_r)
            with tf.variable_scope("concat1", reuse=self.reuse):
                concat1 = tf.concat([relu8, deconv1], axis=-1)
            with tf.variable_scope("add1_conv1", reuse=self.reuse):
                add1_conv1 = tf.layers.conv2d(inputs=concat1, filters=2 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv1')
                add1_norm1 = ops.norm(add1_conv1)
                add1_relu1 = ops.relu(add1_norm1)
            with tf.variable_scope("add1_conv2", reuse=self.reuse):
                add1_conv2 = tf.layers.conv2d(inputs=add1_relu1, filters=2 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv2')
                add1_norm2 = ops.norm(add1_conv2)
                add1_relu2 = ops.relu(add1_norm2)
            with tf.variable_scope("deconv2_r", reuse=self.reuse):
                resize2 = ops.uk_resize(add1_relu2, reuse=self.reuse, name='resize2')
                deconv2_r = tf.layers.conv2d(inputs=resize2, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv2_r')
                deconv2_norm1_r = ops.norm(deconv2_r)
                deconv2 = ops.relu(deconv2_norm1_r)
            with tf.variable_scope("concat2", reuse=self.reuse):
                concat2 = tf.concat([relu5, deconv2], axis=-1)
            with tf.variable_scope("add2_conv1", reuse=self.reuse):
                add2_conv1 = tf.layers.conv2d(inputs=concat2, filters=self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add2_conv1')
                add2_norm1 = ops.norm(add2_conv1)
                add2_relu1 = ops.relu(add2_norm1)
            with tf.variable_scope("add2_conv2", reuse=self.reuse):
                add2_conv = tf.layers.conv2d(inputs=add2_relu1, filters=self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='add2_conv2')
                add2_norm2 = ops.norm(add2_conv)
                add2_relu2 = ops.relu(add2_norm2)
            with tf.variable_scope("deconv3_r", reuse=self.reuse):
                resize3 = ops.uk_resize(add2_relu2, reuse=self.reuse, name='resize3')
                deconv3_r = tf.layers.conv2d(inputs=resize3, filters=self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv3_r')
                deconv3_norm1_r = ops.norm(deconv3_r)
                deconv3 = ops.relu(deconv3_norm1_r)
            with tf.variable_scope("concat2", reuse=self.reuse):
                concat3 = tf.concat([relu2, deconv3], axis=-1)
            with tf.variable_scope("add3_conv1", reuse=self.reuse):
                add3_conv1 = tf.layers.conv2d(inputs=concat3, filters=self.ngf, kernel_size=3, strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0), name='add3_conv1')
                add3_norm1 = ops.norm(add3_conv1)
                add3_relu1 = ops.relu(add3_norm1)
            with tf.variable_scope("add3_conv2", reuse=self.reuse):
                add3_conv2 = tf.layers.conv2d(inputs=add3_relu1, filters=self.ngf, kernel_size=3, strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0), name='add3_conv2')
                add3_norm2 = ops.norm(add3_conv2)
                add3_relu2 = ops.relu(add3_norm2)
            with tf.variable_scope("lastconv", reuse=self.reuse):
                lastconv = tf.layers.conv2d(inputs=add3_relu2, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv')
                lastnorm = ops.norm(lastconv)
                output = tf.nn.sigmoid(lastnorm)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
