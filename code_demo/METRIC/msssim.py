# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK
import os


def norm(input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output


def MSSSIM(output, target):
    ssim = tf.reduce_mean(tf.image.ssim_multiscale(norm(output), norm(target), max_val=1.0))
    return ssim


def calculate_msssim_given_paths(paths):
    graph = tf.Graph()
    with graph.as_default():
        input_x = tf.placeholder(tf.float32)
        input_y = tf.placeholder(tf.float32)
        out = MSSSIM(input_x, input_y)
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        files0 = os.listdir(paths[0])
        files1 = os.listdir(paths[1])
        assert len(files0) == len(files1)
        msssims = []
        for i in range(len(files0)):
            x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(paths[0] + "/" + files0[i])).astype('float32')
            y = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(paths[1] + "/" + files1[i])).astype('float32')
            if len(x.shape) == 2:
                msssim = sess.run(out, feed_dict={input_x: np.asarray([x]).reshape([1, x.shape[0], x.shape[1], 1]),
                                                  input_y: np.asarray([y]).reshape([1, x.shape[0], x.shape[1], 1])})
                msssims.append(msssim)
            elif x.shape[2] == 1:
                msssim = sess.run(out, feed_dict={input_x: np.asarray([x]).reshape([1, x.shape[0], x.shape[1], 1]),
                                                  input_y: np.asarray([y]).reshape([1, x.shape[0], x.shape[1], 1])})
                msssims.append(msssim)
            elif x.shape[2] == 3:
                msssim = sess.run(out, feed_dict={input_x: np.asarray([x]).reshape([1, x.shape[0], x.shape[1], 3]),
                                                  input_y: np.asarray([y]).reshape([1, x.shape[0], x.shape[1], 3])})
                msssims.append(msssim)
        return [np.mean(msssims), np.std(msssims)]
