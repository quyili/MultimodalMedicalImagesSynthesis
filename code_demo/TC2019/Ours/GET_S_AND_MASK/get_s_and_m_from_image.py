# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK
import cv2
import scipy.signal as signal
import os
from skimage import transform

PATH = "./X"
SAVE_F1 = "./F1"
SAVE_M1 = "./M1"
SAVE_F2 = "./F2"
SAVE_M2 = "./M2"
SAVE_F3 = "./F3"
SAVE_M3 = "./M3"
SAVE_F = "./F"
SAVE_M = "./M"
alpha = 0.0125
k_size1 = 5
p = 2
beta = 0.48
k_size2 = 3

def np_norm(input):
    output = (input - np.min(input)) / (np.max(input) - np.min(input))
    return output


def tf_norm(input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output


def gauss_2d_kernel(kernel_size=3, sigma=0.0):
    kernel = np.zeros([kernel_size, kernel_size])
    center = (kernel_size - 1) / 2
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    return kernel * sum_val


def gaussian_blur_op(image, kernel, kernel_size, cdim=3):
    outputs = []
    pad_w = (kernel_size * kernel_size - 1) // 2
    padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        g = tf.reshape(kernel, [1, -1, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        g = tf.reshape(kernel, [-1, 1, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        outputs.append(data_c)
    return tf.concat(outputs, axis=3)


def gaussian_blur(x, sigma=0.5, alpha=0.15, bin=False):
    gauss_filter = gauss_2d_kernel(3, sigma)
    gauss_filter = gauss_filter.astype(dtype=np.float32)
    y = gaussian_blur_op(x, gauss_filter, 3, cdim=1)
    if bin == True:
        y = tf.ones(y.get_shape().as_list()) * tf.cast(y > alpha, dtype="float32")
    return y


def get_s(x, j=0.1):
    x1 = tf_norm(tf.reduce_min(tf.image.sobel_edges(x), axis=-1))
    x2 = tf_norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))

    x1 = gaussian_blur(x1, sigma=0.4)
    x2 = gaussian_blur(x2, sigma=0.4)

    x1 = tf.reduce_mean(x1, axis=[1, 2, 3]) - x1
    x2 = x2 - tf.reduce_mean(x2, axis=[1, 2, 3])

    x1 = tf.ones(x1.get_shape().as_list()) * tf.cast(x1 > j, dtype="float32")
    x2 = tf.ones(x2.get_shape().as_list()) * tf.cast(x2 > j, dtype="float32")

    x12 = x1 + x2
    x12 = tf.ones(x12.get_shape().as_list()) * tf.cast(x12 > 0.0, dtype="float32")
    return x12


def get_mask(m, p=5, beta=0.0):
    m = tf_norm(m)
    mask = 1.0 - tf.ones(m.get_shape().as_list()) * tf.cast(m > beta, dtype="float32")
    shape = m.get_shape().as_list()
    mask = tf.image.resize_images(mask, size=[shape[1] + p, shape[2] + p], method=1)
    mask = tf.image.resize_image_with_crop_or_pad(mask, shape[1], shape[2])
    return mask


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1, 512, 512, 1])
    sx = get_s(x, j=alpha)
    sx = gaussian_blur(sx, sigma=0.3, alpha=0.05, bin=True)
    mask_x = get_mask(x, p=2, beta=beta)

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    try:
        os.makedirs(SAVE_F1)
        os.makedirs(SAVE_M1)
        os.makedirs(SAVE_F2)
        os.makedirs(SAVE_M2)
        os.makedirs(SAVE_F3)
        os.makedirs(SAVE_M3)
        os.makedirs(SAVE_F)
        os.makedirs(SAVE_M)
    except os.error:
        pass

    files = os.listdir(PATH)
    for i in range(len(files)):
        file = files[i]
        input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(PATH + "/" + file))
        input_x = np.asarray(input_x).reshape([512, 512, 3])

        sx1_, mask_x1_ = sess.run([sx, mask_x], feed_dict={x: np.asarray([input_x[:, :, 0:1]]).astype('float32')})
        sx1_ = signal.medfilt2d(np.asarray(sx1_)[0, :, :, 0, ], kernel_size=k_size1)
        mask_x1_ = signal.medfilt2d(np.asarray(mask_x1_)[0, :, :, 0, ], kernel_size=k_size2)

        sx2_, mask_x2_ = sess.run([sx, mask_x], feed_dict={x: np.asarray([input_x[:, :, 1:2]]).astype('float32')})
        sx2_ = signal.medfilt2d(np.asarray(sx2_)[0, :, :, 0, ], kernel_size=k_size1)
        mask_x2_ = signal.medfilt2d(np.asarray(mask_x2_)[0, :, :, 0, ], kernel_size=k_size2)

        sx3_, mask_x3_ = sess.run([sx, mask_x], feed_dict={x: np.asarray([input_x[:, :, 2:3]]).astype('float32')})
        sx3_ = signal.medfilt2d(np.asarray(sx3_)[0, :, :, 0, ], kernel_size=k_size1)
        mask_x3_ = signal.medfilt2d(np.asarray(mask_x3_)[0, :, :, 0, ], kernel_size=k_size2)

        sx_ = np.zeros([512, 512, 3])
        mask_x_ = np.zeros([512, 512, 3])
        sx_[:, :, 0] = (1.0 - mask_x1_) * sx1_
        sx_[:, :, 1] = (1.0 - mask_x2_) * sx2_
        sx_[:, :, 2] = (1.0 - mask_x3_) * sx3_
        mask_x_[:, :, 0] = mask_x1_
        mask_x_[:, :, 1] = mask_x2_
        mask_x_[:, :, 2] = mask_x3_

        NUM = file.replace(".mha", "")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x1_) * sx1_), SAVE_F1 + "/" + NUM + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x1_), SAVE_M1 + "/" + NUM + ".tiff")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x2_) * sx2_), SAVE_F2 + "/" + NUM + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x2_), SAVE_M2 + "/" + NUM + ".tiff")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x3_) * sx3_), SAVE_F3 + "/" + NUM + ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x3_), SAVE_M3 + "/" + NUM + ".tiff")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x_) * sx_), SAVE_F + "/" + NUM + ".mha")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x_), SAVE_M + "/" + NUM + ".mha")
