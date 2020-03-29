# _*_ coding:utf-8 _*_
import tensorflow as tf
import os
import logging
import numpy as np
import SimpleITK
import cv2
from scipy.stats import norm

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir, default: None')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_string('load_model',  "20200101-2020", "default: None")
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_string('code_f_g', "GPU_0/random_normal_1:0", "default: None")
tf.flags.DEFINE_string('s_g', "GPU_0/Reshape_4:0", "default: None")
tf.flags.DEFINE_string('m_g', "GPU_0/Reshape_5:0", "default: None")
tf.flags.DEFINE_integer('num', 2000, ' default: 2000')


def get_mask_from_s(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    gray = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    c_list = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = c_list[-2], c_list[-1]
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)
    return np.asarray(1.0 - img / 255.0, dtype="float32")


def train():
    if FLAGS.load_model is not None:
        if FLAGS.savefile is not None:
            checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")

    else:
        print("<load_model> is None.")
        return
    try:
        os.makedirs("./test_images/S")
        os.makedirs("./test_images/M")
    except os.error:
        pass
    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    model_checkpoint_path = checkpoint.model_checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    meta_graph_path = model_checkpoint_path + ".meta"
    saver = tf.train.import_meta_graph(meta_graph_path)

    graph = tf.get_default_graph()
    code_f_g = tf.get_default_graph().get_tensor_by_name(FLAGS.code_f_g)
    s_g = tf.get_default_graph().get_tensor_by_name(FLAGS.s_g)
    m_g = tf.get_default_graph().get_tensor_by_name(FLAGS.m_g)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, latest_checkpoint)
        index = 0
        while index <= FLAGS.num:
            print("image gen start:" + str(index))
            code_f = sess.run(code_f_g)
            s, m = sess.run([s_g, m_g], feed_dict={code_f_g: code_f})

            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(s)[0, :, :, 0]),
                                 "./test_images/S/" + str(index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(m)[0, :, :, 0]),
                                 "./test_images/M/" + str(index) + ".tiff")
            print("image gen end:" + str(index))
            index += 1


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
