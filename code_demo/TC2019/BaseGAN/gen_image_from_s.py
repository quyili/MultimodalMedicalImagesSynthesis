# _*_ coding:utf-8 _*_
import tensorflow as tf
import os
import logging
import numpy as np
import SimpleITK

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir, default: None')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 4')
tf.flags.DEFINE_string('load_model',  "20200101-2020", "default: None")
tf.flags.DEFINE_list('image_size', [512, 512, 3], 'image size')
tf.flags.DEFINE_string('x_g', "GPU_3/G_X/lastconv/Sigmoid:0", "tensor name")
tf.flags.DEFINE_string('save_path', "./test_images/", "default: ./test_images/")
tf.flags.DEFINE_integer('num', 2000, 'default: 2000')


def train():
    if FLAGS.load_model is not None:
        if FLAGS.savefile is not None:
            checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")

    else:
        logging.error("<load_model> is None.")
        return
    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    model_checkpoint_path = checkpoint.model_checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    meta_graph_path = model_checkpoint_path + ".meta"
    saver = tf.train.import_meta_graph(meta_graph_path)

    graph = tf.get_default_graph()
    x_g = tf.get_default_graph().get_tensor_by_name(FLAGS.x_g)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, latest_checkpoint)
        try:
            os.makedirs(FLAGS.save_path + "x_g")
        except os.error:
            pass
        index = 0
        while index <= FLAGS.num:
            x_g_= sess.run(x_g)

            for b in range(int(FLAGS.batch_size/4)):
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x_g_)[b, :, :, :]),
                                     FLAGS.save_path + "x_g/" + str(index)+".mha")
            index += 1


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
