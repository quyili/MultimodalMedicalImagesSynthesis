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
tf.flags.DEFINE_string('load_model', "20200101-2020", "default: None")
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size')
tf.flags.DEFINE_string('S', '../../../../data/BRATS2015/test/S', 'files path')
tf.flags.DEFINE_string('M', '../../../../data/BRATS2015/test/M', 'files path')
tf.flags.DEFINE_string('x_g', "GPU_3/G_X/lastconv/Sigmoid:0", "tensor name")
tf.flags.DEFINE_string('y_g', "GPU_3/G_X/lastconv/Sigmoid:0", "tensor name")
tf.flags.DEFINE_string('z_g', "GPU_3/G_X/lastconv/Sigmoid:0", "tensor name")
tf.flags.DEFINE_string('w_g', "GPU_3/G_X/lastconv/Sigmoid:0", "tensor name")
tf.flags.DEFINE_string('s', "GPU_3/Placeholder_4:0", "tensor name")
tf.flags.DEFINE_string('m', "GPU_3/Placeholder_5:0", "tensor name")
tf.flags.DEFINE_string('save_path', "./test_images/", "default: ./test_images/")


def read_filename(path, shuffle=True):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


def read_file(l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    L_arr_ = L_arr_.astype('float32')
    return np.asarray(L_arr_)


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
    s = graph.get_tensor_by_name(FLAGS.s)
    m = graph.get_tensor_by_name(FLAGS.m)
    x_g = tf.get_default_graph().get_tensor_by_name(FLAGS.x_g)
    y_g = tf.get_default_graph().get_tensor_by_name(FLAGS.y_g)
    z_g = tf.get_default_graph().get_tensor_by_name(FLAGS.z_g)
    w_g = tf.get_default_graph().get_tensor_by_name(FLAGS.w_g)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, latest_checkpoint)
        try:
            os.makedirs(FLAGS.save_path + "x_g")
            os.makedirs(FLAGS.save_path + "y_g")
            os.makedirs(FLAGS.save_path + "z_g")
            os.makedirs(FLAGS.save_path + "w_g")
        except os.error:
            pass

        s_train_files = read_filename(FLAGS.S)
        index = 0
        while index <= len(s_train_files):
            train_true_s = []
            train_true_m = []
            for b in range(int(FLAGS.batch_size/4)):
                train_s_arr_ = read_file(FLAGS.S, s_train_files, index).reshape(FLAGS.image_size)
                train_m_arr_ = read_file(FLAGS.M, s_train_files, index).reshape(FLAGS.image_size)
                train_true_s.append(train_s_arr_)
                train_true_m.append(train_m_arr_)
                index = index + 1

            x_g_, y_g_, z_g_, w_g_ = sess.run([x_g, y_g, z_g, w_g],
                                                feed_dict={s: np.asarray(train_true_s),
                                                           m: np.asarray(train_true_m)})

            for b in range(int(FLAGS.batch_size/4)):
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x_g_)[b, :, :, 0]),
                                     FLAGS.save_path + "x_g/" + str(index)+".tiff")
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y_g_)[b, :, :, 0]),
                                     FLAGS.save_path + "y_g/" + str(index)+".tiff")
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(z_g_)[b, :, :, 0]),
                                     FLAGS.save_path + "z_g/" + str(index)+".tiff")
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(w_g_)[b, :, :, 0]),
                                     FLAGS.save_path + "w_g/" + str(index)+".tiff")
            index += 1


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
