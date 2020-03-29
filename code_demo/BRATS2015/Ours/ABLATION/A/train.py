# _*_ coding:utf-8 _*_
import tensorflow as tf
from model import GAN
from datetime import datetime
import os
import logging
import numpy as np
import SimpleITK

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir, default: None')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 4')
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size,')
tf.flags.DEFINE_float('learning_rate', 1e-5, 'initial learning rate for Adam, default: 1e-5')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', './data/BRATS2015/train/T1', 'files path')
tf.flags.DEFINE_string('Y', './data/BRATS2015/train/T2', 'files path')
tf.flags.DEFINE_string('Z', './data/BRATS2015/train/T1c', 'files path')
tf.flags.DEFINE_string('W', './data/BRATS2015/train/Flair', 'files path')
tf.flags.DEFINE_string('X_test', './data/BRATS2015/test/T1', 'files path')
tf.flags.DEFINE_string('Y_test', './data/BRATS2015/test/T2', 'files path')
tf.flags.DEFINE_string('Z_test', './data/BRATS2015/test/T1c', 'files path')
tf.flags.DEFINE_string('W_test', './data/BRATS2015/test/Flair', 'files path')
tf.flags.DEFINE_string('load_model', None,'e.g. 20200101-2020, default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False, 'if continue training, step clear, default: False')
tf.flags.DEFINE_integer('epoch', 200, 'default: 200')


def read_file(l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    L_arr_ = L_arr_.astype('float32')
    return np.asarray(L_arr_)


def read_filename(path, shuffle=True):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


def average_gradients(grads_list):
    average_grads = []
    for grad_and_vars in zip(*grads_list):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.device("/cpu:0"):
        if FLAGS.load_model is not None:
            if FLAGS.savefile is not None:
                checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
            else:
                checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            if FLAGS.savefile is not None:
                checkpoints_dir = FLAGS.savefile + "/checkpoints/{}".format(current_time)
            else:
                checkpoints_dir = "checkpoints/{}".format(current_time)
            try:
                os.makedirs(checkpoints_dir + "/samples")
            except os.error:
                pass

        for attr, value in FLAGS.flag_values_dict().items():
            logging.info("%s\t:\t%s" % (attr, str(value)))

        graph = tf.Graph()
        with graph.as_default():
            gan = GAN(FLAGS.image_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.ngf)
            input_shape = [int(FLAGS.batch_size / 4), FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]]
            G_optimizer, D_optimizer = gan.optimize()

            G_grad_list = []
            D_grad_list = []
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device("/gpu:0"):
                    with tf.name_scope("GPU_0"):
                        x_0 = tf.placeholder(tf.float32, shape=input_shape)
                        y_0 = tf.placeholder(tf.float32, shape=input_shape)
                        z_0 = tf.placeholder(tf.float32, shape=input_shape)
                        w_0 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_0,image_list_0,judge_list_0 = gan.model(x_0, y_0, z_0, w_0)
                        variables_list_0 = gan.get_variables()
                        G_grad_0 = G_optimizer.compute_gradients(loss_list_0[0], var_list=variables_list_0[0])
                        D_grad_0 = D_optimizer.compute_gradients(loss_list_0[1], var_list=variables_list_0[1])
                        G_grad_list.append(G_grad_0)
                        D_grad_list.append(D_grad_0)
                with tf.device("/gpu:1"):
                    with tf.name_scope("GPU_1"):
                        x_1 = tf.placeholder(tf.float32, shape=input_shape)
                        y_1 = tf.placeholder(tf.float32, shape=input_shape)
                        z_1 = tf.placeholder(tf.float32, shape=input_shape)
                        w_1 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_1,image_list_1,judge_list_1 = gan.model(x_1, y_1, z_1, w_1)
                        variables_list_1 = gan.get_variables()
                        G_grad_1 = G_optimizer.compute_gradients(loss_list_1[0], var_list=variables_list_1[0])
                        D_grad_1 = D_optimizer.compute_gradients(loss_list_1[1], var_list=variables_list_1[1])
                        G_grad_list.append(G_grad_1)
                        D_grad_list.append(D_grad_1)
                with tf.device("/gpu:2"):
                    with tf.name_scope("GPU_2"):
                        x_2 = tf.placeholder(tf.float32, shape=input_shape)
                        y_2 = tf.placeholder(tf.float32, shape=input_shape)
                        z_2 = tf.placeholder(tf.float32, shape=input_shape)
                        w_2 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_2,image_list_2,judge_list_2  = gan.model(x_2, y_2, z_2, w_2)
                        variables_list_2= gan.get_variables()
                        G_grad_2 = G_optimizer.compute_gradients(loss_list_2[0], var_list=variables_list_2[0])
                        D_grad_2 = D_optimizer.compute_gradients(loss_list_2[1], var_list=variables_list_2[1])
                        G_grad_list.append(G_grad_2)
                        D_grad_list.append(D_grad_2)
                with tf.device("/gpu:3"):
                    with tf.name_scope("GPU_3"):
                        x_3 = tf.placeholder(tf.float32, shape=input_shape)
                        y_3 = tf.placeholder(tf.float32, shape=input_shape)
                        z_3 = tf.placeholder(tf.float32, shape=input_shape)
                        w_3 = tf.placeholder(tf.float32, shape=input_shape)
                        loss_list_3,image_list_3,judge_list_3 = gan.model(x_3, y_3, z_3, w_3)
                        tensor_name_dirct = gan.tenaor_name
                        variables_list_3 = gan.get_variables()
                        G_grad_3 = G_optimizer.compute_gradients(loss_list_3[0], var_list=variables_list_3[0])
                        D_grad_3 = D_optimizer.compute_gradients(loss_list_3[1], var_list=variables_list_3[1])
                        G_grad_list.append(G_grad_3)
                        D_grad_list.append(D_grad_3)

            G_ave_grad = average_gradients(G_grad_list)
            D_ave_grad = average_gradients(D_grad_list)
            G_optimizer_op = G_optimizer.apply_gradients(G_ave_grad)
            D_optimizer_op = D_optimizer.apply_gradients(D_ave_grad)
            optimizers = [G_optimizer_op, D_optimizer_op]

            saver = tf.train.Saver()

        with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if FLAGS.load_model is not None:
                logging.info("restore model:" + FLAGS.load_model)
                if FLAGS.checkpoint is not None:
                    model_checkpoint_path = checkpoints_dir + "/model.ckpt-" + FLAGS.checkpoint
                    latest_checkpoint = model_checkpoint_path
                else:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    model_checkpoint_path = checkpoint.model_checkpoint_path
                    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
                logging.info("model checkpoint path:" + model_checkpoint_path)
                meta_graph_path = model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, latest_checkpoint)
                if FLAGS.step_clear == True:
                    step = 0
                else:
                    step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                step = 0

            sess.graph.finalize()
            logging.info("start step:" + str(step))

            try:
                logging.info("tensor_name_dirct:\n" + str(tensor_name_dirct))
                x_train_files = read_filename(FLAGS.X)
                index = 0
                epoch = 0
                while epoch <= FLAGS.epoch:

                    train_true_x = []
                    train_true_y = []
                    train_true_z = []
                    train_true_w = []
                    for b in range(FLAGS.batch_size):
                        train_x_arr = read_file(FLAGS.X, x_train_files, index).reshape(FLAGS.image_size)
                        train_y_arr = read_file(FLAGS.Y, x_train_files, index).reshape(FLAGS.image_size)
                        train_z_arr = read_file(FLAGS.Z, x_train_files, index).reshape(FLAGS.image_size)
                        train_w_arr = read_file(FLAGS.W, x_train_files, index).reshape(FLAGS.image_size)

                        train_true_x.append(train_x_arr)
                        train_true_y.append(train_y_arr)
                        train_true_z.append(train_z_arr)
                        train_true_w.append(train_w_arr)

                        epoch = int(index / len(x_train_files))
                        index = index + 1

                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                    sess.run(optimizers,
                             feed_dict={
                                 x_0: np.asarray(train_true_x)[0*int(FLAGS.batch_size / 4):1*int(FLAGS.batch_size / 4), :, :, :],
                                 y_0: np.asarray(train_true_y)[0*int(FLAGS.batch_size / 4):1*int(FLAGS.batch_size / 4), :, :, :],
                                 z_0: np.asarray(train_true_z)[0*int(FLAGS.batch_size / 4):1*int(FLAGS.batch_size / 4), :, :, :],
                                 w_0: np.asarray(train_true_w)[0*int(FLAGS.batch_size / 4):1*int(FLAGS.batch_size / 4), :, :, :],

                                 x_1: np.asarray(train_true_x)[1*int(FLAGS.batch_size / 4):2*int(FLAGS.batch_size / 4), :, :, :],
                                 y_1: np.asarray(train_true_y)[1*int(FLAGS.batch_size / 4):2*int(FLAGS.batch_size / 4), :, :, :],
                                 z_1: np.asarray(train_true_z)[1*int(FLAGS.batch_size / 4):2*int(FLAGS.batch_size / 4), :, :, :],
                                 w_1: np.asarray(train_true_w)[1*int(FLAGS.batch_size / 4):2*int(FLAGS.batch_size / 4), :, :, :],

                                 x_2: np.asarray(train_true_x)[2*int(FLAGS.batch_size / 4):3*int(FLAGS.batch_size / 4), :, :, :],
                                 y_2: np.asarray(train_true_y)[2*int(FLAGS.batch_size / 4):3*int(FLAGS.batch_size / 4), :, :, :],
                                 z_2: np.asarray(train_true_z)[2*int(FLAGS.batch_size / 4):3*int(FLAGS.batch_size / 4), :, :, :],
                                 w_2: np.asarray(train_true_w)[2*int(FLAGS.batch_size / 4):3*int(FLAGS.batch_size / 4), :, :, :],

                                 x_3: np.asarray(train_true_x)[3*int(FLAGS.batch_size / 4):4*int(FLAGS.batch_size / 4), :, :, :],
                                 y_3: np.asarray(train_true_y)[3*int(FLAGS.batch_size / 4):4*int(FLAGS.batch_size / 4), :, :, :],
                                 z_3: np.asarray(train_true_z)[3*int(FLAGS.batch_size / 4):4*int(FLAGS.batch_size / 4), :, :, :],
                                 w_3: np.asarray(train_true_w)[3*int(FLAGS.batch_size / 4):4*int(FLAGS.batch_size / 4), :, :, :],
                             })
                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")

                    step += 1
            except Exception as e:
                logging.info("ERROR:" + str(e))
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
