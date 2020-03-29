# _*_ coding:utf-8 _*_
import tensorflow as tf
from ssd import Detector
from discriminator import Discriminator
from unet import Unet
import numpy as np


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 classes_size=2,
                 ngf=64,
                 ):
        """
        Args:
          input_size：list [H, W, C]
          batch_size: integer, batch size
          learning_rate: float, initial learning rate for Adam
          ngf: number of gen filters in first conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.tenaor_name = {}
        self.classes_size = classes_size

        self.G_X = Unet('G_X', ngf=ngf, output_channl=image_size[2], keep_prob=0.97)
        self.D_X = Discriminator('D_X', ngf=ngf, keep_prob=0.9)
        self.G_L_X = Detector('G_L_X', ngf, classes_size=classes_size, keep_prob=0.99, input_channl=image_size[2])

    def pred(self, classes_size, feature_class, background_classes_val, all_default_boxs_len):
        feature_class_softmax = tf.nn.softmax(logits=feature_class, dim=-1)
        background_filter = np.ones(classes_size, dtype=np.float32)
        background_filter[background_classes_val] = 0
        background_filter = tf.constant(background_filter)
        feature_class_softmax = tf.multiply(feature_class_softmax, background_filter)
        feature_class_softmax = tf.reduce_max(feature_class_softmax, 2)
        box_top_set = tf.nn.top_k(feature_class_softmax, int(all_default_boxs_len / 20))
        box_top_index = box_top_set.indices
        box_top_value = box_top_set.values
        return feature_class_softmax, box_top_index, box_top_value

    def model(self, l, x, s, m, groundtruth_class, groundtruth_location,
              groundtruth_positives, groundtruth_negatives):
        self.tenaor_name["s"] = str(s)
        self.tenaor_name["m"] = str(m)
        self.tenaor_name["l"] = str(l)
        l_onehot = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=self.classes_size),
                              shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], self.classes_size])
        new_s = s + tf.random_uniform(self.input_shape, minval=0.5, maxval=0.6,
                                      dtype=tf.float32) * (1.0 - m) * (1.0 - s)
        s_expand = tf.concat([new_s, l_onehot + 0.1], axis=-1)
        x_g = self.G_X(s_expand)
        self.tenaor_name["x_g"] = str(x_g)

        j_x_g = self.D_X(x_g)
        j_x = self.D_X(x)

        D_loss = 0.0
        G_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 2
        D_loss += self.mse_loss(j_x_g, 0.0) * 2
        G_loss += self.mse_loss(j_x_g, 1.0) * 2

        # just for pre-training
        # G_loss += self.mse_loss(x_g, x) * 5
        # G_loss += self.mse_loss(x_g * m, x * m) * 0.1

        feature_class, feature_location = self.G_L_X(x_g)
        groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                               labels=groundtruth_class)
        loss_location = tf.div(tf.reduce_sum(tf.multiply(
            tf.reduce_sum(self.smooth_L1(tf.subtract(groundtruth_location, feature_location)),
                          reduction_indices=2), groundtruth_positives), reduction_indices=1),
            tf.reduce_sum(groundtruth_positives, reduction_indices=1))
        loss_class = tf.div(
            tf.reduce_sum(tf.multiply(softmax_cross_entropy, groundtruth_count), reduction_indices=1),
            tf.reduce_sum(groundtruth_count, reduction_indices=1))
        loss_all = tf.reduce_sum(tf.add(loss_class, loss_location))

        image_list = {}
        judge_list = {}
        image_list["x_g"] = x_g
        judge_list["j_x_g"] = j_x_g
        judge_list["j_x"] = j_x

        loss_list = [G_loss, D_loss]
        detect_loss_list = [loss_all, loss_class, loss_location]

        return loss_list, detect_loss_list, image_list,  feature_class, feature_location

    def get_variables(self):
        return [self.G_X.variables,
                self.D_X.variables,
                self.G_L_X.variables]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')
        D_optimizer = make_optimizer(name='Adam_D')

        return G_optimizer, D_optimizer

    def average_precision_at_k(self, labels, predictions, k):
        return tf.metrics.average_precision_at_k( labels, predictions, k)

    def acc(self, x, y):
        correct_prediction = tf.equal(x, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def accuracy(self, labels, predictions):
        return tf.metrics.accuracy(labels, predictions)

    def auc(self, x, y):
        return tf.metrics.auc(x, y)

    def sensitivity(self, labels,  predictions, specificity):
        return tf.metrics.sensitivity_at_specificity(labels,  predictions, specificity)

    def precision(self, labels, predictions):
        return tf.metrics.precision( labels, predictions)
    def precision_at_k(self, labels, predictions,k):
        return tf.metrics.precision_at_k( labels, predictions,k)

    def recall(self, labels, predictions):
        return tf.metrics.recall( labels, predictions)
    def recall_at_k(self, labels, predictions,k):
        return tf.metrics.recall_at_k( labels, predictions,k)

    def iou(self, labels, predictions, num_classes):
        return tf.metrics.mean_iou( labels, predictions,num_classes)

    def dice_score(self, output, target, loss_type='jaccard', axis=(1, 2, 3, 4), smooth=1e-5):
        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return dice

    def cos_score(self, output, target, axis=(1, 2, 3, 4), smooth=1e-5):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(output), axis))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(target), axis))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(output, target), axis)
        score = tf.reduce_mean(tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + smooth))
        return score

    def euclidean_distance(self, output, target, axis=(1, 2, 3, 4)):
        euclidean = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(output - target), axis)))
        return euclidean

    def MSE(self, output, target):
        mse = tf.reduce_mean(tf.square(output - target))
        return mse

    def MAE(self, output, target):
        mae = tf.reduce_mean(tf.abs(output - target))
        return mae

    def mse_loss(self, x, y):
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def ssim_loss(self, x, y):
        loss = (1.0 - self.SSIM(x, y)) * 20
        return loss

    def PSNR(self, output, target):
        psnr = tf.reduce_mean(tf.image.psnr(output, target, max_val=1.0, name="psnr"))
        return psnr

    def SSIM(self, output, target):
        ssim = tf.reduce_mean(tf.image.ssim(output, target, max_val=1.0))
        return ssim

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output

    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)),
                        tf.subtract(tf.abs(x), 0.5))
