# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from unet import Unet


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 ngf=64,
                 ):
        """
           Args:
             input_size：list [N, H, W, C]
             batch_size: integer, batch size
             learning_rate: float, initial learning rate for Adam
             ngf: number of base gen filters in conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.tenaor_name = {}

        self.G_L_X = Discriminator('G_L_X', ngf=ngf, output_channl=3)

    def model(self, l, x):

        l_onehot = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=3),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 3])

        l_g_prob = self.G_L_X(x)
        l_g = tf.cast(tf.argmax(tf.reduce_mean(l_g_prob, axis=[1, 2]), axis=-1), dtype=tf.int64)

        G_loss = 0.0
        G_loss += self.mse_loss(tf.reduce_mean(l_onehot, axis=[1, 2]),
                                tf.reduce_mean(l_g_prob, axis=[1, 2])) * 0.5

        G_acc = self.acc(tf.cast(tf.reduce_mean(l, axis=[1, 2]), dtype=tf.int64), l_g)

        loss_list = [G_loss,G_acc]

        return loss_list

    def get_variables(self):
        return [self.G_L_X.variables]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')

        return G_optimizer

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

    def softmax_cross_entropy(self, onehot_labels, logits):
        return tf.losses.softmax_cross_entropy( onehot_labels, logits)
    def sparse_softmax_cross_entropy(self, labels, logits):
        return tf.losses.sparse_softmax_cross_entropy( labels, logits)

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