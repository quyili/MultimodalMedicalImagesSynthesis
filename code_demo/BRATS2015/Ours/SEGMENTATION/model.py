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
        self.code_shape = [int(batch_size / 4), int(image_size[0] / 8), int(image_size[1] / 8), 4]
        self.ones_code = tf.ones(self.code_shape, name="ones_code")

        self.G_L_X = Unet('G_L_X', ngf=ngf, output_channl=5)
        self.G_L_Y = Unet('G_L_Y', ngf=ngf, output_channl=5)
        self.G_L_Z = Unet('G_L_Z', ngf=ngf, output_channl=5)
        self.G_L_W = Unet('G_L_W', ngf=ngf, output_channl=5)

    def model(self, l, x, y, z, w):
        l_onehot = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=5),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 5])

        l_g_prob_by_x = self.G_L_X(x)
        l_g_prob_by_y = self.G_L_Y(y)
        l_g_prob_by_z = self.G_L_Z(z)
        l_g_prob_by_w = self.G_L_W(w)
        l_g_by_x = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32), shape=self.input_shape)
        l_g_by_y = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_y, axis=-1), dtype=tf.float32), shape=self.input_shape)
        l_g_by_z = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_z, axis=-1), dtype=tf.float32), shape=self.input_shape)
        l_g_by_w = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_w, axis=-1), dtype=tf.float32), shape=self.input_shape)

        G_loss = 0.0

        G_loss += self.mse_loss(l_onehot[:, :, :, 0],
                                l_g_prob_by_x[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 1],
                                  l_g_prob_by_x[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 2],
                                  l_g_prob_by_x[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 3],
                                  l_g_prob_by_x[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 4],
                                  l_g_prob_by_x[:, :, :, 4]) * 25 * 5

        G_loss += self.mse_loss(l_onehot[:, :, :, 0],
                                l_g_prob_by_y[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 1],
                                  l_g_prob_by_y[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 2],
                                  l_g_prob_by_y[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 3],
                                  l_g_prob_by_y[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 4],
                                  l_g_prob_by_y[:, :, :, 4]) * 25 * 5

        G_loss += self.mse_loss(l_onehot[:, :, :, 0],
                                l_g_prob_by_z[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 1],
                                  l_g_prob_by_z[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 2],
                                  l_g_prob_by_z[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 3],
                                  l_g_prob_by_z[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 4],
                                  l_g_prob_by_z[:, :, :, 4]) * 25 * 5

        G_loss += self.mse_loss(l_onehot[:, :, :, 0],
                                l_g_prob_by_w[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 1],
                                  l_g_prob_by_w[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 2],
                                  l_g_prob_by_w[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 3],
                                  l_g_prob_by_w[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 4],
                                  l_g_prob_by_w[:, :, :, 4]) * 25 * 5

        image_list={}
        image_list["l_g_by_x"] = l_g_by_x
        image_list["l_g_by_y"] = l_g_by_y
        image_list["l_g_by_z"] = l_g_by_z
        image_list["l_g_by_w"] = l_g_by_w

        loss_list = [G_loss]

        return loss_list,image_list

    def get_variables(self):
        return [self.G_L_X.variables +
                self.G_L_Y.variables +
                self.G_L_Z.variables +
                self.G_L_W.variables
                ]

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