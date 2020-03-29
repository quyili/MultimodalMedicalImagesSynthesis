# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from feature_discriminator import FeatureDiscriminator
from VAE_encoder import VEncoder
from VAE_decoder import VDecoder
from unet import Unet


class VAE_GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 ngf=64,
                 units=4096
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
        self.ones = tf.ones(self.input_shape, name="ones")
        self.tenaor_name = {}

        self.EC_S = VEncoder('EC_S', ngf=ngf, units=units, keep_prob=0.85)
        self.DC_S = VDecoder('DC_S', ngf=ngf, output_channl=2 * image_size[2], units=units)

        self.G_M = Unet('G_M', ngf=ngf / 2, keep_prob=0.9, output_channl=2 * image_size[2])

        self.D_S = Discriminator('D_S', ngf=ngf, keep_prob=0.85)
        self.FD_Z = FeatureDiscriminator('FD_Z', ngf=ngf)

    def model(self, s, m):
        # F -> F_R VAE
        s_1 = s[:, :, :, 0:1]
        s_2 = s[:, :, :, 1:2]
        s_3 = s[:, :, :, 2:3]
        s_one_hot_1 = tf.reshape(tf.one_hot(tf.cast(s_1, dtype=tf.int32), depth=2, axis=-1),
                                 shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 2])
        s_one_hot_2 = tf.reshape(tf.one_hot(tf.cast(s_2, dtype=tf.int32), depth=2, axis=-1),
                                 shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 2])
        s_one_hot_3 = tf.reshape(tf.one_hot(tf.cast(s_3, dtype=tf.int32), depth=2, axis=-1),
                                 shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 2])
        s_one_hot = tf.reshape(tf.concat([s_one_hot_1, s_one_hot_2, s_one_hot_3], axis=-1),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2],
                                      2 * self.input_shape[3]])
        m_1 = m[:, :, :, 0:1]
        m_2 = m[:, :, :, 1:2]
        m_3 = m[:, :, :, 2:3]
        m_one_hot_1 = tf.reshape(tf.one_hot(tf.cast(m_1, dtype=tf.int32), depth=2, axis=-1),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 2])
        m_one_hot_2 = tf.reshape(tf.one_hot(tf.cast(m_2, dtype=tf.int32), depth=2, axis=-1),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 2])
        m_one_hot_3 = tf.reshape(tf.one_hot(tf.cast(m_3, dtype=tf.int32), depth=2, axis=-1),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 2])
        m_one_hot = tf.reshape(tf.concat([m_one_hot_1, m_one_hot_2, m_one_hot_3], axis=-1),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2],
                                      2 * self.input_shape[3]])

        code_f_mean, code_f_logvar = self.EC_S(s * 0.1 + 0.9)
        shape = code_f_logvar.get_shape().as_list()
        code_f_std = tf.exp(0.5 * code_f_logvar)
        code_f_epsilon = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)
        code_f = code_f_mean + tf.multiply(code_f_std, code_f_epsilon)
        s_r_prob = self.DC_S(code_f)

        # CODE_F_RM
        code_f_g = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)
        s_g_prob = self.DC_S(code_f_g)

        # D,FD
        j_s = self.D_S(s_one_hot)
        j_s_g = self.D_S(s_g_prob)

        code_f = tf.reshape(code_f, shape=[-1, 64, 64, 1])
        code_f_g = tf.reshape(code_f_g, shape=[-1, 64, 64, 1])
        j_code_f_g = self.FD_Z(code_f_g)
        j_code_f = self.FD_Z(code_f)

        m_r_prob = self.G_M(s_one_hot)
        m_g_prob = self.G_M(s_g_prob)

        D_loss = 0.0
        GS_loss = 0.0
        GM_loss = 0.0
        D_loss += self.mse_loss(j_code_f_g, 1.0) * 10
        D_loss += self.mse_loss(j_code_f, 0.0) * 10
        GS_loss += self.mse_loss(j_code_f, 1.0) * 0.001

        GS_loss += self.mse_loss(tf.reduce_mean(code_f_mean), 0.0) * 0.001
        GS_loss += self.mse_loss(tf.reduce_mean(code_f_std), 1.0) * 0.001

        D_loss += self.mse_loss(j_s, 1.0) * 25
        D_loss += self.mse_loss(j_s_g, 0.0) * 25
        GS_loss += self.mse_loss(j_s_g, 1.0) * 0.1

        GS_loss += self.mse_loss(s_one_hot, s_r_prob) * 25

        GS_loss += tf.reduce_mean(tf.abs(s_one_hot - s_r_prob)) * 10

        GS_loss += self.mse_loss(0.0, m_one_hot * s_r_prob) * 1
        GS_loss += self.mse_loss(0.0, m_g_prob * s_g_prob) * 1

        GM_loss += self.mse_loss(m_one_hot, m_r_prob) * 15

        s_r_1 = tf.reshape(tf.cast(tf.argmax(s_r_prob[:, :, :, 0:2], axis=-1), dtype=tf.float32),
                           shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        s_r_2 = tf.reshape(tf.cast(tf.argmax(s_r_prob[:, :, :, 2:4], axis=-1), dtype=tf.float32),
                           shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        s_r_3 = tf.reshape(tf.cast(tf.argmax(s_r_prob[:, :, :, 4:6], axis=-1), dtype=tf.float32),
                           shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        s_r = tf.reshape(tf.concat([s_r_1, s_r_2, s_r_3], axis=-1), shape=self.input_shape)
        s_g_1 = tf.reshape(tf.cast(tf.argmax(s_g_prob[:, :, :, 0:2], axis=-1), dtype=tf.float32),
                            shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        s_g_2 = tf.reshape(tf.cast(tf.argmax(s_g_prob[:, :, :, 2:4], axis=-1), dtype=tf.float32),
                            shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        s_g_3 = tf.reshape(tf.cast(tf.argmax(s_g_prob[:, :, :, 4:6], axis=-1), dtype=tf.float32),
                            shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        s_g = tf.reshape(tf.concat([s_g_1, s_g_2, s_g_3], axis=-1), shape=self.input_shape)

        m_r_1 = tf.reshape(tf.cast(tf.argmax(m_r_prob[:, :, :, 0:2], axis=-1), dtype=tf.float32),
                              shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        m_r_2 = tf.reshape(tf.cast(tf.argmax(m_r_prob[:, :, :, 2:4], axis=-1), dtype=tf.float32),
                              shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        m_r_3 = tf.reshape(tf.cast(tf.argmax(m_r_prob[:, :, :, 4:6], axis=-1), dtype=tf.float32),
                              shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        m_r = tf.reshape(tf.concat([m_r_1, m_r_2, m_r_3], axis=-1), shape=self.input_shape)
        m_g_1 = tf.reshape(tf.cast(tf.argmax(m_g_prob[:, :, :, 0:2], axis=-1), dtype=tf.float32),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        m_g_2 = tf.reshape(tf.cast(tf.argmax(m_g_prob[:, :, :, 2:4], axis=-1), dtype=tf.float32),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        m_g_3 = tf.reshape(tf.cast(tf.argmax(m_g_prob[:, :, :, 4:6], axis=-1), dtype=tf.float32),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 1])
        m_g = tf.reshape(tf.concat([m_g_1, m_g_2, m_g_3], axis=-1), shape=self.input_shape)

        self.tenaor_name["code_f_g"] = str(code_f_g)
        self.tenaor_name["s_g"] = str(s_g)
        self.tenaor_name["m_g"] = str(m_g)
        self.tenaor_name["j_s_g"] = str(j_s_g)

        image_list = [s, s_r, s_g, m, m_r, m_g]
        j_list = [j_code_f, j_code_f_g, j_s, j_s_g]
        loss_list = [GS_loss + GM_loss, D_loss]

        return loss_list, image_list, j_list,

    def get_variables(self):
        return  [self.EC_S.variables
                + self.DC_S.variables
                + self.G_M.variables ,
                self.D_S.variables +
                self.FD_Z.variables]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')
        D_optimizer = make_optimizer(name='Adam_D')

        return G_optimizer, D_optimizer

    def acc(self, x, y):
        correct_prediction = tf.equal(x, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def auc(self, x, y):
        return tf.metrics.auc(x, y)

    def sensitivity(self, labels, predictions, specificity):
        return tf.metrics.sensitivity_at_specificity(labels, predictions, specificity)

    def precision(self, labels, predictions):
        return tf.metrics.precision(labels, predictions)

    def precision_at_k(self, labels, predictions, k):
        return tf.metrics.precision_at_k(labels, predictions, k)

    def recall(self, labels, predictions):
        return tf.metrics.recall(labels, predictions)

    def recall_at_k(self, labels, predictions, k):
        return tf.metrics.recall_at_k(labels, predictions, k)

    def iou(self, labels, predictions, num_classes):
        return tf.metrics.mean_iou(labels, predictions, num_classes)

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