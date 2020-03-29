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

        self.G_X = Unet('G_X', ngf=ngf)
        self.D_X = Discriminator('D_X', ngf=ngf)
        self.G_Y = Unet('G_Y', ngf=ngf)
        self.D_Y = Discriminator('D_Y', ngf=ngf)
        self.G_Z = Unet('G_Z', ngf=ngf)
        self.D_Z = Discriminator('D_Z', ngf=ngf)
        self.G_W = Unet('G_W', ngf=ngf)
        self.D_W = Discriminator('D_W', ngf=ngf)

    def model(self, x, y, z, w):
        x_g = self.G_X(tf.random_normal(self.input_shape, mean=0., stddev=1., dtype=tf.float32))
        y_g = self.G_Y(tf.random_normal(self.input_shape, mean=0., stddev=1., dtype=tf.float32))
        z_g = self.G_Z(tf.random_normal(self.input_shape, mean=0., stddev=1., dtype=tf.float32))
        w_g = self.G_W(tf.random_normal(self.input_shape, mean=0., stddev=1., dtype=tf.float32))

        self.tenaor_name["x_g"] = str(x_g)
        self.tenaor_name["y_g"] = str(y_g)
        self.tenaor_name["z_g"] = str(z_g)
        self.tenaor_name["w_g"] = str(w_g)

        j_x_g = self.D_X(x_g)
        j_y_g = self.D_Y(y_g)
        j_z_g = self.D_Z(z_g)
        j_w_g = self.D_W(w_g)

        j_x = self.D_X(x)
        j_y = self.D_Y(y)
        j_z = self.D_Z(z)
        j_w = self.D_W(w)

        D_loss = 0.0
        G_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_x_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_x_g, 1.0) * 35 * 2

        D_loss += self.mse_loss(j_y, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_y_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_y_g, 1.0) * 35 * 2

        D_loss += self.mse_loss(j_z, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_z_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_z_g, 1.0) * 35 * 2

        D_loss += self.mse_loss(j_w, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_w_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_w_g, 1.0) * 35 * 2

        image_list={}
        image_list["x_g"] = x_g
        image_list["y_g"] = y_g
        image_list["z_g"] = z_g
        image_list["w_g"] = w_g

        judge_list={}
        judge_list["j_x_g"] = j_x_g
        judge_list["j_y_g"] = j_y_g
        judge_list["j_z_g"] = j_z_g
        judge_list["j_w_g"] = j_w_g

        judge_list["j_x"] = j_x
        judge_list["j_y"] = j_y
        judge_list["j_z"] = j_z
        judge_list["j_w"] = j_w

        loss_list = [G_loss, D_loss]

        return loss_list,image_list,judge_list

    def get_variables(self):
        return [self.G_X.variables +
                self.G_Y.variables +
                self.G_Z.variables +
                self.G_W.variables
            ,
                self.D_X.variables +
                self.D_Y.variables +
                self.D_Z.variables +
                self.D_W.variables
                ]

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