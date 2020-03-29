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

        self.G_T = Unet('G_T', ngf=ngf)
        self.D_T = Discriminator('D_T', ngf=ngf, output=2)

    def model(self,x, y, z, w):
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx, dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy, dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz, dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw, dtype=tf.int32), depth=4)

        y_t_by_x = self.G_T(x, cy_code)
        z_t_by_x = self.G_T(x, cz_code)
        w_t_by_x = self.G_T(x, cw_code)
        x_by_y_t = self.G_T(y_t_by_x, cx_code)
        x_by_z_t = self.G_T(z_t_by_x, cx_code)
        x_by_w_t = self.G_T(w_t_by_x, cx_code)

        x_t_by_y = self.G_T(y, cx_code)
        z_t_by_y = self.G_T(y, cz_code)
        w_t_by_y = self.G_T(y, cw_code)
        y_by_x_t = self.G_T(x_t_by_y, cy_code)
        y_by_z_t = self.G_T(z_t_by_y, cy_code)
        y_by_w_t = self.G_T(w_t_by_y, cy_code)

        x_t_by_z = self.G_T(z, cx_code)
        y_t_by_z = self.G_T(z, cy_code)
        w_t_by_z = self.G_T(z, cw_code)
        z_by_x_t = self.G_T(x_t_by_z, cz_code)
        z_by_y_t = self.G_T(y_t_by_z, cz_code)
        z_by_w_t = self.G_T(w_t_by_z, cz_code)

        x_t_by_w = self.G_T(w, cx_code)
        y_t_by_w = self.G_T(w, cy_code)
        z_t_by_w = self.G_T(w, cz_code)
        w_by_y_t = self.G_T(x_t_by_w, cw_code)
        w_by_z_t = self.G_T(y_t_by_w, cw_code)
        w_by_x_t = self.G_T(x_t_by_w, cw_code)

        j_t_y_by_x, j_t_y_by_x_c = self.D_T(y_t_by_x)
        j_t_z_by_x, j_t_z_by_x_c = self.D_T(z_t_by_x)
        j_t_w_by_x, j_t_w_by_x_c = self.D_T(w_t_by_x)

        j_t_x_by_y, j_t_x_by_y_c = self.D_T(x_t_by_y)
        j_t_z_by_y, j_t_z_by_y_c = self.D_T(z_t_by_y)
        j_t_w_by_y, j_t_w_by_y_c = self.D_T(w_t_by_y)

        j_t_x_by_z, j_t_x_by_z_c = self.D_T(x_t_by_z)
        j_t_y_by_z, j_t_y_by_z_c = self.D_T(y_t_by_z)
        j_t_w_by_z, j_t_w_by_z_c = self.D_T(w_t_by_z)

        j_t_x_by_w, j_t_x_by_w_c = self.D_T(x_t_by_w)
        j_t_y_by_w, j_t_y_by_w_c = self.D_T(y_t_by_w)
        j_t_z_by_w, j_t_z_by_w_c = self.D_T(z_t_by_w)

        j_x, j_x_c = self.D_T(x)
        j_y, j_y_c = self.D_T(y)
        j_z, j_z_c = self.D_T(z)
        j_w, j_w_c = self.D_T(w)

        D_loss = 0.0
        G_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 5
        D_loss += self.mse_loss(j_x, 0.0) * 3
        G_loss += self.mse_loss(j_x, 1.0) * 3

        D_loss += self.mse_loss(j_y, 1.0) * 5
        D_loss += self.mse_loss(j_y, 0.0) * 3
        G_loss += self.mse_loss(j_y, 1.0) * 3

        D_loss += self.mse_loss(j_z, 1.0) * 5
        D_loss += self.mse_loss(j_z, 0.0) * 3
        G_loss += self.mse_loss(j_z, 1.0) * 3

        D_loss += self.mse_loss(j_w, 1.0) * 5
        D_loss += self.mse_loss(j_w, 0.0) * 3
        G_loss += self.mse_loss(j_w, 1.0) * 3

        D_loss += self.mse_loss(j_x_c, cx) * 5
        D_loss += self.mse_loss(j_y_c, cy) * 5
        D_loss += self.mse_loss(j_z_c, cz) * 5
        D_loss += self.mse_loss(j_w_c, cw) * 5

        G_loss += self.mse_loss(j_t_x_by_y_c, cx) * 5
        G_loss += self.mse_loss(j_t_x_by_z_c, cx) * 5
        G_loss += self.mse_loss(j_t_x_by_w_c, cx) * 5
        G_loss += self.mse_loss(j_t_y_by_x_c, cy) * 5
        G_loss += self.mse_loss(j_t_y_by_z_c, cy) * 5
        G_loss += self.mse_loss(j_t_y_by_w_c, cy) * 5
        G_loss += self.mse_loss(j_t_z_by_x_c, cz) * 5
        G_loss += self.mse_loss(j_t_z_by_y_c, cz) * 5
        G_loss += self.mse_loss(j_t_z_by_w_c, cz) * 5
        G_loss += self.mse_loss(j_t_w_by_x_c, cw) * 5
        G_loss += self.mse_loss(j_t_w_by_y_c, cw) * 5
        G_loss += self.mse_loss(j_t_w_by_z_c, cw) * 5


        G_loss += self.mse_loss(x, x_by_y_t) * 20
        G_loss += self.mse_loss(x, x_by_z_t) * 20
        G_loss += self.mse_loss(x, x_by_w_t) * 20
        G_loss += self.mse_loss(x_t_by_y, x_t_by_z) * 5
        G_loss += self.mse_loss(x_t_by_y, x_t_by_w) * 5
        G_loss += self.mse_loss(x_t_by_z, x_t_by_w) * 5

        G_loss += self.mse_loss(y, y_by_x_t) * 20
        G_loss += self.mse_loss(y, y_by_z_t) * 20
        G_loss += self.mse_loss(y, y_by_w_t) * 20
        G_loss += self.mse_loss(y_t_by_x, y_t_by_z) * 5
        G_loss += self.mse_loss(y_t_by_x, y_t_by_w) * 5
        G_loss += self.mse_loss(y_t_by_z, y_t_by_w) * 5

        G_loss += self.mse_loss(z, z_by_x_t) * 20
        G_loss += self.mse_loss(z, z_by_y_t) * 20
        G_loss += self.mse_loss(z, z_by_w_t) * 20
        G_loss += self.mse_loss(z_t_by_x, z_t_by_y) * 5
        G_loss += self.mse_loss(z_t_by_x, z_t_by_w) * 5
        G_loss += self.mse_loss(z_t_by_y, z_t_by_w) * 5

        G_loss += self.mse_loss(w, w_by_x_t) * 20
        G_loss += self.mse_loss(w, w_by_y_t) * 20
        G_loss += self.mse_loss(w, w_by_z_t) * 20
        G_loss += self.mse_loss(w_t_by_x, w_t_by_y) * 5
        G_loss += self.mse_loss(w_t_by_x, w_t_by_z) * 5
        G_loss += self.mse_loss(w_t_by_y, w_t_by_z) * 5


        image_list={}
        judge_list={}
        image_list["x"] = x
        image_list["y"] = y
        image_list["z"] = z
        image_list["w"] = w

        judge_list["j_x"], judge_list["j_x_c"] = j_x, j_x_c
        judge_list["j_y"], judge_list["j_y_c"] = j_y, j_y_c
        judge_list["j_z"], judge_list["j_z_c"] = j_z, j_z_c
        judge_list["j_w"], judge_list["j_w_c"] = j_w, j_w_c

        loss_list = [G_loss, D_loss]

        return loss_list,image_list,judge_list

    def get_variables(self):
        return [self.G_T.variables ,
                self.D_T.variables ]

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

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output
