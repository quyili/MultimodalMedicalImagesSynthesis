# _*_ coding:utf-8 _*_
from PIL import Image
import numpy as np
import os
import SimpleITK

image_size = [184, 144]
L = '../mydata/BRATS2015/trainLabel'
L_E = '../mydata/BRATS2015/trainLabelE'
L_EV = '../mydata/BRATS2015/trainLabelEV'
change_times = 2
epoch = 5


def rand_rotate(img_arr):
    expand = np.random.randint(0, 2)
    angle = np.random.randint(0, 360)
    img = Image.fromarray(img_arr)
    img = img.rotate(angle, resample=0, expand=expand)
    img = img.resize([img_arr.shape[1], img_arr.shape[0]])
    img_arr = np.asarray(img, dtype="int32")
    return img_arr


def transpose(img_arr):
    img = Image.fromarray(img_arr)
    if np.random.randint(0, 2) == 0:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_arr = np.asarray(img, dtype="int32")
    return img_arr


def rand_translation(img_arr):
    pixl_len = np.random.randint(2, 6)
    direction = np.random.randint(0, 4)
    zeros_arr = np.zeros((img_arr.shape), dtype="int32")
    if direction == 0:
        zeros_arr[:-pixl_len, :] = img_arr[pixl_len:, :]
    elif direction == 1:
        zeros_arr[pixl_len:, :] = img_arr[:-pixl_len, :]
    elif direction == 2:
        zeros_arr[:, :-pixl_len] = img_arr[:, pixl_len:]
    elif direction == 3:
        zeros_arr[:, pixl_len:] = img_arr[:, :-pixl_len]
    img_arr = zeros_arr
    return img_arr


def rand_resize(img_arr):
    x_pixl_len = np.random.randint(2, 8)
    y_pixl_len = np.random.randint(2, 8)
    zeros_arr = np.zeros((img_arr.shape), dtype="int32")
    zeros_img = Image.fromarray(zeros_arr)
    img = Image.fromarray(img_arr)
    img = img.resize((img_arr.shape[0] - y_pixl_len, img_arr.shape[1] - x_pixl_len))
    zeros_img.paste(img, (int(y_pixl_len / 2), int(x_pixl_len / 2)))
    img_arr = np.asarray(zeros_img, dtype="int32")
    return img_arr


def read_filename(path, shuffle=False):
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
    return np.asarray(L_arr_, dtype="int32")


def main():
    try:
        os.makedirs(L_E)
    except os.error:
        pass
    try:
        os.makedirs(L_EV)
    except os.error:
        pass
    l_train_files = read_filename(L)
    index = 0
    while index <= len(l_train_files) * epoch:
        l_arr = read_file(L, l_train_files, index).reshape(image_size)
        op = ""
        for k in range(change_times):
            method_index = np.random.rand()
            if 0.0 <= method_index and method_index < 0.25:
                l_arr = rand_rotate(l_arr)
                op += "rotate+"
            elif 0.25 <= method_index and method_index < 0.5:
                l_arr = transpose(l_arr)
                op += "transpose+"
            elif 0.5 <= method_index and method_index < 0.75:
                l_arr = rand_translation(l_arr)
                op += "translation+"
            elif 0.75 <= method_index and method_index <= 1.0:
                l_arr = rand_resize(l_arr)
                op += "resize+"
        print(str(index), op[:-1])
        img = SimpleITK.GetImageFromArray(l_arr.astype('float32'))
        SimpleITK.WriteImage(img, L_E + "/" + l_train_files[index])
        img = SimpleITK.GetImageFromArray((l_arr * 0.25).astype('float32'))
        SimpleITK.WriteImage(img, L_EV + "/" + l_train_files[index])
        index = index + 1


if __name__ == '__main__':
    main()
