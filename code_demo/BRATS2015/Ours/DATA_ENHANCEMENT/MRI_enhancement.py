# _*_ coding:utf-8 _*_
import os
import numpy as np
import SimpleITK
from PIL import Image


def norm(input):
    output = (input - np.min(input)) / (np.max(input) - np.min(input))
    return output


def rand_rotate(arr_list):
    new_list = []
    expand = np.random.randint(0, 2)
    angle = np.random.randint(0, 360)
    for j in range(len(arr_list)):
        img_arr = arr_list[j]
        img = Image.fromarray(img_arr)
        img = img.rotate(angle, resample=0, expand=expand)
        img = img.resize(img_arr.shape)
        img_arr = np.asarray(img)
        new_list.append(img_arr)
    return new_list


def transpose(arr_list):
    new_list = []
    for j in range(len(arr_list)):
        img_arr = arr_list[j]
        img = Image.fromarray(img_arr)
        img = img.transpose(1)
        img_arr = np.asarray(img)
        new_list.append(img_arr)
    return new_list


def rand_crop(arr_list):
    new_list = []
    x_pixl_len = np.random.randint(2, 8)
    y_pixl_len = np.random.randint(2, 8)
    for j in range(len(arr_list)):
        img_arr = arr_list[j]
        img = Image.fromarray(img_arr)
        img = img.resize((img_arr.shape[0] + y_pixl_len, img_arr.shape[1] + x_pixl_len))
        subimg = img.crop([int(y_pixl_len / 2),
                           int(x_pixl_len / 2),
                           img_arr.shape[0] + int(y_pixl_len / 2),
                           img_arr.shape[1] + int(x_pixl_len / 2)])
        img_arr = np.asarray(subimg)
        new_list.append(img_arr)
    return new_list


def rand_translation(arr_list):
    new_list = []
    pixl_len = np.random.randint(2, 8)
    direction = np.random.randint(0, 4)
    for j in range(len(arr_list)):
        img_arr = arr_list[j]
        zeros_arr = np.zeros((img_arr.shape))
        if direction == 0:
            zeros_arr[:-pixl_len, :] = img_arr[pixl_len:, :]
        elif direction == 1:
            zeros_arr[pixl_len:, :] = img_arr[:-pixl_len, :]
        elif direction == 2:
            zeros_arr[:, :-pixl_len] = img_arr[:, pixl_len:]
        elif direction == 3:
            zeros_arr[:, pixl_len:] = img_arr[:, :-pixl_len]
        img_arr = zeros_arr
        new_list.append(img_arr)
    return new_list


def rand_add_noise(arr_list):
    new_list = []
    l = arr_list[-1]
    for j in range(len(arr_list) - 1):
        img_arr = arr_list[j]
        new_list.append(norm(img_arr + np.random.uniform(-0.03, 0.03, img_arr.shape)))
    new_list.append(l)
    return new_list


def rand_resize(arr_list):
    new_list = []
    x_pixl_len = np.random.randint(2, 8)
    y_pixl_len = np.random.randint(2, 8)
    for j in range(len(arr_list)):
        img_arr = arr_list[j]
        zeros_arr = np.zeros((img_arr.shape))
        zeros_img = Image.fromarray(zeros_arr)
        img = Image.fromarray(img_arr)
        img = img.resize((img_arr.shape[0] - y_pixl_len, img_arr.shape[1] - x_pixl_len))
        zeros_img.paste(img, (int(y_pixl_len / 2), int(x_pixl_len / 2)))
        img_arr = np.asarray(zeros_img)
        new_list.append(img_arr)
    return new_list


def save_image(image, name, dir="./samples", form=""):
    try:
        os.makedirs(dir)
    except os.error:
        pass
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(image), dir + "/" + name + form)


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
    img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    arr_ = SimpleITK.GetArrayFromImage(img)
    return np.asarray(arr_, dtype="float32")


def extend(
        image_size=[184, 144],
        X='../mydata/BRATS2015/trainT1',
        Y='../mydata/BRATS2015/trainT2',
        Z='../mydata/BRATS2015/trainT1c',
        W='../mydata/BRATS2015/trainFlair',
        L='../mydata/BRATS2015/trainLabel',
        save_path="../mydata/enhancement_1F_MRI",
        epoch=1
):
    try:
        os.makedirs(save_path + "/T1")
        os.makedirs(save_path + "/T2")
        os.makedirs(save_path + "/T1c")
        os.makedirs(save_path + "/Flair")
        os.makedirs(save_path + "/Label")
    except os.error:
        pass
    l_val_files = read_filename(L)
    val_index = 0
    while val_index <= len(l_val_files) * epoch:
        val_x_arr = read_file(X, l_val_files, val_index).reshape(image_size)
        val_y_arr = read_file(Y, l_val_files, val_index).reshape(image_size)
        val_z_arr = read_file(Z, l_val_files, val_index).reshape(image_size)
        val_w_arr = read_file(W, l_val_files, val_index).reshape(image_size)
        val_l_arr = read_file(L, l_val_files, val_index).reshape(image_size)
        arr_list = []
        arr_list.append(val_x_arr)
        arr_list.append(val_y_arr)
        arr_list.append(val_z_arr)
        arr_list.append(val_w_arr)
        arr_list.append(val_l_arr)

        change_times = np.random.randint(1, 3)
        extend_code = ""
        for k in range(change_times):
            method_index = np.random.rand() * 1.2
            if 0.0 <= method_index and method_index < 0.2:
                arr_list = rand_rotate(arr_list)
                extend_code += "_rotate"
            elif 0.2 <= method_index and method_index < 0.4:
                arr_list = transpose(arr_list)
                extend_code += "_transpose"
            elif 0.4 <= method_index and method_index < 0.6:
                arr_list = rand_translation(arr_list)
                extend_code += "_translation"
            elif 0.6 <= method_index and method_index < 0.8:
                arr_list = rand_add_noise(arr_list)
                extend_code += "_noise"
            elif 0.8 <= method_index and method_index <= 1.0:
                arr_list = rand_crop(arr_list)
                extend_code += "_crop"
            elif 1.0 <= method_index and method_index <= 1.2:
                arr_list = rand_resize(arr_list)
                extend_code += "_resize"

        print(str(val_index), extend_code)
        save_image(arr_list[0].astype('float32'), str(val_index) + "_" + l_val_files[val_index % len(l_val_files)],
                   dir=save_path + "/T1")
        save_image(arr_list[1].astype('float32'), str(val_index) + "_" + l_val_files[val_index % len(l_val_files)],
                   dir=save_path + "/T2")
        save_image(arr_list[2].astype('float32'), str(val_index) + "_" + l_val_files[val_index % len(l_val_files)],
                   dir=save_path + "/T1c")
        save_image(arr_list[3].astype('float32'), str(val_index) + "_" + l_val_files[val_index % len(l_val_files)],
                   dir=save_path + "/Flair")
        save_image(arr_list[4].astype('float32'), str(val_index) + "_" + l_val_files[val_index % len(l_val_files)],
                   dir=save_path + "/Label")
        val_index = val_index + 1


if __name__ == '__main__':
    extend(
        image_size=[184, 144],
        X='../mydata/BRATS2015/trainT1',
        Y='../mydata/BRATS2015/trainT2',
        Z='../mydata/BRATS2015/trainT1c',
        W='../mydata/BRATS2015/trainFlair',
        L='../mydata/BRATS2015/trainLabel',
        save_path="../mydata/enhancement_1F_MRI",
        epoch=1
    )
    extend(
        image_size=[184, 144],
        X='../mydata/BRATS2015/trainT1',
        Y='../mydata/BRATS2015/trainT2',
        Z='../mydata/BRATS2015/trainT1c',
        W='../mydata/BRATS2015/trainFlair',
        L='../mydata/BRATS2015/trainLabel',
        save_path="../mydata/enhancement_2F_MRI",
        epoch=2
    )
    extend(
        image_size=[184, 144],
        X='../mydata/BRATS2015/trainT1',
        Y='../mydata/BRATS2015/trainT2',
        Z='../mydata/BRATS2015/trainT1c',
        W='../mydata/BRATS2015/trainFlair',
        L='../mydata/BRATS2015/trainLabel',
        save_path="../mydata/enhancement_3F_MRI",
        epoch=3
    )
