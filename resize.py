from os.path import join
import glob
import tensorflow as tf
import cv2
import os
import numpy as np

dataset_list = [
    # join('data/input/chest_xray', 'test', 'NORMAL'),
    # join('data/input/chest_xray', 'test', 'PNEUMONIA'),
    # join('data/input/chest_xray', 'train', 'NORMAL'),
    # join('data/input/chest_xray', 'train', 'PNEUMONIA'),
    join('data/input/chest_xray', 'val', 'NORMAL'),
    join('data/input/chest_xray', 'val', 'PNEUMONIA'),
]

resize_list = [
    224,
    # 256,
    # 700,
    # 1000,
]

for dataset_dir in dataset_list:
    data_list = glob.glob(join(dataset_dir, '*.png'))

    for size in resize_list:
        for data_path in data_list:
            if not os.path.isdir(join(os.path.dirname(data_path) + '_' + str(size))):
                os.mkdir(join(os.path.dirname(data_path) + '_' + str(size)))
                
            img = cv2.imread(data_path)
            resized_img = tf.image.resize_with_pad(img, size, size, method=tf.image.ResizeMethod.BILINEAR).numpy().astype(np.int)
            print('Resizing img to {}px ... ({})'.format(size, data_path))

            save_path = join(os.path.dirname(data_path) + '_' + str(size), os.path.splitext(os.path.basename(data_path))[0] + '.png')
            cv2.imwrite(save_path, resized_img)