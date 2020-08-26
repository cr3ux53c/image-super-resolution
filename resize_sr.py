from os.path import join
import glob
import tensorflow as tf
import cv2
import os
import numpy as np


dataset_list = [
    join('results/chest_xray', 'mod3', 'test_NORMAL'),
    join('results/chest_xray', 'mod3', 'test_PNEUMONIA'),
    join('results/chest_xray', 'mod3', 'train_NORMAL'),
    join('results/chest_xray', 'mod3', 'train_PNEUMONIA'),
    join('results/chest_xray', 'mod3', 'val_NORMAL'),
    join('results/chest_xray', 'mod3', 'val_PNEUMONIA'),
]

resize_list = [
    # 224,
    256,
    700,
    1000,
]

for dataset_dir in dataset_list:
    for size in resize_list:
        data_list = glob.glob(join(dataset_dir + '_' + str(size) + '_mod3', '*.png'))
        for data_path in data_list:

            save_dir = join(os.path.dirname(data_path) + '_224')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
                
            img = cv2.imread(data_path)
            resized_img = tf.image.resize_with_pad(img, 224, 224, method=tf.image.ResizeMethod.BILINEAR).numpy().astype(np.int)
            print('Resizing img from {}px to 224 ... ({})'.format(size, data_path))

            save_path = join(save_dir, os.path.splitext(os.path.basename(data_path))[0] + '.png')
            cv2.imwrite(save_path, resized_img)