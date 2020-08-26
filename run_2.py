import isr
import sys
import os
from os.path import join
import numpy as np
import glob
from PIL import Image
from ISR.models import RDN, RRDN

model_list = [
    # RDN(weights='psnr-small'),
    # RDN(weights='psnr-large'),
    # RDN(weights='noise-cancel'),
    '',
    '',
    '',
    RRDN(weights='gans'),
]

dataset_list = [
    join('data', 'input', 'chest_xray', 'test', 'NORMAL'),
    join('data', 'input', 'chest_xray', 'test', 'PNEUMONIA'),
    join('data', 'input', 'chest_xray', 'train', 'NORMAL'),
    join('data', 'input', 'chest_xray', 'train', 'PNEUMONIA'),
    join('data', 'input', 'chest_xray', 'val', 'NORMAL'),
    join('data', 'input', 'chest_xray', 'val', 'PNEUMONIA'),
]

result_dir = 'results/chest_xray'

resize_list = [
    256,
    700,
    1000,
]

for mod_idx, model in enumerate(model_list):

    if mod_idx < 3: continue # mod0, mod1 스킵

    for dataset_idx, dataset_dir in enumerate(dataset_list):

        for size in resize_list:
            data_list = glob.glob(join(dataset_dir + '_' + str(size), '*.png'))

            for idx, data_path in enumerate(data_list):
                save_dir = join(result_dir, 'mod' + str(mod_idx), '_'.join(os.path.dirname(data_path).split('/')[3:5]) + '_mod' + str(mod_idx))
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                sr_img = model.predict(np.array(Image.open(data_path)), by_patch_of_size=50)
                Image.fromarray(sr_img).save(join(save_dir, os.path.splitext(os.path.basename(data_path))[0] + '.png'))
                print('Super-resolutioning... model_num: {:1}, src: {:4}/{:4} {}'.format(mod_idx, idx, len(data_list), data_path))
