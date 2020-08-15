import isr
import numpy as np
from PIL import Image
from ISR.models import RDN

lr_img = np.array(Image.open('data/input/sample/baboon.png'))

model = RDN(
    weights='psnr-small'
    # weights='psnr-large'
    # weights='noise-cancel'
    # weights='gans'
)

sr_img = model.predict(lr_img, by_patch_of_size=50)
Image.fromarray(sr_img).save('test.png')
