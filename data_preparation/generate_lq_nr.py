from glob import glob
import itertools
import os
import cv2
import numpy as np
import torch
from tifffile import imread

def add_noise(hq_path, dst_path, sigma=10, color=False):

    img_name = hq_path.split('/')[-1]  # e.g., hq_path = '/DIV2K/train/0801.png'
    img_name = img_name.split('.')[0]  # e.g., input: 0801.png; output: 0801

    nch = 3 if color else 1

    # Get image
    _, ext = os.path.splitext(hq_path)
    if ext.lower() in ['.tif', '.tiff']:
        hq = imread(hq_path)
    else:
        hq = cv2.imread(hq_path)  # BGR
    hq = hq[:, :, :nch]
    height, width, _ = hq.shape

    # Convert numpy to torch
    hq = torch.from_numpy(np.float32(hq) / 255.)  # (H, W, C)

    # Add white gaussian noise
    noise = torch.normal(0., sigma/255., size=hq.shape, dtype=torch.float32)
    lq = hq + noise
    lq = torch.clamp(lq, 0, 1)

    # Convert torch to numpy
    lq = np.uint8(np.clip(np.round(255*lq.numpy() ), 0, 255))

    # Write degraded image
    if ext.lower() in ['.tif', '.tiff']:
        lq = lq[:, :, ::-1]  # RGB -> BGR
    cv2.imwrite(os.path.join(dst_path, f'{img_name}.png'), lq)

    return None


if __name__ == '__main__':
    db_name = 'LIVE1'
    degType = 'AWGN'
    sigmas = [10, 15, 25, 30, 50]
    color = True

    # Specify path of input
    SRC_PATH = 'DB/LIVE1/GT/'

    for sigma in sigmas:
        # Specify path of output
        if color:
            folder_name = f'awgn_s{sigma}_torch'
        else:
            folder_name = f'awgn_s{sigma}_gray_torch'
        DST_PATH = os.path.join('DB/LIVE1/LQ/', folder_name)

        # Get list of images
        file_flist = [glob(os.path.join(SRC_PATH, e)) for e in ['*.png', '*.jpg', '*.bmp', '*.tif']]
        file_flist = list(itertools.chain.from_iterable(file_flist)) # concatenate nested-lists
        file_flist.sort()

        # Create directory if it doesn't exist
        os.makedirs(DST_PATH, exist_ok=True)

        for file in file_flist:
            add_noise(file, DST_PATH, sigma=sigma, color=color)
