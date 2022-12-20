## iiTransformer for Image Restoration

**iiTransformer: A Unified Approach to Exploiting Local and Non-Local Information for Image Restoration**<br />
([PDF](https://bmvc2022.mpi-inf.mpg.de/0377.pdf), 
[Supplemental](https://bmvc2022.mpi-inf.mpg.de/0377_supp.pdf), 
[Poster](https://bmvc2022.mpi-inf.mpg.de/0377_poster.pdf), 
[Video](https://bmvc2022.mpi-inf.mpg.de/0377_video.mp4))<br />
Soo Min Kang, Youngchan Song, Hanul Shin, and Tammy Lee <br />
Samsung Research <br />
<br />
This is an official repository for [iiTransformer](https://bmvc2022.mpi-inf.mpg.de/377/) (BMVC2022) (**Spotlight**).

-----

iiTransformer is a general purpose backbone architecture for image restoration, where the goal is to recover a high quality (HQ) image from its degraded low quality (LQ) input. Some examples of image restoration include image denoising, compression artifact removal, and single image super-resolution. 
![Image Restoration](/assets/img_restoration.pdf)
Our approach exploits local and non-local information by utilizing information within the local vicinity of a degraded pixel as well as the data recurrence tendency in natural images. To this end, we use Transformers to capture long-range dependencies at the pixel- or patch-level by switching between intra and inter MSA in the Transformer layer with a reshape operation. We also provide a solution to support inferencing images with arbitrary resolutions without image rescaling or applying the sliding window approach. 
![iiTransformer](/assets/iiTransformer.pdf)


## Data Preparation 
We demonstrate the effectiveness of iiTransformer on 3 restoration tasks: image denoising, compression artifact removal (CAR), and single image super-resolution (SISR). 

During training, LQ images for the image denoising task are generated on-the-fly, while LQ images of CAR and SISR are prepared before-hand. To ensure consistency across experiments during inferece time, we prepared LQ images of image denoising before-hand. We provide code to prepare LQ images of each task in the `data_preparation` folder; refer to the table below for a summary of the file corresponding to each task. For each code, specify the path of HQ/GT and target LQ path in the `SRC_PATH` and `DST_PATH` variables, respectively.

|   | Task | Script Filename | Language |
| - | ---- | --------------- | -------- |
| 1 | Image Denoising | generate_lq_nr.py | PyTorch |
| 2 | Compression Artifact Removal (CAR) | generate_lq_car.m | MATLAB |
| 3 | Single Image Super-resolution (SISR) | generate_lr.m | MATLAB |


## Training iiTransformer
To train iiTransformer, 
1. prepare configuration file in the `options` folder (for reference, see sample json files for each task in the `options` folder), then
2. run the following training code:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train.py --opt options/train_ii_denoise_color_25.json --dist True
```
Successful progress in training should create a path `{root}/{task}` according to the config file (e.g., `denoising/ii_denoise_color_25`) then further create 3 additional folders: images, models, and options. The `images` and `models` directories will contain images from the validation set and model weights, respectively, at every `checkpoint_save`; the `options` directory will contain the config file that was used to train the model.


## Testing iiTransformer
To test the effectiveness of iiTransformer, 
1. specify the path to the trained weight as `model_path`.
2. Specify the path to the config file as `json_file`.
3. Specify the path containing the LQ and HQ images as `lr_path` and `hr_path`, respectively.
4. Specify the path to save the restored images, `save_path`.
5. Specify the task as one of the following: `classical_sr`, `color_dn`, or `jpeg_car` and its corresponding degradation level. That is, 
   - if the task is `color_dn`, use the `noise` option to specify the noise level; 
   - if the task is `jpeg_car`, use the `jpeg` option to specify the compression level;
   - if the task is `classical_sr`, then use the `scale` option to specify the scale.
7. Run the following inference code:
```
python3 main_test.py --task {task} --{degradation_type} {degradation_level} --model_path ${model_path} --folder_lq ${lr_path} --folder_gt ${hr_path} --save_path ${save_path} --tile 128 --tile_overlap 64 --opt ${json_file}
```

See below for an example of full inference code restoring noisy images with AWGN of sigma=25:
```
model_path='denoising/ii_denoise_color_25/models/460000_E.pth'
json_file='denoising/ii_denoise_color_25/options/train_ii_denoise_color_25_220506_121143.json'
lr_path='DB/Kodak24/DN/s25_torch'
hr_path='DB/Kodak24/GT'
save_path='denoising/ii_denoise_color_25/results/Kodak24'

python3 main_test.py --task color_dn --noise 25 --model_path ${model_path} --folder_lq ${lr_path} --folder_gt ${hr_path} --save_path ${save_path} --tile 128 --tile_overlap 64 --opt ${json_file}
```

## Citation
Please cite our paper if you use this code or our model:
```
@inproceedings{Kang_2022_BMVC,
author    = {Soo Min Kang and Youngchan Song and Hanul Shin and Tammy Lee},
title     = {iiTransformer: A Unified Approach to Exploiting Local and Non-local Information for Image Restoration},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0377.pdf}
}
```

## Acknowledgements
This repository was built on the [Image Restoration Toolbox (PyTorch)](https://github.com/cszn/KAIR).
