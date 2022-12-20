# Test code adapted from SwinIR of KAIR (https://github.com/cszn/KAIR)

import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch

from models.network_iiTransformer import iiTransformer as net
from utils import utils_image as util
from utils import utils_option as option


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--save_path', type=str, default='results', help='result save path')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--opt', type=str, default=None, help='Path to option JSON file')

    args = parser.parse_args()
    json = option.parse(args.opt, is_train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        NotImplementedError('pretrained model path not specified')

    model = define_model(args, json)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args, json)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, border=border)
            ssim = util.calculate_ssim(output, img_gt, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            if args.task in ['jpeg_car']:
                psnr_b = util.calculate_psnrb(output, img_gt, border=border)
                test_results['psnr_b'].append(psnr_b)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
                  'PSNR_B: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))
        if args.task in ['jpeg_car']:
            ave_psnr_b = sum(test_results['psnr_b']) / len(test_results['psnr_b'])
            print('-- Average PSNR_B: {:.2f} dB'.format(ave_psnr_b))


def define_model(args, json=None):
    opt_net = json['netG']
    model = net(upscale=opt_net['upscale'],
           in_chans=opt_net['in_chans'],
           img_size=opt_net['img_size'],
           window_size=opt_net['window_size'],
           img_range=opt_net['img_range'],
           depths=opt_net['depths'],
           embed_dim=opt_net['embed_dim'],
           num_heads=opt_net['num_heads'],
           shift_sizes=opt_net['shift_sizes'] if 'shift_sizes' in opt_net.keys() else None,
           mlp_ratio=opt_net['mlp_ratio'],
           upsampler=opt_net['upsampler'],
           resi_connection=opt_net['resi_connection'],
           attn_types=opt_net['attn_type'] if 'attn_type' in opt_net.keys() else None,
           inter_pos=opt_net['inter_pos'] if 'inter_pos' in opt_net.keys() else None,
           inter_mask=opt_net['inter_mask'] if 'inter_mask' in opt_net.keys() else None)
    param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(args, json=None):
    # classical image sr
    if args.task in ['classical_sr']:
        if args.save_path == 'results':
            save_dir = os.path.join(args.save_path, f'{json["netG"]["net_type"]}_{args.task}_x{args.scale}')
        else:
            save_dir = args.save_path
        folder = args.folder_lq
        border = args.scale
        window_size = 8

    # color image denoising
    elif args.task in ['color_dn']:
        if args.save_path == 'results':
            save_dir = f'results/ii_{args.task}_noise{args.noise}'
        else:
            save_dir = args.save_path
        folder = args.folder_gt
        border = 0
        window_size = 8

    # JPEG compression artifact reduction
    elif args.task in ['jpeg_car']:
        if args.save_path == 'results':
            save_dir = f'results/ii_{args.task}_jpeg{args.jpeg}'
        else:
            save_dir = args.save_path
        folder = args.folder_gt
        border = 0
        window_size = json['netG']['window_size']

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # classical image sr (load lq-gt image pairs)
    if args.task in ['classical_sr']:
        if args.folder_gt is not None:
            img_gt = cv2.imread(f'{args.folder_gt}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
        else:
            img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # color image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ['color_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        if args.folder_lq is None:
            np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        else:
            lq_path = os.path.join(args.folder_lq, imgname+'.png')
            img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ['jpeg_car']:
        img_gt = cv2.imread(path, 0)
        if args.folder_lq is None:
            result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
            img_lq = cv2.imdecode(encimg, 0)
        else:
            lq_path = os.path.join(args.folder_lq, imgname+'.png')
            #lq_path = os.path.join(args.folder_lq, imgname+'.bmp')
            img_lq = cv2.imread(lq_path, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == '__main__':
    main()