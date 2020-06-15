from __future__ import print_function
import argparse
from time import perf_counter
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np


def inference(input_image, model, cuda=False):
    """ inference function """
    img = input_image.convert('YCbCr')
    y, cb, cr = img.split()

    model = torch.load(model)
    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    if cuda:
        model = model.cuda()
        input = input.cuda()

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    return out_img


if __name__ == '__main__':
    # Inference settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--input_image', type=str, required=True, help='input image to use')
    parser.add_argument('--model', type=str, required=True, help='model file to use')
    parser.add_argument('--output_filename', type=str, help='where to save the output image')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    opt = parser.parse_args()
    print(opt)
    
    input_image = Image.open(opt.input_image)

    start_time = perf_counter()
    out_img = inference(input_image, opt.model, cuda=opt.cuda)
    stop_time = perf_counter()
    print(f'inference took {stop_time - start_time} seconds')

    # Save output image
    out_img.save(opt.output_filename)
    print(f'output image saved to {opt.output_filename}')