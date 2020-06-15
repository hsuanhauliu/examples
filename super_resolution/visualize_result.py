import argparse
import os
from random import choice

import matplotlib.pyplot as plt
from PIL import Image

from inference import inference


def main(opt):
    """ main function """
    print(opt.input_image)
    if opt.input_image:
        visualize(opt.input_image, opt.model, opt.cuda)
    else:
        test_img_dir = 'dataset/BSDS300/images/test/'
        train_img_dir = 'dataset/BSDS300/images/train/'
        test_images = os.listdir(test_img_dir)
        train_images = os.listdir(train_img_dir)
        all_images = [test_img_dir + img for img in test_images] + [train_img_dir + img for img in train_images]

        # Ctrl + c to stop
        while True:
            input_image = choice(all_images)
            visualize(input_image, opt.model, opt.cuda)
        

def visualize(input_image, model, cuda):
    input_image = Image.open(input_image)
    output_image = inference(input_image, model, cuda=cuda)
    print(f'input image size: {input_image.size}')
    print(f'output image size: {output_image.size}')

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(input_image)
    axarr[1].imshow(output_image)
    plt.show()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Compare input and output result')
    parser.add_argument('--input_image', type=str, default='', help='input image to use')
    parser.add_argument('--model', type=str, required=True, help='model file to use')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    opt = parser.parse_args()
    print(opt)
    main(opt)
