import argparse

import matplotlib.pyplot as plt
from PIL import Image

from inference import inference


def main(opt):
    """ main function """
    input_image = Image.open(opt.input_image)
    output_img = inference(input_image, opt.model, cuda=opt.cuda)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(input_image)
    axarr[1].imshow(output_img)
    plt.show()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Compare input and output result')
    parser.add_argument('--input_image', type=str, required=True, help='input image to use')
    parser.add_argument('--model', type=str, required=True, help='model file to use')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    opt = parser.parse_args()
    print(opt)
    main(opt)
