"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
import cv2
from PIL import Image, ImageDraw
import numpy as np


def main():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    epochs = opt.epoch
    web_dir = os.path.join(opt.results_dir, opt.name, opt.direction_label)
    os.makedirs(web_dir, exist_ok=True)
    save_joined_images_dir = os.path.join(opt.joined_results_dir, opt.direction_label, 'epochs_' + str(epochs))
    os.makedirs(save_joined_images_dir, exist_ok=True)

    for epoch in epochs:
        opt.epoch = epoch
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        # create a website
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(web_dir, visuals, img_path, epoch, is_test=True, aspect_ratio=opt.aspect_ratio,
                        width=opt.display_winsize)

    domains = opt.direction_label.split('to')
    join_real_fake_images(opt.dataroot, web_dir, save_joined_images_dir, epochs, src_domain=domains[0], target_domain=domains[1])


def join_real_fake_images(src_images_dir, fake_images_dir, save_dir, epochs, src_domain, target_domain):

    real_images = os.listdir(src_images_dir)

    for img in real_images:
        img_name, _ = os.path.splitext(img)
        print("processing ---> " + img_name)
        study_name = img_name.split('_')[0]
        study_dir = os.path.join(save_dir, study_name)
        if not os.path.isdir(study_dir):
            os.mkdir(study_dir)
        real_img_path = os.path.join(src_images_dir, img)
        img = __get_uint8_image(real_img_path)
        __mark_image(img, "real " + src_domain)
        fake_images = __get__marked_fake_images(img_name, fake_images_dir, epochs, target_domain)
        if len(epochs) < 4:
            width, height = img.size[0] * (len(epochs) + 1), img.size[1]
        else:
            width = img.size[0] * 4
            if len(epochs) % 4 == 0:
                height = img.size[1] * (len(epochs) // 4)
            else:
                height = img.size[1] * (len(epochs) // 4) + 1

        new_im = Image.new('LA', (256 * 4, 256 * 4))

        x_offset, y_offset = 0, 0
        new_im.paste(img, (x_offset, y_offset))
        x_offset += img.size[0]

        i = 1
        for fake_img in fake_images:
            new_im.paste(fake_img, (x_offset, y_offset))
            i += 1
            if i % 4 == 0:
                y_offset += img.size[1]
                x_offset = 0
            else:
                x_offset += img.size[0]
        new_filename = os.path.join(study_dir, img_name)
        new_im.save(new_filename + '.png')


def __get_uint8_image(path):
    img = Image.open(path)
    arr = np.array(img)
    uint8_img = Image.fromarray(np.uint8(cv2.normalize(arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)))
    return uint8_img


def __mark_image(img, text):
    draw = ImageDraw.Draw(img)
    (x, y) = (0, 0)
    color = 'rgb(255, 255, 255)'  # white color
    draw.text((x, y), text, fill=color)


def __get__marked_fake_images(real_image_name, fake_images_dir, epochs, domain):
    fake_images = []
    for epoch in epochs:
        fake_img_path = os.path.join(fake_images_dir, real_image_name + '_' + str(epoch) + 'epoch.tif')
        fake_img = __get_uint8_image(fake_img_path)
        __mark_image(fake_img, "fake " + domain + '_' + epoch + 'epoch')
        fake_images.append(fake_img)
    return fake_images


if __name__ == '__main__':
    main()
