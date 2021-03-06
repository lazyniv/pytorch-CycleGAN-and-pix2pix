import os
import ntpath
from fpdf import FPDF
from PIL import Image, ImageDraw
import numpy as np
import cv2


def save_studies_to_pdf(studies_inference, save_dir, opt):
    for study in studies_inference:
        pdf_name = f"{os.path.splitext(ntpath.basename(study['study_path']))[0]}.pdf"
        save_path = os.path.join(save_dir, pdf_name)
        image_folder = os.path.join(save_dir, study['study_path'])
        os.mkdir(image_folder)
        _study_to_pdf(study, save_path, image_folder, opt)


def save_studies_to_folders(studies_inference, save_root_dir):
    for study in studies_inference:
        study_dir_name = os.path.splitext(ntpath.basename(study['study_path']))[0]
        study_dir = os.path.join(save_root_dir, study_dir_name)
        os.mkdir(study_dir)
        _save_study_to_dir(study, study_dir)


def _save_study_to_dir(study, save_dir):
    for _slice in study['slices']:
        slice_name = os.path.splitext(ntpath.basename(_slice['slice_path']))[0]
        slice_save_dir = os.path.join(save_dir, slice_name)
        os.mkdir(slice_save_dir)

        real_pixel_array = _slice['real_slice']
        real_image = Image.fromarray(real_pixel_array, mode='I')
        real_image.save(os.path.join(slice_save_dir), 'real.tif')

        for fake_visual in _slice['fake_visuals']:
            epoch = fake_visual['epoch']

            fake_pixel_array = fake_visual['fake_slice']
            fake_image = Image.fromarray(fake_pixel_array, mode='I')
            fake_image.save(os.path.join(slice_save_dir), f'{epoch}_epoch.tif')


def _study_to_pdf(study, save_path, image_save_folder, opt):
    width = opt.load_size * 4
    height = opt.load_size * 12 + 20 * (12 // (len(opt.epochs) // 4 + 1) - 1)

    pdf = FPDF(unit="pt", format=[width, height])
    pdf.add_page()

    slice_offset = (len(opt.epochs) // 4 + 1) * opt.load_size + 20

    row_counter_by_page = 0
    slice_counter_by_page = 0

    for _slice in study['slices']:
        slice_name = ntpath.basename(_slice['slice_path'])
        real = _slice['real_slice']
        real_image_path = _prepare_image_to_pdf(
            real,
            slice_name,
            f"real {opt.source_modality} " + slice_name,
            image_save_folder,
        )
        pdf.image(real_image_path, x=0, y=slice_offset * slice_counter_by_page)
        slice_counter_by_page += 1
        row_counter_by_page += 1
        slice_counter_by_row = 1
        next_row_offset = 0

        for fake_visual in _slice['fake_visuals']:
            epoch = fake_visual['epoch']
            fake_pixel_array = fake_visual['fake_slice']
            fake_image_path = _prepare_image_to_pdf(
                fake_pixel_array,
                slice_name,
                f"fake {opt.target_modality} {epoch} epoch",
                image_save_folder,
                epoch
            )
            pdf.image(
                fake_image_path,
                x=opt.load_size * slice_counter_by_row,
                y=slice_offset * slice_counter_by_page + opt.load_size * next_row_offset
            )
            slice_counter_by_row += 1

            if slice_counter_by_row == 4:
                slice_counter_by_row = 0
                next_row_offset += 1
                row_counter_by_page += 1

            if row_counter_by_page == 12:
                next_row_offset = 0
                slice_counter_by_page = 0
                pdf.add_page()

        pdf.output(save_path, "F")


def _prepare_image_to_pdf(
        pixel_array: np.ndarray,
        slice_name: str,
        text: str,
        image_save_folder: str,
        epoch: str = None,
) -> str:
    image = _convert_to_uint8_image(pixel_array)
    image = _write_txt_to_image(image, text)
    if epoch:
        image_save_path = os.path.join(image_save_folder, os.path.splitext(slice_name)[0])
    else:
        image_save_path = os.path.join(image_save_folder, os.path.splitext(slice_name)[0] + f'_{epoch}_epoch')
    _save_image(image, image_save_path)
    return image_save_path


def _convert_to_uint8_image(pixel_array: np.ndarray) -> Image:
    uint8_img = Image.fromarray(np.uint8(cv2.normalize(pixel_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)))
    return uint8_img


def _save_image(image: Image, save_path, ext='.png'):
    image.save(save_path + ext)


def _write_txt_to_image(image: Image, text: str) -> Image:
    draw = ImageDraw.Draw(image)
    (x, y) = (0, 0)
    color = 'rgb(255, 255, 255)'  # white color
    draw.text((x, y), text, fill=color)
    return image

