import torch
import pydicom
import numpy as np
import os

from typing import List


DICOM_EXTENSION = '.dcm'


def is_dicom_file(filename: str) -> bool:
    return filename.endswith(DICOM_EXTENSION)


def load_dcm_paths(folder: str) -> List[str]:
    images = []
    assert os.path.isdir(folder), '%s is not a valid directory' % dir

    for root, _, files in sorted(os.walk(folder)):
        for file in files:
            if is_dicom_file(file):
                path = os.path.join(root, file)
                images.append(path)

    return images


def load_image(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array
    return pixel_array


def to_dicom(tensor: torch.Tensor): ...

