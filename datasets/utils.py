import torch
import pydicom
import os
import numpy as np
import logging

from typing import List


DICOM_EXTENSION = '.dcm'


def load_dcm_paths_slices(path: str) -> List[str]:
    with open(path, 'r') as f:
        slices = f.read().splitlines()

    if slices[-1] == '\n':
        slices = slices[:-1]

    return slices


def load_image(path: str) -> np.ndarray:
    try:
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array
        return pixel_array
    except Exception as e:
        logging.error("Error in file {} :".format(path), str(e))


def to_dicom(tensor: torch.Tensor): ...

