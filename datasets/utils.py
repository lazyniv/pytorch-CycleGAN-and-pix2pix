import torch
import pydicom
import os
import numpy as np

from typing import List


DICOM_EXTENSION = '.dcm'


def load_dcm_paths_slices(path: str) -> List[str]:
    with open(path, 'r') as f:
        studies = f.read().splitlines()

    if studies[-1] == '\n':
        studies = studies[:-1]

    def flatten(list_2d):
        return [item for sublist in list_2d for item in sublist]

    slices = [list(map(lambda x: os.path.join(study, x), os.listdir(study))) for study in studies]

    return flatten(slices)


def load_image(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array
    return pixel_array


def to_dicom(tensor: torch.Tensor): ...

