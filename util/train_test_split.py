from typing import List
from random import sample, seed
import os
import numpy as np


def train_test_split_by_studies(
        A_folders: List[str],
        B_folders: List[str],
        A_test_size: int,
        B_test_size: int,
        dst_folder: str,
        _seed: int = None
):
    if _seed is not None:
        seed(_seed)

    A_studies = list(np.concatenate([
        list(map(lambda x: os.path.join(folder, x), os.listdir(folder))) for folder in A_folders
    ]).flat)

    B_studies = list(np.concatenate([
        list(map(lambda x: os.path.join(folder, x), os.listdir(folder))) for folder in B_folders
    ]).flat)

    A_test_indexes = sample(range(0, len(A_studies) - 1), A_test_size)
    A_test = [A_studies[i] for i in A_test_indexes]
    A_train = list(set(A_studies) - set(A_test))

    B_test_indexes = sample(range(0, len(B_studies) - 1), B_test_size)
    B_test = [B_studies[i] for i in B_test_indexes]
    B_train = list(set(B_studies) - set(B_test))

    _save_slices_paths_to_file(A_train, dst_folder, 'trainA')
    _save_slices_paths_to_file(B_train, dst_folder, 'trainB')
    _save_studies_paths_to_file(A_test, dst_folder, 'testA')
    _save_studies_paths_to_file(B_test, dst_folder, 'testB')


def _save_studies_paths_to_file(
        studies: List[str],
        dst_folder: str,
        label: str
):
    file_path = os.path.join(os.path.abspath(dst_folder), label)

    with open(file_path, 'w') as f:
        for study in studies:
            f.write("{}\n".format(study))


def _save_slices_paths_to_file(
        studies: List[str],
        dst_folder: str,
        label: str
):
    file_path = os.path.join(os.path.abspath(dst_folder), label)

    def flatten(list_2d):
        return [item for sublist in list_2d for item in sublist]

    slices = [list(map(lambda x: os.path.join(study, x), os.listdir(study))) for study in studies]

    slices = flatten(slices)

    with open(file_path, 'w') as f:
        for _slice in slices:
            f.write("{}\n".format(_slice))


def split():
    A_folders = [
        '/home/dima/code/diploma/pytorch-CycleGAN-and-pix2pix/modalityA1',
        '/home/dima/code/diploma/pytorch-CycleGAN-and-pix2pix/modalityA2',
    ]

    B_folders = [
        '/home/dima/code/diploma/pytorch-CycleGAN-and-pix2pix/modalityB1',
        '/home/dima/code/diploma/pytorch-CycleGAN-and-pix2pix/modalityB2',
    ]

    A_test_size = 1
    B_test_size = 1

    dst_folder = '/home/dima/code/diploma/pytorch-CycleGAN-and-pix2pix/data/test1'

    train_test_split_by_studies(A_folders, B_folders, A_test_size, B_test_size, dst_folder)

if __name__ == '__main__':
    split()
