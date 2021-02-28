import os

from datasets.base_dataset import BaseTestDataset
from datasets.utils import load_paths, studies_to_slices
from datasets.utils import load_image


class SingleMRIDataset(BaseTestDataset):

    def __init__(self, opt, transformer):
        BaseTestDataset.__init__(self, opt, transformer)
        self.file = os.path.join(opt.dataroot)
        self.transform = transformer.get_transform()
        self.reverse_transform = transformer.get_reverse_transform()
        self.studies = studies_to_slices(load_paths(self.file))

    def __getitem__(self, index):
        study = self.studies[index]
        return {
            'study': study,
            'slices': [{'A': self.transform(load_image(_slice)), 'A_paths': _slice} for _slice in study]
        }

    def __len__(self):
        return len(self.studies)
