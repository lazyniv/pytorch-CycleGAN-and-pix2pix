import os

from datasets.base_dataset import BaseTestDataset
from datasets.utils import load_paths, group_slices_by_study
from datasets.utils import load_image


class SingleMRIDataset(BaseTestDataset):

    def __init__(self, opt, transformer):
        BaseTestDataset.__init__(self, opt, transformer)
        self.file = os.path.join(opt.dataroot)
        self.paths = load_paths(self.file)
        self.studies = group_slices_by_study(self.paths)

    def __getitem__(self, index):
        return {
            'study_path': self.paths[index],
            'slices': [
                {
                    'A': self.transformer.forward_transform(load_image(_slice)),
                    'A_paths': _slice
                }
                for _slice in self.studies[index]
            ]
        }

    def __len__(self):
        return len(self.studies)
