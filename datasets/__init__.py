"""This package includes all the modules related to datasets loading and preprocessing

 To add a custom datasets class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseTrainDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseTrainDataset.__init__(self, opt).
    -- <__len__>:                       return the size of datasets.
    -- <__getitem__>:                   get a datasets point from datasets loader.
    -- <modify_commandline_options>:    (optionally) add datasets-specific options and set default options.

Now you can use the datasets class by specifying flag '--dataset_mode dummy'.
See our template datasets class 'template_dataset.py' for more details.
"""

import importlib
from datasets.base_dataset import BaseTrainDataset, BaseTestDataset


def find_dataset_using_name(dataset_name):
    """Import the module "datasets/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseTrainDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    dataset_lib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in dataset_lib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and (issubclass(cls, BaseTrainDataset) or issubclass(cls, BaseTestDataset)):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseTrainDataset with class name that matches %s "
            "in lowercase." % (dataset_filename, target_dataset_name)
        )

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the datasets class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options
