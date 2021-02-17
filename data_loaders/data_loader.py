import torch.utils.data
from datasets import find_dataset_using_name
from transforms import find_transformer_using_name


class DataLoader:
    """Wrapper class of Dataset class that performs multi-threaded datasets loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a datasets instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        transformer_class = find_transformer_using_name(opt.transformer_mode)
        transformer = transformer_class(opt)
        self.dataset = dataset_class(opt, transformer)
        print("datasets {} with transformer {} was created".format(
            type(self.dataset).__name__,
            type(transformer).__name__
        ))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads)
        )

    def __len__(self):
        """Return the number of datasets in the datasets"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of datasets"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def shuffle_index(self):
        self.dataset.shuffle_index()
