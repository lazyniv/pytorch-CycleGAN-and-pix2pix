from data_loaders.data_loader import DataLoader


def create_data_loader(opt) -> DataLoader:
    """Create a datasets given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    """

    data_loader = DataLoader(opt)
    return data_loader
