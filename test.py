import os

from tqdm import tqdm

from options.test_options import TestOptions
from data_loaders import create_data_loader
from models import create_model
from util.inference_utils import save_studies_to_pdf, save_studies_to_folders

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.display_id = -1
    data_loader = create_data_loader(opt)

    inference = [
        {
            'study_path': data['study_path'][0],
            'slices': [
                {
                    'input': _slice['A'],
                    'slice_path': _slice['A_paths'][0],
                    'real_slice': data_loader.dataset.transformer.backward_transform(_slice['A']),
                    'fake_visuals': []
                }
                for _slice in data['slices']
            ]
        }
        for data in data_loader
    ]

    for epoch in tqdm(opt.epochs):
        print(f'processing {epoch}-th epoch')

        opt.epoch = epoch
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)
        model.eval()

        for i, study in enumerate(inference):
            print(f"processing {i}-th study {study['study_path']}")
            for j, _slice in enumerate(study['slices']):
                print(f"processing {j}-th slice {_slice['slice_path']}")
                model.set_input({
                    'A': _slice['input'],
                    'A_paths': _slice['slice_path']
                })
                model.test()

                _slice['fake_visuals'].append(
                    {
                        'epoch': epoch,
                        'fake_slice': data_loader.dataset.transformer.backward_transform(model.get_current_visuals()['fake'])
                    }
                )

    root_inference_dir = os.path.join(opt.results_dir, opt.name)
    os.makedirs(root_inference_dir, exist_ok=True)

    print('save inference results to folders ...')
    save_images_root = os.path.join(root_inference_dir, 'images')
    os.makedirs(save_images_root, exist_ok=True)
    save_studies_to_folders(inference, save_images_root)

    if opt.save_inference_to_pdf:
        print('save inference results to pdf ...')
        pdfs_save_dir = os.path.join(root_inference_dir, 'pdf')
        os.makedirs(pdfs_save_dir, exist_ok=True)
        save_studies_to_pdf(inference, pdfs_save_dir, opt)
