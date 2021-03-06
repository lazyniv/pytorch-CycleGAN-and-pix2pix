import os

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

    studies_inference = []

    for i, data in enumerate(data_loader):

        if i >= opt.num_test:
            break

        study = data['path_to_study']
        slices = data['slices']
        print('processing {}-th study {}...'.format(i, study))

        study_inference = {
            'study_path': study,
            'slices': []
        }

        for j, _slice in enumerate(slices):
            print('processing {}-th slice {}...'.format(j, _slice['A_paths']))

            slice_inference = {
                'slice_path': _slice['A_paths'],
                'real_slice': data_loader.dataset.transformer.backward_transform(_slice['A']),
            }

            fake_visuals = []

            for epoch in opt.epochs:
                print('processing {}-th epoch'.format(epoch))

                opt.epoch = epoch
                model = create_model(opt)  # create a model given opt.model and other options
                model.setup(opt)
                model.eval()

                model.set_input(_slice)
                model.test()           # run inference
                fake_visuals.append({
                    "epoch": epoch,
                    "fake_slice": data_loader.dataset.transformer.backward_transform(model.get_current_visuals()['fake'])
                })

            slice_inference['fake_visuals'] = fake_visuals
            study_inference['slices'].append(slice_inference)

        studies_inference.append(study_inference)

    root_inference_dir = os.path.join(opt.results_dir, opt.name)
    os.mkdir(root_inference_dir)

    if opt.save_inference_to_pdf:
        pdfs_save_dir = os.path.join(root_inference_dir, 'pdf')
        os.mkdir(pdfs_save_dir)
        save_studies_to_pdf(studies_inference, pdfs_save_dir, opt)

    save_images_root = os.path.join(root_inference_dir, 'images')
    os.mkdir(save_images_root)
    save_studies_to_folders(studies_inference, opt.save_inference_root_dir)

