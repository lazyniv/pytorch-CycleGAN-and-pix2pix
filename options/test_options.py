from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--direction_label', type=str, default='AtoB', help='direction of transfer e.g. T1toT2')
        parser.add_argument('--generator_label', type=str, default='A', help='which generator we would use')
        parser.add_argument('--joined_results_dir', type=str, required=True, help='save joined results here')
        parser.add_argument('--group_results_by_epoch', action='store_true', help='group the results by epoch folders')
        parser.add_argument('--join_results', action='store_true', help='join the results of many epochs to one image')
        parser.add_argument('--test_on_all_epochs', action='store_true', help='run inference on all of the epochs')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
