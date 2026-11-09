import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from utils.load_tdata import TrainData, ValData
from utils.load_vdata import TestData
from utils.config import get_pscc_args
from utils.utils import save_image
import train


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_dataset(self, counts=(2, 2, 2, 2, 1), mode='RGB'):
        for category, count in zip(
                ('authentic', 'splice', 'copymove', 'inpainting', 'splice_randmask'), counts):
            folder = self.root / category
            (folder / 'fake').mkdir(parents=True)
            (folder / 'mask').mkdir()
            names = []
            for i in range(count):
                name = f'fake/{i}.png'
                Image.new(mode, (16, 16)).save(folder / name)
                Image.fromarray(np.ones((16, 16), dtype=np.uint8)).save(folder / 'mask' / f'{i}.png')
                names.append(name)
            listing = 'authentic.txt' if category == 'authentic' else 'fake.txt'
            (folder / listing).write_text('\n'.join(names) + '\n', encoding='utf-8')
        args = get_pscc_args()
        args.path = str(self.root)
        args.crop_size = [16, 16]
        args.val_num = 1
        args.train_num = 4
        return args

    def test_short_validation_lists_keep_labels_aligned(self):
        args = self.make_dataset(counts=(1, 2, 1, 2, 1))
        args.val_num = 2
        data = ValData(args)
        self.assertEqual(data.image_class, [0, 1, 1, 2, 3, 3])

    def test_empty_training_split_has_clear_error(self):
        args = self.make_dataset(counts=(1, 2, 2, 2, 1))
        with self.assertRaisesRegex(ValueError, 'authentic'):
            TrainData(args)

    def test_training_accepts_grayscale_and_rgba(self):
        for mode in ('L', 'RGBA'):
            with self.subTest(mode=mode):
                # Rewrite fixtures in place for each image mode.
                args = self.make_dataset(mode=mode) if mode == 'L' else get_pscc_args()
                args.path = str(self.root)
                args.crop_size = [16, 16]
                args.val_num = 1
                args.train_num = 4
                if mode == 'RGBA':
                    for f in self.root.glob('*/fake/*.png'):
                        Image.new('RGBA', (16, 16), (0, 0, 0, 0)).save(f)
                data = TrainData(args)
                for i in range(4):
                    image, masks, label = data[i]
                    self.assertEqual(tuple(image.shape), (3, 16, 16))
                    if mode == 'RGBA':
                        self.assertTrue(torch.all(image == 1))
                    if label:
                        self.assertTrue(torch.all(masks[0] == 1))

    def test_validation_normalizes_binary_one_masks(self):
        args = self.make_dataset()
        _, mask, label, _ = ValData(args)[1]
        self.assertEqual(label, 1)
        self.assertEqual(mask.ndim, 2)
        self.assertTrue(torch.all(mask == 1))

    def test_prediction_filters_non_images_and_reads_grayscale(self):
        Image.new('L', (16, 16)).save(self.root / 'gray.png')
        (self.root / 'notes.txt').write_text('ignore', encoding='utf-8')
        (self.root / 'nested').mkdir()
        args = get_pscc_args()
        args.sample_dir = str(self.root)
        data = TestData(args)
        self.assertEqual(len(data), 1)
        self.assertEqual(tuple(data[0][0].shape), (3, 16, 16))

    def test_default_paths_ignore_current_working_directory(self):
        args = get_pscc_args()
        with patch('os.getcwd', return_value=str(self.root)):
            data = TestData(args)
        self.assertTrue(all(Path(p).is_absolute() for p in data.image_names))
        self.assertTrue(Path(args.checkpoint_dir).is_absolute())

    def test_saved_name_handles_windows_paths_and_long_suffixes(self):
        with patch('utils.utils.tv_utils.save_image') as save:
            save_image(torch.zeros(1, 1, 2, 2), [r'C:\input\a.b.jpeg'], 'mask')
        self.assertEqual(Path(save.call_args.args[1]).name, 'a.b.png')

    def test_validation_batch_two_does_not_broadcast(self):
        class Backbone(nn.Module):
            def forward(self, x):
                return x

        class Segment(nn.Module):
            def forward(self, x):
                return [x[:, :1]]

        class Classify(nn.Module):
            def forward(self, x):
                return torch.tensor([[1., 0.], [1., 0.]])

        images = torch.zeros(2, 3, 2, 2)
        labels = torch.tensor([0, 0])
        samples = [(images, images[:, 0], labels, ['a.png', 'b.png'])]
        args = get_pscc_args()
        with patch.object(train, 'ValData', return_value=[]), \
                patch.object(train, 'DataLoader', return_value=[samples]), \
                patch.object(train, 'device', torch.device('cpu')):
            self.assertEqual(train.validation(Backbone(), Segment(), Classify(), args), 1.)

    def test_missing_resume_file_does_not_modify_models(self):
        from utils.checkpoint import resume_training
        models = {n: nn.Linear(2, 1) for n in ('HRNet', 'NLCDetection', 'DetectionHead')}
        optimizer = torch.optim.Adam([p for m in models.values() for p in m.parameters()])
        before = {n: copy.deepcopy(m.state_dict()) for n, m in models.items()}
        for name in ('HRNet', 'NLCDetection'):
            folder = self.root / f'{name}_checkpoint'
            folder.mkdir()
            state = {k: torch.ones_like(v) * 99 for k, v in models[name].state_dict().items()}
            torch.save({'model': state, 'optimizer': optimizer.state_dict()}, folder / f'{name}_3.pth')
        with self.assertRaisesRegex((RuntimeError, FileNotFoundError), 'DetectionHead'):
            resume_training(models, optimizer, self.root)
        for name, model in models.items():
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name][key]))

    def test_loads_legacy_prefixed_weights_into_plain_model(self):
        from utils.checkpoint import load_model_weights
        model = nn.Linear(2, 1)
        state = {f'module.{k}': torch.ones_like(v) for k, v in model.state_dict().items()}
        path = self.root / 'legacy.pth'
        torch.save(state, path)
        load_model_weights(model, path)
        self.assertTrue(all(torch.all(p == 1) for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
