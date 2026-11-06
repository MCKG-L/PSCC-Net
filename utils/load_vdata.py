import numpy as np
import torch.utils.data as data
import torch
from utils.image_io import read_rgb_image
from utils.paths import PROJECT_ROOT, project_path


class TestData(data.Dataset):
    def __init__(self, args):
        super(TestData, self).__init__()

        ddir = project_path(args.get('sample_dir', PROJECT_ROOT / 'sample'))
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
        names = sorted(path for path in ddir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
        authentic_names = [str(path) for path in names if 'authentic' in path.name.lower()]
        authentic_class = [0] * len(authentic_names)

        fake_names = [str(path) for path in names if 'authentic' not in path.name.lower()]
        fake_class = [1] * len(fake_names)

        self.image_names = authentic_names + fake_names
        self.image_class = authentic_class + fake_class
        if not self.image_names:
            raise ValueError(f'No supported images found in {ddir}')

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        image = read_rgb_image(image_name)

        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        return image, cls, image_name

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)
