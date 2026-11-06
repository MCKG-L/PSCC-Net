import numpy as np
import torch.utils.data as data
from PIL import Image
import math
from random import randrange
import torch
from torch.utils.data import default_collate
from utils.image_io import read_rgb_image, read_mask, mask_path
from utils.paths import project_path


def _read_names(folder, filename, start=0, stop=None):
    names = (folder / filename).read_text(encoding='utf-8-sig').splitlines()
    names = [name.strip().replace('\\', '/') for name in names if name.strip()]
    return [str(folder / name) for name in names[start:stop]]


def validation_collate(samples):
    """Batch equal-size images; keep variable-size originals as single samples."""
    shapes = {(tuple(sample[0].shape), tuple(sample[1].shape)) for sample in samples}
    if len(shapes) == 1:
        return [default_collate(samples)]
    return [default_collate([sample]) for sample in samples]


def generate_4masks(mask):
    mask_pil = Image.fromarray(mask)

    (width2, height2) = (mask_pil.width // 2, mask_pil.height // 2)
    (width3, height3) = (mask_pil.width // 4, mask_pil.height // 4)
    (width4, height4) = (mask_pil.width // 8, mask_pil.height // 8)

    mask2 = mask_pil.resize((width2, height2))
    mask3 = mask_pil.resize((width3, height3))
    mask4 = mask_pil.resize((width4, height4))

    mask = mask.astype(np.float32) / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    mask2 = np.asarray(mask2).astype(np.float32) / 255
    mask2[mask2 > 0.5] = 1
    mask2[mask2 <= 0.5] = 0

    mask3 = np.asarray(mask3).astype(np.float32) / 255
    mask3[mask3 > 0.5] = 1
    mask3[mask3 <= 0.5] = 0

    mask4 = np.asarray(mask4).astype(np.float32) / 255
    mask4[mask4 > 0.5] = 1
    mask4[mask4 <= 0.5] = 0

    mask = torch.from_numpy(mask)
    mask2 = torch.from_numpy(mask2)
    mask3 = torch.from_numpy(mask3)
    mask4 = torch.from_numpy(mask4)

    return mask, mask2, mask3, mask4


def data_aug(img, data_aug_ind):
    img = Image.fromarray(img)
    if data_aug_ind == 0:
        return np.asarray(img)
    elif data_aug_ind == 1:
        return np.asarray(img.rotate(90, expand=True))
    elif data_aug_ind == 2:
        return np.asarray(img.rotate(180, expand=True))
    elif data_aug_ind == 3:
        return np.asarray(img.rotate(270, expand=True))
    elif data_aug_ind == 4:
        return np.asarray(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 5:
        return np.asarray(img.rotate(90, expand=True).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 6:
        return np.asarray(img.rotate(180, expand=True).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 7:
        return np.asarray(img.rotate(270, expand=True).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    else:
        raise Exception('Data augmentation index is not applicable.')


class TrainData(data.Dataset):
    def __init__(self, args):
        super(TrainData, self).__init__()
        path, crop_size, train_num, train_ratio, val_num = args['path'], args['crop_size'], args['train_num'], args['train_ratio'], args['val_num']

        path = project_path(path)
        authentic_names = _read_names(path / 'authentic', 'authentic.txt', start=val_num)
        splice_names = _read_names(path / 'splice', 'fake.txt', start=val_num)
        splice_names += _read_names(path / 'splice_randmask', 'fake.txt')
        copymove_names = _read_names(path / 'copymove', 'fake.txt', start=val_num)
        inpainting_names = _read_names(path / 'inpainting', 'fake.txt', start=val_num)
        self.image_names = [authentic_names, splice_names, copymove_names, inpainting_names]
        if (len(train_ratio) != 4 or any(ratio < 0 for ratio in train_ratio)
                or not math.isclose(sum(train_ratio), 1.0) or not 0 < train_ratio[0] < 1):
            raise ValueError('train_ratio must contain four nonnegative ratios summing to 1, with authentic in (0, 1)')
        if train_num <= 0 or val_num < 0:
            raise ValueError('train_num must be positive and val_num must be nonnegative')
        if len(crop_size) != 2 or any(size <= 0 or size % 8 for size in crop_size):
            raise ValueError('crop_size must contain height/width that are positive multiples of 8')
        for name, names, ratio in zip(('authentic', 'splice', 'copymove', 'inpainting'),
                                      self.image_names, train_ratio):
            if ratio > 0 and not names:
                raise ValueError(f'Empty {name} training split; add images or reduce val_num={val_num}')
        self.train_num = train_num
        self.train_ratio = train_ratio
        self.crop_size = crop_size

    def get_item(self, index):

        crop_height, crop_width = self.crop_size
        train_num = self.train_num
        train_ratio = self.train_ratio

        # get 4 class
        if index < train_num * train_ratio[0]:
            cls = 0
        elif train_num * train_ratio[0] <= index < train_num * (train_ratio[0] + train_ratio[1]):
            cls = 1
        elif train_num * (train_ratio[0] + train_ratio[1]) <= index < train_num * (
                train_ratio[0] + train_ratio[1] + train_ratio[2]):
            cls = 2
        else:
            cls = 3

        # get images in that class
        one_cls_names = self.image_names[cls]

        index = randrange(0, len(one_cls_names))

        # read the chosen image
        image_name = one_cls_names[index]
        image = read_rgb_image(image_name)
        im_height, im_width = image.shape[:2]

        # authentic
        if cls == 0:
            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((crop_width, crop_height), resample=Image.Resampling.BICUBIC)
                image = np.asarray(image)

            mask = np.zeros((crop_height, crop_width)).astype(np.uint8)

        # splice
        elif cls == 1:
            mask = read_mask(mask_path(image_name, cls))
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_width, crop_height), resample=Image.Resampling.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_width, crop_height), resample=Image.Resampling.BICUBIC)
                mask = np.asarray(mask)

        # copymove
        elif cls == 2:
            mask = read_mask(mask_path(image_name, cls))
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_width, crop_height), resample=Image.Resampling.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_width, crop_height), resample=Image.Resampling.BICUBIC)
                mask = np.asarray(mask)

        # inpainting
        elif cls == 3:
            mask = read_mask(mask_path(image_name, cls))
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_width, crop_height), resample=Image.Resampling.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_width, crop_height), resample=Image.Resampling.BICUBIC)
                mask = np.asarray(mask)

        else:
            raise Exception('class is not defined!')

        # image
        aug_index = randrange(0, 8)
        if crop_height != crop_width:
            aug_index = (0, 2, 4, 6)[randrange(0, 4)]
        image = data_aug(image, aug_index)
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        # mask
        mask = data_aug(mask, aug_index)
        mask, mask2, mask3, mask4 = generate_4masks(mask)

        return image, [mask, mask2, mask3, mask4], cls

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return self.train_num


class ValData(data.Dataset):
    def __init__(self, args):
        super(ValData, self).__init__()

        path, val_num = args['path'], args['val_num']

        if val_num <= 0:
            raise ValueError('val_num must be positive for validation')
        path = project_path(path)
        authentic_names = _read_names(path / 'authentic', 'authentic.txt', stop=val_num)
        splice_names = _read_names(path / 'splice', 'fake.txt', stop=val_num)
        copymove_names = _read_names(path / 'copymove', 'fake.txt', stop=val_num)
        inpainting_names = _read_names(path / 'inpainting', 'fake.txt', stop=val_num)

        authentic_cls = [0] * len(authentic_names)
        splice_cls = [1] * len(splice_names)
        copymove_cls = [2] * len(copymove_names)
        inpainting_cls = [3] * len(inpainting_names)

        self.image_names = authentic_names + splice_names + copymove_names + inpainting_names
        self.image_class = authentic_cls + splice_cls + copymove_cls + inpainting_cls
        if not self.image_names:
            raise ValueError('Validation split is empty')

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        image = read_rgb_image(image_name)
        im_height, im_width = image.shape[:2]

        # image
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        # authentic
        if cls == 0:
            # mask
            mask = np.zeros((im_height, im_width))
            mask = torch.from_numpy(mask.astype(np.float32))

        # splice
        elif cls == 1:
            # mask
            mask = read_mask(mask_path(image_name, cls))
            mask = torch.from_numpy((mask > 127.5).astype(np.float32))

        # copymove
        elif cls == 2:
            # mask
            mask = read_mask(mask_path(image_name, cls))
            mask = torch.from_numpy((mask > 127.5).astype(np.float32))

        # inpainting
        elif cls == 3:
            # mask
            mask = read_mask(mask_path(image_name, cls))
            mask = torch.from_numpy((mask > 127.5).astype(np.float32))

        else:
            raise Exception('class is not defined!')

        if tuple(mask.shape) != (im_height, im_width):
            raise ValueError(f'Image and mask sizes differ: {image_name}')
        return image, mask, cls, image_name

    def __getitem__(self, index):
        res = self.get_item(index)

        return res

    def __len__(self):
        return len(self.image_names)
