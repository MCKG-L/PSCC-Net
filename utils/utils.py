import os
from pathlib import Path

import torch
import torchvision.utils as tv_utils
from utils.paths import PROJECT_ROOT, image_basename


def adjust_learning_rate(optimizer, epoch, lr_strategy, lr_decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    current_learning_rate = lr_strategy[epoch // lr_decay_step]
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))


def save_image(image, image_name, category):
    images = torch.split(image, 1, dim=0)
    batch_num = len(images)

    for ind in range(batch_num):
        save_dir = PROJECT_ROOT / '{}_results'.format(category)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / Path(image_basename(image_name[ind])).with_suffix('.png')
        tv_utils.save_image(images[ind], save_path)


def findLastCheckpoint(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        result = 0
        for file in file_list:
            try:
                num = int(file.split('.')[0].split('_')[-1])
                result = max(result, num)
            except:
                continue
        return result
    else:
        os.mkdir(save_dir)
        return 0
