from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import save_image
from utils.checkpoint import load_model_weights
from utils.paths import PROJECT_ROOT, project_path, image_basename

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.load_vdata import TestData

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_network_weight(net, checkpoint_dir, name):
    weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
    load_model_weights(net, weight_path)
    print('{} weight-loading succeeds'.format(name))


def test(args):
    # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    checkpoint_root = project_path(args.get('checkpoint_dir', PROJECT_ROOT / 'checkpoint'))
    FENet_checkpoint_dir = checkpoint_root / f'{FENet_name}_checkpoint'
    SegNet_checkpoint_dir = checkpoint_root / f'{SegNet_name}_checkpoint'
    ClsNet_checkpoint_dir = checkpoint_root / f'{ClsNet_name}_checkpoint'

    # load FENet weight
    FENet = FENet.to(device)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)

    # load SegNet weight
    SegNet = SegNet.to(device)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)

    # load ClsNet weight
    ClsNet = ClsNet.to(device)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

    test_data_loader = DataLoader(TestData(args), batch_size=1, shuffle=False,
                                  num_workers=args.get('num_workers', 4))

    for batch_id, test_data in enumerate(test_data_loader):

        image, cls, name = test_data
        image = image.to(device)

        with torch.no_grad():

            # backbone network
            FENet.eval()
            feat = FENet(image)

            # localization head
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
                                      align_corners=True)

            # classification head
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        _, binary_cls = torch.max(pred_logit, 1)

        pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'

        if args.save_tag:
            save_image(pred_mask, name, 'mask')

        print_name = Path(image_basename(name[0])).stem

        print(f'The image {print_name} is {pred_tag}')


if __name__ == '__main__':
    args = get_pscc_args()
    test(args)
