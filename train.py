from itertools import chain
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.utils import save_image, adjust_learning_rate
from utils.checkpoint import load_model_weights, resume_training
from utils.paths import PROJECT_ROOT, project_path

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.config import get_pscc_args
from utils.load_tdata import TrainData
from utils.load_tdata import ValData, validation_collate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(args):
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

    # load train data
    train_data_loader = DataLoader(TrainData(args), batch_size=args['train_bs'], shuffle=True,
                                   num_workers=args.get('num_workers', 4))

    FENet = FENet.to(device)
    SegNet = SegNet.to(device)
    ClsNet = ClsNet.to(device)

    params = list(FENet.parameters()) + list(SegNet.parameters()) + list(ClsNet.parameters())
    optimizer = torch.optim.Adam(params, lr=args['learning_rate'])

    # Keep the original three-file format and accept both prefixed/plain weights.
    checkpoint_root = project_path(args.get('checkpoint_dir', PROJECT_ROOT / 'checkpoint'))
    models = {FENet_name: FENet, SegNet_name: SegNet, ClsNet_name: ClsNet}
    directories = {name: checkpoint_root / f'{name}_checkpoint' for name in models}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    FENet_dir, SegNet_dir, ClsNet_dir = (directories[name] for name in models)
    weight_paths = {name: directories[name] / f'{name}.pth' for name in models}
    if any(path.exists() for path in weight_paths.values()):
        missing = [str(path) for path in weight_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'Incomplete pretrained weights: {missing}')
        for name, model in models.items():
            load_model_weights(model, weight_paths[name])
            print(f'{name} weight-loading succeeds: {weight_paths[name]}')
    else:
        warnings.warn('No PSCC-Net pretrained weights found; training starts from initialized weights.')

    # validation
    print('length of traindata: {}'.format(len(train_data_loader)))
    previous_score = validation(FENet, SegNet, ClsNet, args)
    print('previous_score {0:.4f}'.format(previous_score))

    # cross entropy loss
    authentic_ratio = args['train_ratio'][0]
    fake_ratio = 1 - authentic_ratio
    print('authentic_ratio: {}'.format(authentic_ratio), 'fake_ratio: {}'.format(fake_ratio))
    weights = [1. / authentic_ratio, 1. / fake_ratio]
    weights = torch.tensor(weights)
    CE_loss = nn.CrossEntropyLoss(weight=weights).to(device)

    BCE_loss_full = nn.BCELoss(reduction='none').to(device)

    initial_epoch = resume_training(models, optimizer, checkpoint_root)
    if initial_epoch > 0:
        print(f'Resuming all networks and optimizer from epoch {initial_epoch}')

    for epoch in range(initial_epoch, args['num_epochs']):

        adjust_learning_rate(optimizer, epoch, args['lr_strategy'], args['lr_decay_step'])
        seg_total, seg_correct, seg_loss_sum = [0] * 3
        cls_total, cls_correct, cls_loss_sum = [0] * 3

        for batch_id, train_data in enumerate(train_data_loader):

            image, masks, cls = train_data
            cls[cls != 0] = 1

            mask1, mask2, mask3, mask4 = masks

            # median-frequency class weighting
            mask1_balance = torch.ones_like(mask1)
            if (mask1 == 1).sum():
                mask1_balance[mask1 == 1] = 0.5 / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
                mask1_balance[mask1 == 0] = 0.5 / ((mask1 == 0).sum().to(torch.float) / mask1.numel())
            else:
                print('Mask1 balance is not working!')

            mask2_balance = torch.ones_like(mask2)
            if (mask2 == 1).sum():
                mask2_balance[mask2 == 1] = 0.5 / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
                mask2_balance[mask2 == 0] = 0.5 / ((mask2 == 0).sum().to(torch.float) / mask2.numel())
            else:
                print('Mask2 balance is not working!')

            mask3_balance = torch.ones_like(mask3)
            if (mask3 == 1).sum():
                mask3_balance[mask3 == 1] = 0.5 / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
                mask3_balance[mask3 == 0] = 0.5 / ((mask3 == 0).sum().to(torch.float) / mask3.numel())
            else:
                print('Mask3 balance is not working!')

            mask4_balance = torch.ones_like(mask4)
            if (mask4 == 1).sum():
                mask4_balance[mask4 == 1] = 0.5 / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
                mask4_balance[mask4 == 0] = 0.5 / ((mask4 == 0).sum().to(torch.float) / mask4.numel())
            else:
                print('Mask4 balance is not working!')

            image = image.to(device)
            mask1, mask2, mask3, mask4 = mask1.to(device), mask2.to(device), mask3.to(device), mask4.to(device)
            mask1_balance, mask2_balance, mask3_balance, mask4_balance, = mask1_balance.to(device), mask2_balance.to(
                device), mask3_balance.to(device), mask4_balance.to(device)

            cls = cls.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # feature extraction network
            FENet.train()
            feat = FENet(image)
            SegNet.train()
            [pred_mask1, pred_mask2, pred_mask3, pred_mask4] = SegNet(feat)

            ClsNet.train()
            pred_logit = ClsNet(feat)

            pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_mask1.squeeze(dim=1), pred_mask2.squeeze(
                dim=1), pred_mask3.squeeze(dim=1), pred_mask4.squeeze(dim=1)

            mask1_loss = torch.mean(BCE_loss_full(pred_mask1, mask1) * mask1_balance)
            mask2_loss = torch.mean(BCE_loss_full(pred_mask2, mask2) * mask2_balance)
            mask3_loss = torch.mean(BCE_loss_full(pred_mask3, mask3) * mask3_balance)
            mask4_loss = torch.mean(BCE_loss_full(pred_mask4, mask4) * mask4_balance)
            seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss

            cls_loss = CE_loss(pred_logit, cls)

            loss = seg_loss + cls_loss

            loss.backward()
            optimizer.step()

            # localization
            binary_mask1 = torch.zeros_like(pred_mask1)
            binary_mask1[pred_mask1 > 0.5] = 1
            binary_mask1[pred_mask1 <= 0.5] = 0

            seg_correct += (binary_mask1 == mask1).sum().item()
            seg_total += int(torch.ones_like(mask1).sum().item())

            # detection
            _, binary_cls = torch.max(pred_logit, 1)

            cls_correct += (binary_cls == cls).sum().item()
            cls_total += int(torch.ones_like(cls).sum().item())

            # print statistics
            seg_loss_sum += seg_loss.item()
            cls_loss_sum += cls_loss.item()

            if batch_id % 100 == 99:
                print(
                    '[{0}, {1}] batch_loc_acc: [{2}/{3}] {4:.2f}, seg_loss: {5:.4f}; batch_cls_acc: [{6}/{7}] {8:.2f}, '
                    'cls_loss: {9:.4f}'.format(epoch + 1, batch_id + 1, seg_correct, seg_total,
                                               seg_correct / seg_total * 100,
                                               seg_loss_sum / 100, cls_correct, cls_total,
                                               cls_correct / cls_total * 100,
                                               cls_loss_sum / 100))

                seg_total, seg_correct, seg_loss_sum, cls_total, cls_correct, cls_loss_sum = [0] * 6

        if epoch % 1 == 0:
            FENet_checkpoint = {'model': FENet.state_dict(),
                                'optimizer': optimizer.state_dict()}
            torch.save(FENet_checkpoint,
                       '{0}/{1}_{2}.pth'.format(FENet_dir, FENet_name, epoch + 1))

            SegNet_checkpoint = {'model': SegNet.state_dict(),
                                 'optimizer': optimizer.state_dict()}
            torch.save(SegNet_checkpoint,
                       '{0}/{1}_{2}.pth'.format(SegNet_dir, SegNet_name, epoch + 1))

            ClsNet_checkpoint = {'model': ClsNet.state_dict(),
                                 'optimizer': optimizer.state_dict()}
            torch.save(ClsNet_checkpoint,
                       '{0}/{1}_{2}.pth'.format(ClsNet_dir, ClsNet_name, epoch + 1))

            current_score = validation(FENet, SegNet, ClsNet, args)
            print('current_score: {0:.4f}'.format(current_score))

            if current_score >= previous_score:
                torch.save(FENet.state_dict(), '{0}/{1}.pth'.format(FENet_dir, FENet_name))
                torch.save(SegNet.state_dict(), '{0}/{1}.pth'.format(SegNet_dir, SegNet_name))
                torch.save(ClsNet.state_dict(), '{0}/{1}.pth'.format(ClsNet_dir, ClsNet_name))
                previous_score = current_score


def validation(FENet, SegNet, ClsNet, args):
    val_data_loader = DataLoader(ValData(args), batch_size=args['val_bs'], shuffle=False,
                               num_workers=args.get('num_workers', 4), collate_fn=validation_collate)
    seg_correct, seg_total, cls_correct, cls_total = [0] * 4

    for batch_id, val_data in enumerate(chain.from_iterable(val_data_loader)):

        image, mask, cls, name = val_data

        image = image.to(device)
        mask = mask.to(device)
        cls[cls != 0] = 1
        cls = cls.to(device)

        with torch.no_grad():

            # feature extraction network
            FENet.eval()
            feat = FENet(image)

            # localization network
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            # classification network
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        if pred_mask.shape[-2:] != mask.shape[-2:]:
            pred_mask = F.interpolate(pred_mask, size=(mask.size(1), mask.size(2)), mode='bilinear', align_corners=True)

        binary_mask1 = pred_mask.squeeze(1) > 0.5
        if binary_mask1.shape != mask.shape:
            raise ValueError(f'Prediction/target shapes differ: {binary_mask1.shape} vs {mask.shape}')
        seg_correct += (binary_mask1 == (mask > 0.5)).sum().item()
        seg_total += mask.numel()

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)

        _, binary_cls = torch.max(pred_logit, 1)
        cls_correct += (binary_cls == cls).sum().item()
        cls_total += int(torch.ones_like(cls).sum().item())

        if args['save_tag']:
            save_image(pred_mask, name, 'mask')
    if not seg_total or not cls_total:
        raise ValueError('Validation produced no samples')
    return (seg_correct / seg_total + cls_correct / cls_total) / 2


if __name__ == '__main__':
    args = get_pscc_args()
    train(args)
