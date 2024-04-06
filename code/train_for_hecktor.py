import json
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.losses import cal_variance
from utils import losses, test_util, cube_utils, validation_util
from networks.resnet3D import Resnet34
from networks.Unet3d import BaselineUNet
from dataloaders.dataset import *
from monai.losses import DiceCELoss

from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    PersistentDataset,
    decollate_batch,
)

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotated,
    ToTensord,
    PadListDataCollate,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='HECKTOR', help='dataset_name')
parser.add_argument('--root_path', type=str, default='..\\hecktor\\', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='DoubleNet', help='exp_name')
parser.add_argument('--model', type=str, default='U+R', help='model_name')
parser.add_argument('--max_iteration', type=int, default=50000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=524, help='total samples of the dataset')
parser.add_argument('--labeled_bs', type=int, default=1, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=74, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cube_size', type=int, default=36, help='size of each cube')
parser.add_argument('--weight_decay', default=1e-5, help='weight_decay')
parser.add_argument('--consistency', type=float, default=0.2, help='consistency')
args = parser.parse_args()


def create_model(name='Unet3D', num_classes=3):
    # Network definition
    if name == "Unet3D":
        net = BaselineUNet(in_channels=2, n_cls=num_classes, n_filters=24)
    if name == "resnet34":
        net = Resnet34(n_channels=2, n_classes=num_classes)
    return net

snapshot_path = "../model" + "/{}_{}_{}labeled_cons{}_cube_size{}/{}".format(args.dataset_name, args.exp, args.labelnum, args.consistency, args.cube_size, args.model)
root_path = args.root_path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]

# Initialize variables
# Data
ct_a_min = -200
ct_a_max = 400
pt_a_min = 0
pt_a_max = 25
strength = 1  # Data aug strength
p = 0.5  # Data aug transforms probability

# Paths
data_path = os.path.join(root_path, 'resampled_larger\\')

num_classes = 3
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs
cube_size = args.cube_size
weight_decay = args.weight_decay

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def config_log(snapshot_path_tmp, typename):

    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh

# Data transforms
image_keys = ['ct', 'pt', 'gtv']  # Do not change
modes_3d = ['trilinear', 'trilinear', 'nearest']
modes_2d = ['bilinear', 'bilinear', 'nearest']
train_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='RAS'),
        Spacingd(
            keys=image_keys,
            pixdim=(1, 1, 1),
            mode=modes_2d,
        ),
        ScaleIntensityRanged(
            keys=['ct'],
            a_min=ct_a_min,
            a_max=ct_a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=['pt'],
            a_min=pt_a_min,
            a_max=pt_a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandAffined(keys=image_keys, prob=p,
                    translate_range=(round(10 * strength), round(10 * strength), round(10 * strength)),
                    padding_mode='border', mode=modes_2d),
        RandAffined(keys=image_keys, prob=p, scale_range=(0.10 * strength, 0.10 * strength, 0.10 * strength),
                    padding_mode='border', mode=modes_2d),
        RandFlipd(
            keys=image_keys,
            spatial_axis=[0],
            prob=p / 3,
        ),
        RandFlipd(
            keys=image_keys,
            spatial_axis=[1],
            prob=p / 3,
        ),
        RandFlipd(
            keys=image_keys,
            spatial_axis=[2],
            prob=p / 3,
        ),
        RandShiftIntensityd(
            keys=['ct', 'pt'],
            offsets=0.10,
            prob=p,
        ),
        ToTensord(keys=image_keys),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='RAS'),
        Spacingd(keys=image_keys, pixdim=(1, 1, 1), mode=modes_2d),
        ScaleIntensityRanged(keys=['ct'], a_min=ct_a_min, a_max=ct_a_max, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=['pt'], a_min=pt_a_min, a_max=pt_a_max, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=image_keys),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='RAS'),
        Spacingd(keys=image_keys, pixdim=(1, 1, 1), mode=modes_2d),
        ScaleIntensityRanged(keys=['ct'], a_min=ct_a_min, a_max=ct_a_max, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=['pt'], a_min=pt_a_min, a_max=pt_a_max, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=image_keys),
    ]
)

def train(train_list, val_list, test_list, fold_id=1):
    snapshot_path_tmp = os.path.join(snapshot_path, "FOLD{}".format(fold_id))
    if not os.path.exists(snapshot_path_tmp):
        os.makedirs(snapshot_path_tmp)

    handler, sh = config_log(snapshot_path_tmp, 'fold' + str(fold_id))
    logging.info(str(args))
    
    model_unet = create_model(name="Unet3D", num_classes=num_classes)
    model_resnet = create_model(name="resnet34", num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model_unet = torch.nn.DataParallel(model_unet, device_ids=device_ids).to(device)
        model_resnet = torch.nn.DataParallel(model_resnet, device_ids=device_ids).to(device)

    train_ct = [f"{patient_name}__CT.nii.gz" for patient_name in train_list]
    train_pt = [f"{patient_name}__PT.nii.gz" for patient_name in train_list]
    train_gtv = [f"{patient_name}__gtv.nii.gz" for patient_name in train_list]

    val_ct = [f"{patient_name}__CT.nii.gz" for patient_name in val_list]
    val_pt = [f"{patient_name}__PT.nii.gz" for patient_name in val_list]
    val_gtv = [f"{patient_name}__gtv.nii.gz" for patient_name in val_list]

    test_ct = [f"{patient_name}__CT.nii.gz" for patient_name in test_list]
    test_pt = [f"{patient_name}__PT.nii.gz" for patient_name in test_list]
    test_gtv = [f"{patient_name}__gtv.nii.gz" for patient_name in test_list]

    # Set semi_ratio
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(train_list)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    # Initialize DataLoader
    train_dict = [
        {'ct': os.path.join(data_path, ct), 'pt': os.path.join(data_path, pt), 'gtv': os.path.join(data_path, gtv)} for
        ct, pt, gtv in zip(train_ct, train_pt, train_gtv)]
    train_ds = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=1.0, num_workers=0)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)

    val_dict = [
        {'ct': os.path.join(data_path, ct), 'pt': os.path.join(data_path, pt), 'gtv': os.path.join(data_path, gtv)} for
        ct, pt, gtv in zip(val_ct, val_pt, val_gtv)]
    val_ds = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True)

    test_dict = [
        {'ct': os.path.join(data_path, ct), 'pt': os.path.join(data_path, pt), 'gtv': os.path.join(data_path, gtv)} for
        ct, pt, gtv in zip(test_ct, test_pt, test_gtv)]
    test_ds = CacheDataset(data=test_dict, transform=test_transforms, cache_rate=1.0, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=True)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_unet = torch.optim.AdamW(model_unet.parameters(), lr=base_lr, weight_decay=weight_decay)
    optimizer_resnet = torch.optim.AdamW(model_resnet.parameters(), lr=base_lr, weight_decay=weight_decay)

    # writer = SummaryWriter(snapshot_path_tmp)
    logging.info("{} iterations per epoch".format(len(train_loader)))

    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    iter_num = 0
    metric_all_cases = None
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            ct, pt, label_batch = sampled_batch['ct'], sampled_batch['pt'], sampled_batch['gtv']
            ct, pt, label_batch = ct.to(device), pt.to(device), label_batch.to(device)
            volume_batch = torch.concat([ct, pt], axis=1)

            model_unet.train()
            model_resnet.train()

            u_outputs = cube_utils.cross_image_partition_and_recovery(volume_batch, model_unet, cube_size,
                                                                      nb_chnls=24)
            r_outputs = cube_utils.cross_image_partition_and_recovery(volume_batch, model_resnet, cube_size,
                                                                      nb_chnls=16)

            DC_u_loss = loss_function(u_outputs[:labeled_bs], label_batch[:labeled_bs])
            DC_r_loss = loss_function(r_outputs[:labeled_bs], label_batch[:labeled_bs])

            # CDR
            u_outputs_soft = F.softmax(u_outputs[:labeled_bs], dim=1)
            r_outputs_soft = F.softmax(r_outputs[:labeled_bs], dim=1)

            u_predict = torch.argmax(u_outputs_soft, dim=1)
            r_predict = torch.argmax(r_outputs_soft, dim=1)
            diff_mask = (u_predict != r_predict).to(torch.int32)

            u_ce_dist = F.cross_entropy(u_outputs[:labeled_bs], torch.squeeze(label_batch[:labeled_bs], dim=1).long())
            r_ce_dist = F.cross_entropy(r_outputs[:labeled_bs], torch.squeeze(label_batch[:labeled_bs], dim=1).long())
            u_ce = torch.sum(diff_mask * u_ce_dist) / (torch.sum(diff_mask) + 1e-16)
            r_ce = torch.sum(diff_mask * r_ce_dist) / (torch.sum(diff_mask) + 1e-16)

            u_supervised_loss = DC_u_loss + 0.5 * u_ce
            r_supervised_loss = DC_r_loss + 0.5 * r_ce

            # Cross Pseudo Supervision for the unsupervised
            with torch.no_grad():
                _, u_label = torch.max(u_outputs[labeled_bs:], dim=1)
                u_label = torch.unsqueeze(u_label, dim=1)
                u_label = u_label.long()

                _, r_label = torch.max(r_outputs[labeled_bs:], dim=1)
                r_label = torch.unsqueeze(r_label, dim=1)
                r_label = r_label.long()

            # Calculate variance
            var_u, exp_var_u = cal_variance(u_outputs[labeled_bs:], r_outputs[labeled_bs:])
            var_r, exp_var_r = cal_variance(r_outputs[labeled_bs:], u_outputs[labeled_bs:])

            # Calculate the unsupervised loss
            DC_u_loss_unsupervised = loss_function(u_outputs[labeled_bs:], r_label)
            DC_r_loss_unsupervised = loss_function(r_outputs[labeled_bs:], u_label)

            u_unsupervised_loss = torch.mean(exp_var_u * DC_u_loss_unsupervised)
            r_unsupervised_loss = torch.mean(exp_var_r * DC_r_loss_unsupervised)

            # Total loss
            u_loss = u_supervised_loss + 0.5 * u_unsupervised_loss
            r_loss = r_supervised_loss + 0.5 * r_unsupervised_loss

            optimizer_unet.zero_grad()
            optimizer_resnet.zero_grad()
            u_loss.backward()
            r_loss.backward()
            optimizer_unet.step()
            optimizer_resnet.step()
            iter_num = iter_num + 1

            # writer.add_scalar('u_supervised_loss', u_supervised_loss, iter_num)
            # writer.add_scalar('r_supervised_loss', r_supervised_loss, iter_num)
            # writer.add_scalar('u_unsupervised_loss', u_unsupervised_loss, iter_num)
            # writer.add_scalar('r_unsupervised_loss', r_unsupervised_loss, iter_num)
            # writer.add_scalar('u_loss', u_loss, iter_num)
            # writer.add_scalar('r_loss', r_loss, iter_num)

            if iter_num % 100 == 0:
                logging.info('Fold {}, iteration {}:'
                             ' u_supervised_loss: {:.3f}, r_supervised_loss: {:.3f},'
                             ' u_unsupervised_loss: {:.3f}, r_unsupervised_loss: {:.3f},'
                             ' u_loss: {:.3f}, r_loss: {:.3f}, '
                             .format(fold_id, iter_num, u_supervised_loss, r_supervised_loss,
                                     u_unsupervised_loss, r_unsupervised_loss,
                                     u_loss, r_loss))

            if iter_num % 5000 == 0:
                epoch_iterator_val = tqdm(val_loader, dynamic_ncols=True)
                model_unet.eval()
                model_resnet.eval()
                logging.info("------------------------Validation-------------------------")
                dice_all, hd_all, asd_all, nsd_all, jc_all = validation_util.validation(epoch_iterator_val, model_unet,
                                                                                        model_resnet)

                dice_avg = (dice_all[0] + dice_all[1]) / 2
                hd_avg = (hd_all[0] + hd_all[1]) / 2
                asd_avg = (asd_all[0] + asd_all[1]) / 2
                nsd_avg = (nsd_all[0] + nsd_all[1]) / 2
                jc_avg = (jc_all[0] + jc_all[1]) / 2

                logging.info('fold{}, iteration {}, '
                             'average DSC:{:.4f} ''DSC_tumor:{:.3f} ''DSC_lymph:{:.3f} '
                             'average HD95:{:.4f} ''HD95_tumor:{:.3f} ''HD95_lymph:{:.3f} '
                             'average asd:{:.4f} ''asd_tumor:{:.3f} ''asd_lymph:{:.3f} '
                             'average nsd:{:.4f} ''nsd_tumor:{:.3f} ''nsd_lymph:{:.3f} '
                             'average jc:{:.4f} ''jc_tumor:{:.3f} ''jc_lymph:{:.3f} '
                             .format(fold_id, iter_num, dice_avg, dice_all[0], dice_all[1], hd_avg, hd_all[0], hd_all[1]
                                     , asd_avg, asd_all[0], asd_all[1], nsd_avg, nsd_all[0], nsd_all[1], jc_avg, jc_all[0], jc_all[1]))
                logging.info("--------------------------End Validation--------------------------")

                u_save_mode_path = os.path.join(snapshot_path_tmp, 'unet_iter_{}.pth'.format(iter_num))
                r_save_mode_path = os.path.join(snapshot_path_tmp, 'resnet_iter_{}.pth'.format(iter_num))

                torch.save(model_unet.state_dict(), u_save_mode_path)
                torch.save(model_resnet.state_dict(), r_save_mode_path)
                logging.info("save best u_model to {}".format(u_save_mode_path))
                logging.info("save best r_model to {}".format(r_save_mode_path))

                logging.info("-----------------------------Test---------------------------------")
                epoch_iterator_test = tqdm(test_loader, dynamic_ncols=True)

                dice_all_test, hd_all_test, asd_all_test, nsd_all_test, jc_all_test = test_util.test(epoch_iterator_test,
                                                                                        model_unet, model_resnet)
                dice_avg_test = (dice_all_test[0] + dice_all_test[1]) / 2
                hd_avg_test = (hd_all_test[0] + hd_all_test[1]) / 2
                asd_avg_test = (asd_all_test[0] + asd_all_test[1]) / 2
                nsd_avg_test = (nsd_all_test[0] + nsd_all_test[1]) / 2
                jc_avg_test = (jc_all_test[0] + jc_all_test[1]) / 2

                logging.info('fold{}, iteration {}, '
                             'average DSC:{:.4f} ''DSC_tumor:{:.3f} ''DSC_lymph:{:.3f} '
                             'average HD95:{:.4f} ''HD95_tumor:{:.3f} ''HD95_lymph:{:.3f} '
                             'average asd:{:.4f} ''asd_tumor:{:.3f} ''asd_lymph:{:.3f} '
                             'average nsd:{:.4f} ''nsd_tumor:{:.3f} ''nsd_lymph:{:.3f} '
                             'average jc:{:.4f} ''jc_tumor:{:.3f} ''jc_lymph:{:.3f} '
                             .format(fold_id, iter_num, dice_avg_test, dice_all_test[0], dice_all_test[1], hd_avg_test, hd_all_test[0],
                                     hd_all_test[1]
                                     , asd_avg_test, asd_all_test[0], asd_all_test[1], nsd_avg_test, nsd_all_test[0], nsd_all_test[1], jc_avg_test,
                                     jc_all_test[0], jc_all_test[1]))
                logging.info("-----------------------------End Test---------------------------------")

                model_unet.train()
                model_resnet.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
    # writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases


if __name__ == "__main__":

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    load_split_path = r"../data/split_1.pkl"

    with open(load_split_path) as f:
        split_dict = json.load(f)

    train_list = split_dict['train']
    val_list = split_dict['val']
    test_list = split_dict['test']

    metrics = train(train_list, val_list, test_list, fold_id=0)


