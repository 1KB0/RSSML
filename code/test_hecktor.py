import json
import logging
import os
import argparse
import torch
from tqdm import tqdm

from networks.resnet3D import Resnet34
from networks.Unet3d import BaselineUNet

import numpy as np
from medpy import metric
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
import SimpleITK as sitk


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
parser.add_argument('--root_path', type=str, default='../hecktor_monai/resampled_larger', help='Name of Experiment')  # todo change dataset path
parser.add_argument('--model', type=str,  default="U+R+CPS", help='model_name')                # todo change test model name
FLAGS = parser.parse_args()

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

data_path = FLAGS.root_path

snapshot_path = "../model/HECKTOR_DoubleNet_74labeled_cons0.2_cube_size36/U+R/" + FLAGS.model+'/'

num_classes = 3
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
dice_metric = DiceMetric(include_background=False, reduction='none', get_not_nans=False)

def create_model(name='Unet3D', num_classes=3):
    # Network definition
    if name == "Unet3D":
        net = BaselineUNet(in_channels=2, n_cls=num_classes, n_filters=24)
    if name == "resnet34":
        net = Resnet34(n_channels=2, n_classes=num_classes)
    return net


def test_for_nii(epoch_iterator_test, unet, resnet):
    total_metric = []
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_test):
            ct, pt, label = (batch['ct'].cuda(), batch['pt'].cuda(), batch['gtv'].cuda())
            inputs = torch.concat([ct, pt], axis=1)

            u_outputs, _ = unet(inputs)
            r_outputs, _ = resnet(inputs)
            outputs = (u_outputs + r_outputs) / 2
          
            label_list = decollate_batch(label)
            label_convert = [post_label(label_tensor) for label_tensor in label_list]
            outputs_list = decollate_batch(outputs)
            output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
            # Compute dice
            dice_metric(y_pred=output_convert, y=label_convert)
            # Compute other metrics
            outputs_numpy = outputs.cpu().data.numpy()
            label_numpy = label.cpu().data.numpy()
            outputs_numpy =outputs_numpy[0, :, :, :, :]
            label_numpy = label_numpy[0, :, :, :, :]
            prediction = np.argmax(outputs_numpy, axis=0)
            mask = np.squeeze(label_numpy, axis=0)

            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, mask == i)
            total_metric.append(np.expand_dims(case_metric, axis=0))

        all_metric = np.concatenate(total_metric, axis=0)
        avg_hd, avg_asd, avg_nsd, avg_jc = np.mean(all_metric, axis=0)[0], np.mean(all_metric, axis=0)[1], \
            np.mean(all_metric, axis=0)[2], np.mean(all_metric, axis=0)[3]

        dice_values_per_class = dice_metric.aggregate().tolist()
        dice_values_array = np.array(dice_values_per_class)
        avg_dice = np.nanmean(dice_values_array, axis=0)
        dice_metric.reset()

    return avg_dice, avg_hd, avg_asd, avg_nsd, avg_jc


def cal_metric(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        sf = compute_surface_distances(gt, pred, spacing_mm=(1., 1., 1.))
        nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.)
        jc = metric.binary.jc(pred, gt)
        return np.array([hd95, asd, nsd, jc])
    else:
        return np.zeros(4)

load_split_path = r"../data/split_1.pkl"

with open(load_split_path) as f:
    split_dict = json.load(f)

test_list = split_dict['test']
print(test_list)

image_keys = ['ct', 'pt', 'gtv']  # Do not change
modes_3d = ['trilinear', 'trilinear', 'nearest']
modes_2d = ['bilinear', 'bilinear', 'nearest']

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

model_unet = create_model(name="Unet3D", num_classes=num_classes)
model_resnet = create_model(name="resnet34", num_classes=num_classes)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model_unet = torch.nn.DataParallel(model_unet, device_ids=device_ids).to(device)
    model_resnet = torch.nn.DataParallel(model_resnet, device_ids=device_ids).to(device)

test_ct = [f"{patient_name}__CT.nii.gz" for patient_name in test_list]
test_pt = [f"{patient_name}__PT.nii.gz" for patient_name in test_list]
test_gtv = [f"{patient_name}__gtv.nii.gz" for patient_name in test_list]

test_dict = [
    {'ct': os.path.join(data_path, ct), 'pt': os.path.join(data_path, pt), 'gtv': os.path.join(data_path, gtv)} for
    ct, pt, gtv in zip(test_ct, test_pt, test_gtv)]
test_ds = CacheDataset(data=test_dict, transform=test_transforms, cache_rate=1.0, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                         pin_memory=True)


def test_calculate_metric(epoch_num):
    epoch_iterator_test = tqdm(test_loader, dynamic_ncols=True)

    u_save_mode_path = os.path.join(snapshot_path, 'unet_iter_' + str(epoch_num) + '.pth')
    model_unet.load_state_dict(torch.load(u_save_mode_path))
    print("init weight from {}".format(u_save_mode_path))
    model_unet.eval()

    r_save_mode_path = os.path.join(snapshot_path, 'resnet_iter_' + str(epoch_num) + '.pth')
    model_resnet.load_state_dict(torch.load(r_save_mode_path))
    print("init weight from {}".format(r_save_mode_path))
    model_resnet.eval()

    dice_all_test, hd_all_test, asd_all_test, nsd_all_test, jc_all_test = test_for_nii(epoch_iterator_test,
                                                                                         model_unet, model_resnet)
    dice_avg_test = (dice_all_test[0] + dice_all_test[1]) / 2
    hd_avg_test = (hd_all_test[0] + hd_all_test[1]) / 2
    asd_avg_test = (asd_all_test[0] + asd_all_test[1]) / 2
    nsd_avg_test = (nsd_all_test[0] + nsd_all_test[1]) / 2
    jc_avg_test = (jc_all_test[0] + jc_all_test[1]) / 2

    print('average DSC:{:.4f} '
          'DSC_tumor:{:.4f} '
          'DSC_lymph:{:.4f} '
          'average HD95:{:.4f} '
          'HD95_tumor:{:.4f} '
          'HD95_lymph:{:.4f} '
          'average asd:{:.4f} '
          'asd_tumor:{:.4f} '
          'asd_lymph:{:.4f} '
          'average nsd:{:.4f} '
          'nsd_tumor:{:.4f} '
          'nsd_lymph:{:.4f} '
          'average jc:{:.4f} '
          'jc_tumor:{:.4f} '
          'jc_lymph:{:.4f} '
          .format(dice_avg_test, dice_all_test[0], dice_all_test[1],
                  hd_avg_test, hd_all_test[0], hd_all_test[1],
                  asd_avg_test, asd_all_test[0], asd_all_test[1],
                  nsd_avg_test, nsd_all_test[0], nsd_all_test[1],
                  jc_avg_test, jc_all_test[0], jc_all_test[1]))


if __name__ == '__main__':
    iters = 50000
    test_calculate_metric(iters)
