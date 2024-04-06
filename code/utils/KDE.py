import json
import os
import sys
import seaborn as sns
from matplotlib import pyplot as plt
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


snapshot_path = "../model" + "/{}_{}_{}labeled_cons{}_cube_size{}/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                             args.consistency, args.cube_size,
                                                                             args.model)
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
p_num = 500
bw_ad = 0.5
line_wid = 18
snapshot_path = '../model/HECKTOR_DoubleNet_74labeled_cons0.2_cube_size36/U+R/FOLD0/'

def get_masks(output):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    return probs


def plot_kde(feature, RAR_pred, labels, specific_c, f_dim, pic_num):
    total_pixel, total_fdim = feature.shape[0], feature.shape[1]
    labeled_pixel = int(total_pixel / 2) + 1
    save_path = f"../KDE/{f_dim}/labeled_{args.labelnum}/class_{specific_c}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # # Chose specific class pixels:
    # BCP_pred=[1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    l_pred, u_pred = np.where(RAR_pred[:labeled_pixel,:]==specific_c), np.where(RAR_pred[labeled_pixel:,:]==specific_c)
    l_lab, u_lab = np.where(labels[:labeled_pixel,:]==specific_c), np.where(labels[labeled_pixel:,:]==specific_c)
    correct_cor_l, correct_cor_u = np.intersect1d(l_pred[0], l_lab[0]), np.intersect1d(u_pred[0], u_lab[0]) + labeled_pixel
    # l_lab, u_lab = np.where(labels[:labeled_pixel,]==specific_c), np.where(labels[labeled_pixel:,]==specific_c)
    # l_len, u_len = len(l_lab[0]), len(u_lab[0])
    pixel_num = min(len(correct_cor_l), len(correct_cor_u), p_num)
    #l_lab, u_lab = l_lab[0], u_lab[0] + labeled_pixel
    print(f"Total {pixel_num} pixels for class {specific_c}")
    feature_l, feature_u = np.mean(feature[correct_cor_l[:pixel_num],], axis=1), np.mean(feature[correct_cor_u[:pixel_num],], axis=1)

    # method_name_list = ["RAR"]
    feature_list = [feature_l, feature_u]

    plt.figure()
    fig = plt.figure(figsize=(15, 15))
    sns.set_context("notebook", font_scale=2)
    for i in range(0, 1):
        plt.subplot(1, 1, i + 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.3, hspace=None)
        sns.kdeplot(feature_list[0], bw_adjust=bw_ad, color='g', linewidth=line_wid)
        sns.kdeplot(feature_list[1], bw_adjust=bw_ad, color='b', linewidth=line_wid)
        plt.xticks(ticks=plt.xticks()[0][::2], size=60, fontsize=60)
        plt.yticks(ticks=plt.yticks()[0][::2], size=60, fontsize=60)

        plt.ylabel(" ")
        # plt.title(method_name_list[i])

    plt.savefig(f"../KDE/{f_dim}/labeled_{args.labelnum}/class_{specific_c}/kde_test_mean{pic_num}_{args.labelnum}_{specific_c}.png")
    print(f"Save to: ../KDE/{f_dim}/labeled_{args.labelnum}/class_{specific_c}/kde_test_mean{pic_num}_{args.labelnum}_{specific_c}.png")
    plt.clf()

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
        # model_resnet = torch.nn.DataParallel(model_resnet, device_ids=device_ids).to(device)

    u_save_mode_path = os.path.join(snapshot_path, 'unet_iter_' + str(5000) + '.pth')
    model_unet.load_state_dict(torch.load(u_save_mode_path))
    print("init weight from {}".format(u_save_mode_path))

    train_ct = [f"{patient_name}__CT.nii.gz" for patient_name in train_list]
    train_pt = [f"{patient_name}__PT.nii.gz" for patient_name in train_list]
    train_gtv = [f"{patient_name}__gtv.nii.gz" for patient_name in train_list]

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

    logging.info("{} iterations per epoch".format(len(train_loader)))

    iter_num = 0
    metric_all_cases = None
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    picture_number = 0

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            ct, pt, label_batch = sampled_batch['ct'], sampled_batch['pt'], sampled_batch['gtv']
            ct, pt, label_batch = ct.to(device), pt.to(device), label_batch.to(device)
            label_batch = torch.squeeze(label_batch, dim=1)
            volume_batch = torch.concat([ct, pt], axis=1)
            label_batch = label_batch.detach().cpu().numpy()

            model_unet.eval()
            # model_resnet.train()

            pred, feature = cube_utils.cross_image_partition_and_recovery(volume_batch, model_unet, cube_size,
                                                                      nb_chnls=24)
            # r_outputs = cube_utils.cross_image_partition_and_recovery(volume_batch, model_resnet, cube_size,
            #                                                           nb_chnls=16)
            R_pred = get_masks(pred)

            f_dim, x_, y_, z_ = feature.shape[1], feature.shape[2], feature.shape[3], feature.shape[4]

            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(-1, f_dim)  # 1000, 16

            resized_label = np.zeros((args.batch_size, x_, y_, z_))

            for i in range(args.batch_size):
                resized_label[i] = label_batch[i].copy()

            # resized_label = cv2.resize(label_batch, (x_, y_))
            label_batch = torch.from_numpy(resized_label).cuda()
            label = label_batch.view(-1, 1)  # a (3, 1) b[a, :]
            RAR_pred = R_pred.view(-1, 1)

            feature = feature.detach().cpu().numpy()
            RAR_pred = RAR_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            # try:
            # for spi_c in range(1, 4):
            spi_c = 2
            plot_kde(feature, RAR_pred, label, spi_c, f_dim, picture_number)
            picture_number += 1
            iter_num = iter_num + 1

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
    # 读取划分结果
    with open(load_split_path) as f:
        split_dict = json.load(f)

    # 获取训练集、验证集和测试集的索引列表
    train_list = split_dict['train']
    val_list = split_dict['val']
    test_list = split_dict['test']

    metrics = train(train_list, val_list, test_list, fold_id=0)