import torch
import numpy as np
from medpy import metric
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance


n_classes = 3
post_label = AsDiscrete(to_onehot=n_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=n_classes)
dice_metric = DiceMetric(include_background=False, reduction='none', get_not_nans=False)

def validation(epoch_iterator_val, unet, resnet):
    total_metric = []
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            ct, pt, label = (batch['ct'].cuda(), batch['pt'].cuda(), batch['gtv'].cuda())
            inputs = torch.concat([ct, pt], axis=1)

            u_outputs, _ = unet(inputs)
            r_outputs, _ = resnet(inputs)
            outputs = (u_outputs + r_outputs)/2

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

            case_metric = np.zeros((4, n_classes - 1))
            for i in range(1, n_classes):
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