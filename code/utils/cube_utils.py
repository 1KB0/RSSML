import torch
from networks.Unet3d import *
from networks.resnet3D import *
import math

def cross_image_partition_and_recovery(input, net, cube_size, nb_chnls):
    bs, c, w, h, d = input.shape
    nb_cubes = h // cube_size
    cube_part_ind, cube_rec_ind = get_part_and_rec_ind(volume_shape=input.shape,
                                                                  nb_cubes=nb_cubes,
                                                                  nb_chnls=nb_chnls)
    img_cross_mix = input.view(bs, c, w, h, d)
    img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
    img_cross_mix = img_cross_mix.view(bs, c, w, h, d)

    output_mix, embedding = net(img_cross_mix)
    c_ = embedding.shape[1]
    pred_rec = torch.gather(embedding, dim=0, index=cube_rec_ind)
    pred_rec = pred_rec.view(bs, c_, w, h, d)
    outputs_unmix = net.module.out_conv(pred_rec)

    return outputs_unmix, embedding


def get_part_and_rec_ind(volume_shape, nb_cubes, nb_chnls):
    bs, c, w, h, d = volume_shape

    # partition
    x = torch.cat([torch.ones(1, nb_cubes, nb_cubes, nb_cubes),
                   torch.zeros(1, nb_cubes, nb_cubes, nb_cubes)], dim=0).cuda()
    # Randomly select N positions in batch 1 to index it as 0
    indices_batch1 = torch.randperm(nb_cubes * nb_cubes * nb_cubes)[:10]
    x[0, indices_batch1 // (nb_cubes * nb_cubes), (
            indices_batch1 % (nb_cubes * nb_cubes)) // nb_cubes, indices_batch1 % nb_cubes] = 0
    # Set the relative position in the batch 2, and set the position of 0 in the batch 1 to 1
    x[1] = 1 - x[0]
    rand_loc_ind = torch.argsort(x, dim=0).cuda()
    cube_part_ind = rand_loc_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes)
    cube_part_ind = cube_part_ind.repeat_interleave(c, dim=1)
    cube_part_ind = cube_part_ind.repeat_interleave(w // nb_cubes, dim=2)
    cube_part_ind = cube_part_ind.repeat_interleave(h // nb_cubes, dim=3)
    cube_part_ind = cube_part_ind.repeat_interleave(d // nb_cubes, dim=4)

    # recovery
    rec_ind = torch.argsort(rand_loc_ind, dim=0).cuda()
    cube_rec_ind = rec_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes)
    cube_rec_ind = cube_rec_ind.repeat_interleave(nb_chnls, dim=1)
    cube_rec_ind = cube_rec_ind.repeat_interleave(w // nb_cubes, dim=2)
    cube_rec_ind = cube_rec_ind.repeat_interleave(h // nb_cubes, dim=3)
    cube_rec_ind = cube_rec_ind.repeat_interleave(d // nb_cubes, dim=4)
    return cube_part_ind, cube_rec_ind



