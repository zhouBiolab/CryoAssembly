import pdb
from functools import partial

import numpy as np
import torch

from pareconv.modules.ops import grid_subsample, radius_search
from pareconv.utils.torch import build_dataloader


from typing import Tuple, List
from pytorch3d.ops import sample_farthest_points

# Stack mode utilities
'''

def precompute_subsample(points, lengths, num_stages, voxel_size, num_neighbors, subsample_ratio):
    assert num_stages == len(num_neighbors)

    points_list = []
    lengths_list = []
    j = 0
    # grid subsampling
    for i in range(num_stages):
        #if i > 0:
            #if voxel_size == 4:
                #voxel_size = 2
                #j =1
        points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        print("voxel_size", voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        #if j == 1:
            #voxel_size = 4
            #j = 0
        voxel_size *= subsample_ratio  # 2 for 3DMatch, 2.5 for KITTI


    return {
        'points': points_list,
        'lengths': lengths_list,
    }
'''
voxel_size_list = [2, 3.6, 6.48, 11.664]
#voxel_size_list = [1, 2, 4, 11]
#voxel_size_list = [1, 1,4 ,12]
#voxel_size_list = [1, 1,3.6 ,6.48]
#voxel_size_list = [1, 3.6, 6.48, 11.664]
def farthest_point_sampling(points: torch.Tensor, lengths: torch.Tensor, target_ratio: float = 0.5) -> Tuple[
    torch.Tensor, torch.Tensor]:
    device = points.device
    batch_size = lengths.shape[0]

    xyz_points = points[:, :3]

    sampled_indices_list = []
    sampled_lengths = []

    start_idx = 0
    for b in range(batch_size):
        end_idx = start_idx + lengths[b].item()
        batch_points = xyz_points[start_idx:end_idx]
        batch_size_points = batch_points.shape[0]

        batch_num_samples = max(1, int(batch_size_points * target_ratio))

        if batch_num_samples >= batch_size_points:
            sampled_indices = torch.arange(start_idx, end_idx, device=device)
        else:
            sampled_indices = fps_single_batch(batch_points, batch_num_samples, start_idx)

        sampled_indices_list.append(sampled_indices)
        sampled_lengths.append(sampled_indices.shape[0])
        start_idx = end_idx

    all_sampled_indices = torch.cat(sampled_indices_list, dim=0)
    sampled_points = points[all_sampled_indices]
    sampled_lengths = torch.tensor(sampled_lengths, dtype=torch.long, device=device)

    return sampled_points, sampled_lengths
def farthest_point_sampling_gpu(points: torch.Tensor, lengths: torch.Tensor, target_ratio: float = 0.25) -> Tuple[
    torch.Tensor, torch.Tensor]:
    device = points.device
    batch_size = lengths.shape[0]

    sampled_points_list = []
    sampled_lengths = []

    start_idx = 0
    for b in range(batch_size):
        end_idx = start_idx + lengths[b].item()
        batch_points = points[start_idx:end_idx]
        batch_size_points = batch_points.shape[0]

        batch_num_samples = max(1, int(batch_size_points * target_ratio))

        if batch_num_samples >= batch_size_points:
            sampled_points_list.append(batch_points)
            sampled_lengths.append(batch_size_points)
        else:
            batch_points_3d = batch_points[:, :3].unsqueeze(0)

            sampled_pts, sampled_idx = sample_farthest_points(
                batch_points_3d,
                K=batch_num_samples,
                random_start_point=True
            )

            sampled_idx = sampled_idx[0]
            sampled_full_features = batch_points[sampled_idx]

            sampled_points_list.append(sampled_full_features)
            sampled_lengths.append(batch_num_samples)

        start_idx = end_idx

    sampled_points = torch.cat(sampled_points_list, dim=0)
    sampled_lengths = torch.tensor(sampled_lengths, dtype=torch.long, device=device)

    return sampled_points, sampled_lengths


def fps_single_batch(points: torch.Tensor, num_samples: int, offset: int) -> torch.Tensor:
    device = points.device
    N = points.shape[0]

    distances = torch.full((N,), float('inf'), device=device)

    farthest_idx = torch.randint(0, N, (1,), device=device).item()
    sampled_indices = [farthest_idx]

    for _ in range(num_samples - 1):
        current_point = points[farthest_idx:farthest_idx + 1]
        dist_to_current = torch.sum((points - current_point) ** 2, dim=1)

        distances = torch.minimum(distances, dist_to_current)

        farthest_idx = torch.argmax(distances).item()
        sampled_indices.append(farthest_idx)

    sampled_indices = torch.tensor(sampled_indices, dtype=torch.long, device=device) + offset

    return sampled_indices


def precompute_subsample(points, lengths, num_stages, voxel_size_list, num_neighbors, subsample_ratio):
    assert num_stages == len(num_neighbors)
    #assert len(voxel_size_list) == num_stages, "voxel_size_list must have the same length as num_stages"

    points_list = []
    lengths_list = []

    for i in range(num_stages):
        #print("i",i)
        if i != 31 and i>0  :
            #print("i")
            current_voxel_size = voxel_size_list[i]
            #print(f"Stage {i}: voxel_size = {current_voxel_size}")
            points, lengths = grid_subsample(points, lengths, voxel_size=current_voxel_size)

        elif i ==31 :
            points, lengths = farthest_point_sampling_gpu(
                points, lengths, target_ratio=0.25#0.25
            )

        points_list.append(points)
        lengths_list.append(lengths)

    return {
        'points': points_list,
        'lengths': lengths_list,
    }

def precompute_neibors(points_list, lengths_list, num_stages, num_neighbors):

    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # knn search
    for i in range(num_stages):
        cur_points = points_list[i][:, :3]
        cur_lengths = lengths_list[i]
        if i < num_stages:    # without adaptive sampling  i < num_stages, with: i < num_stages
            neighbors = radius_search(
                cur_points,
                cur_points,
                cur_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            subsampling_list.append(subsampling)

            if i > 0:
                upsampling = radius_search(
                    cur_points,
                    sub_points,
                    cur_lengths,
                    sub_lengths,
                    1,
                )
                upsampling_list.append(upsampling)
    return {
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }
#xiu gai
def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, num_neighbors, subsample_ratio, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        num_neighbors (List[int])
        precompute_data (bool)
    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)

    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)
    #print("data_vd",points.shape)
    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        with torch.no_grad():
            input_dict = precompute_subsample(points, lengths, num_stages, voxel_size_list, num_neighbors, subsample_ratio)
            torch.cuda.empty_cache()

            collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 2))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    num_neighbors,
    subsample_ratio,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            num_neighbors=num_neighbors,
            subsample_ratio=subsample_ratio,
            precompute_data=precompute_data,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
