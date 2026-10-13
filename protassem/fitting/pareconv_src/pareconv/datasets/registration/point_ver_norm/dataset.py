
import csv



import random
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor
from typing import Union, Tuple, List, Dict
from functools import partial

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pareconv.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
    uniform_2_sphere
)
from pareconv.utils.registration import (
    get_correspondences,
    compute_overlap_mask
)
def compute_overlap(src: Union[np.ndarray, o3d.geometry.PointCloud],
                    tgt: Union[np.ndarray, o3d.geometry.PointCloud],
                    search_voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        if isinstance(src, np.ndarray):
            src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
            src_xyz = src
        else:
            src_pcd = src
            src_xyz = np.asarray(src.points)

        if isinstance(tgt, np.ndarray):
            tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
            tgt_xyz = tgt
        else:
            tgt_pcd = tgt
            tgt_xyz = np.asarray(tgt.points)

        tgt_corr = np.full(tgt_xyz.shape[0], -1, dtype=int)
        src_tree = o3d.geometry.KDTreeFlann(src_pcd)
        for i, t in enumerate(tgt_xyz):
            [k, idx, _] = src_tree.search_radius_vector_3d(t, search_voxel_size)
            if k > 0:
                tgt_corr[i] = idx[0]

        src_corr = np.full(src_xyz.shape[0], -1, dtype=int)
        tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
        for i, s in enumerate(src_xyz):
            [k, idx, _] = tgt_tree.search_radius_vector_3d(s, search_voxel_size)
            if k > 0:
                src_corr[i] = idx[0]

        src_indices = np.arange(len(src_corr))
        valid_src = src_corr >= 0
        mutual = np.zeros(len(src_corr), dtype=bool)
        mutual[valid_src] = (tgt_corr[src_corr[valid_src]] == src_indices[valid_src])
        src_tgt_corr = np.stack([np.nonzero(mutual)[0], src_corr[mutual]])
        has_corr_src = src_corr >= 0
        has_corr_tgt = tgt_corr >= 0
        return has_corr_src, has_corr_tgt, src_tgt_corr
    except Exception as e:
        print(f"Overlap calculation failed: {e}")
        return None, None, None
def normalize_points(points):
    r"""Normalize point cloud to a unit sphere at origin."""
    points = points - points.mean(axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))
    return points

def extract_point_cloud_data(structured_array, point_limit=30000):
    pts   = structured_array['point']
    dens  = structured_array['density']
    vec   = structured_array['vector']
    idx   = structured_array['index']

    N = pts.shape[0]
    if point_limit is not None and N > point_limit:
        perm = np.random.permutation(N)[:point_limit]
        pts  = pts[perm]
        dens = dens[perm]
        vec  = vec[perm]
        idx  = idx[perm]


    '''
    
    dens_min = dens.min()
    dens_max = dens.max()
    if dens_max > dens_min:
        dens = (dens - dens_min) / (dens_max - dens_min)
    else:
        dens = np.zeros_like(dens)

    dens = dens.reshape(-1, 1)
    pd   = np.concatenate([pts, dens], axis=1)  # (N,4)
    '''
    return pts, vec, idx

#def restore_coordinates(points, centroid, scale):
    #return points * scale + centroid
def descale_points(points, centroid, scale):
    return points * scale + centroid
def apply_transformation(points, R, t):
    return (R @ points.T).T + t
class point_vd_Dataset(Dataset):
    def __init__(self,
                 subset,
                 point_limit=None,
                 use_augmentation=False,
                 augmentation_noise=0.005,
                 augmentation_rotation=1,
                 augmentation_crop=False,
                 point_keep_ratio=1.0,
                 return_corr_indices=False,
                 matching_radius=None,
                 rotated=False):
        super().__init__()
        self.subset = subset
        self.point_limit        = point_limit
        self.use_augmentation   = use_augmentation
        self.aug_noise          = augmentation_noise
        self.aug_rotation       = augmentation_rotation
        self.augmentation_crop  = augmentation_crop
        self.point_keep_ratio   = point_keep_ratio
        self.return_corr_indices= return_corr_indices
        self.matching_radius    = matching_radius
        self.rotated            = rotated

        data_dir = f'/xiangyux/AF3-scale-train/{subset}'
        self.metadata_list = glob.glob(os.path.join(data_dir, "*.npz"))

    def __len__(self):
        return len(self.metadata_list)

    def _random_crop(self, ref, src,vec_ref, vec_src, p_keep, ref_masks, src_masks):
        """
        ref, src: (N,4) points_with_density
        ref_masks, src_masks: boolean masks on original (N,) points
        """
        def crop_one(pts4,vecs,masks):
            xyz = pts4[:, :3]
            dens= pts4[:, 3:]
            rand_dir = uniform_2_sphere()
            centroid = xyz.mean(axis=0)
            proj = (xyz - centroid) @ rand_dir
            th_high = np.percentile(proj, (1-p_keep)*100)
            th_low  = np.percentile(proj, p_keep*100)
            m1 = proj > th_high
            m2 = proj < th_low
            mask = m1 if masks[m1].sum() < masks[m2].sum() else m2
            vecs_new = vecs[mask]
            return np.concatenate([xyz[mask], dens[mask]], axis=1), vecs_new,mask

        #ref4, mask_ref = crop_one(ref, ref_masks)
        #src4, mask_src = crop_one(src, src_masks)
        ref4_new, vec_ref_new, mask_ref = crop_one(ref, vec_ref, ref_masks)
        src4_new, vec_src_new, mask_src = crop_one(src, vec_src, src_masks)
        return ref4_new, src4_new, vec_ref_new, vec_src_new, mask_ref, mask_src

    def _augment_point_cloud(self, ref4, src4, vec_ref, vec_src, rotation, translation):
        R = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref4[:, :3] = ref4[:, :3] @ R.T
            rotation    = R @ rotation
            vec_ref = vec_ref @ R.T
            translation = R @ translation
        else:
            src4[:, :3] = src4[:, :3] @ R.T
            vec_src = vec_src @ R.T
            rotation    = rotation @ R.T

        ref4[:, :3] += (np.random.rand(*ref4[:, :3].shape) - 0.5) * self.aug_noise
        src4[:, :3] += (np.random.rand(*src4[:, :3].shape) - 0.5) * self.aug_noise

        return ref4, src4,vec_ref, vec_src, rotation, translation


    def __getitem__(self, idx):
        npz = np.load(self.metadata_list[idx])
        #print("npz",self.metadata_list[idx])
        scale = npz['scale']
        s_cen_1 = npz['centroid_src']
        s_cen_1 = npz['centroid_tgt']
        op = npz['overlap']
        #print("ori",op)
        if self.subset == 'train':
            src4,vec_src, _ = extract_point_cloud_data(npz['source_structured'], 30000)
            ref4,vec_ref, _ = extract_point_cloud_data(npz['target_structured'], 30000)
        else :
            src4,vec_src, _ = extract_point_cloud_data(npz['source_structured'], 30000)
            ref4,vec_ref, _ = extract_point_cloud_data(npz['target_structured'], 30000)
        if len(src4) < 400 or len(ref4) < 400:
            # print("on_no1111")
            return self.__getitem__(random.randint(0, len(self) - 1))
        #src4 = restore_coordinates(src4, 0, scale)
        #ref4 = restore_coordinates(ref4, 0, scale)
        #R0  = np.array(npz['rotation'],    dtype=np.float32)
        #t0  = np.array(npz['translation'], dtype=np.float32)*scale

        R0  = np.array(npz['rotation'], dtype='float32')
        t0  = np.array(npz['translation'], dtype=np.float32)
        T0  = get_transform_from_rotation_translation(R0, t0)
        src4_gt = apply_transformation(src4[:, :3], R0, t0)
        _, _, corr = compute_overlap(ref4[:, :3], src4_gt,2.0/scale)
        overlap = corr.shape[1] / max(len(ref4[:, :3]), len(src4[:, :3])) if corr is not None and corr.size else 0.0
        #print("op",op)
        #print("op2", overlap)



        if self.use_augmentation:
            ref4, src4, vec_ref, vec_src, R0, t0 = self._augment_point_cloud(ref4, src4, vec_ref, vec_src,R0, t0)

        if self.rotated:
            Rr = random_sample_rotation_v2()
            ref4[:, :3]  = ref4[:, :3] @ Rr.T
            vec_ref = vec_ref @ Rr.T
            R0            = Rr @ R0
            t0            = Rr @ t0
            Rs = random_sample_rotation_v2()
            src4[:, :3]  = src4[:, :3] @ Rs.T
            vec_src = vec_src @ Rs.T
            R0            = R0 @ Rs.T

        T  = get_transform_from_rotation_translation(R0, t0)
        #overlap = compute_overlap(ref4[:, :3], src4[:, :3], T, positive_radius=2.0)

        if self.augmentation_crop and overlap > 0.6:
            ref_mask, src_mask = compute_overlap_mask(ref4[:, :3], src4[:, :3], T)
            ref4, src4, vec_ref, vec_src, _, _ = self._random_crop(
                ref4, src4, vec_ref, vec_src,
                self.point_keep_ratio, ref_mask, src_mask
            )

        src4_gt=apply_transformation(src4[:, :3],R0,t0)
        _, _, corr = compute_overlap(ref4[:, :3], src4_gt,2.0/scale)
        overlap = corr.shape[1] / max(len(ref4[:, :3]), len(src4[:, :3])) if corr is not None and corr.size else 0.0
        #print("over",overlap)
        #overlap = compute_overlap(ref4[:, :3], src4[:, :3], T, positive_radius=2.5 )
        #print("overlap11",overlap)
        if(len(src4) < 450 or len(ref4) < 450) :
            #print("on_no1111")
            return self.__getitem__(random.randint(0, len(self)-1))


        c_ref = ref4[:, :3].mean(axis=0)
        c_src = src4[:, :3].mean(axis=0)
        ref4[:, :3] -= c_ref
        src4[:, :3] -= c_src

        
        t0 = (t0 + R0 @ c_src - c_ref)
        T  = get_transform_from_rotation_translation(R0, t0)

        #print("overlap22", overlap)

        data_dict = {
            #'overlap': overlap,
            'ref_points': ref4.astype(np.float32),   # (N,4)
            'src_points': src4.astype(np.float32),   # (M,4)
            'ref_vectors': vec_ref.astype(np.float32),  # (N,3)
            'src_vectors': vec_src.astype(np.float32),  # (M,3)
            'ref_feats' : np.ones((len(ref4), 1), dtype=np.float32),
            'src_feats' : np.ones((len(src4), 1), dtype=np.float32),
            'transform': T.astype(np.float32),
            'scale':scale.astype(np.float32),
            #'overlap':overlap.astype(np.float32),
        }


        if self.return_corr_indices:
            data_dict['corr_indices'] = get_correspondences(
                ref4[:, :3], src4[:, :3], T, self.matching_radius)

        return data_dict




    




if __name__ == "__main__":
    main()
