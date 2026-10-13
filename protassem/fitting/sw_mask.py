'Generate spherical target masks.'

import numpy as np
import os
import logging
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from scipy.spatial import cKDTree

from protassem.core.points_txt import (read_point_cloud_file,
                                       write_point_cloud)


@dataclass
class MaskResult:
    id: int
    center: np.ndarray
    radius: float
    points_count: int
    point_cloud_data: np.ndarray
    mask: np.ndarray
    strategy: str = 'spherical'
    type: str = 'sphere'


class SphericalMaskGenerator:

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.setup_logging()

    def setup_logging(self):
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    @staticmethod
    def load_sample_points(file_path: str) -> np.ndarray:
        return read_point_cloud_file(file_path)

    @staticmethod
    def calculate_gyration_radius(points: np.ndarray) -> Tuple[float, np.ndarray]:
        centroid = np.mean(points, axis=0)
        squared_distances = np.sum((points - centroid) ** 2, axis=1)
        gyration_radius = np.sqrt(np.mean(squared_distances))
        return gyration_radius, centroid

    @staticmethod
    def calculate_max_radius(points: np.ndarray, centroid: np.ndarray) -> float:
        distances_squared = np.sum((points - centroid) ** 2, axis=1)
        return np.sqrt(np.max(distances_squared))

    @staticmethod
    def points_in_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        radius_squared = radius ** 2
        distances_squared = np.sum((points - center) ** 2, axis=1)
        return distances_squared <= radius_squared

    def select_initial_points(self, src_points: np.ndarray, ref_points: np.ndarray,
                              mask_radius: float, min_point_distance: float) -> np.ndarray:
        ref_centroid = np.mean(ref_points, axis=0)
        distances_squared = np.sum((ref_points - ref_centroid) ** 2, axis=1)
        ref_max_radius = np.sqrt(np.max(distances_squared))

        search_extent = min(max(0.85 * (ref_max_radius - mask_radius), 5.0), ref_max_radius)

        if self.verbose:
            print(f"Reference point-cloud maximum radius: {ref_max_radius:.3f}")
            print(f"Spherical mask radius: {mask_radius:.3f}")
            print(f"Search scope: {search_extent:.3f}")
            print(f"Minimum point spacing: {min_point_distance:.3f}")

        distances_to_centroid = np.sqrt(distances_squared)
        valid_mask = distances_to_centroid < search_extent
        candidate_points = ref_points[valid_mask]

        if len(candidate_points) == 0:
            return np.array([ref_centroid])

        mask_centers = [ref_centroid]
        
        tree = cKDTree([ref_centroid])
        
        for point in candidate_points:
            dist, _ = tree.query(point, k=1)
            
            if dist >= min_point_distance:
                mask_centers.append(point.copy())
                tree = cKDTree(mask_centers)

        return np.array(mask_centers)

    def select_initial_points_fps(self, src_points, ref_points,
                                  mask_radius, min_point_distance,
                                  max_centers=4000):
        pts = np.asarray(ref_points, dtype=np.float64)
        if len(pts) == 0:
            return np.array([np.zeros(3)])

        centroid = pts.mean(axis=0)
        seed = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))
        centers_idx = [seed]
        min_d2 = np.sum((pts - pts[seed]) ** 2, axis=1)
        thr2 = float(min_point_distance) ** 2

        while len(centers_idx) < max_centers:
            i = int(np.argmax(min_d2))
            if min_d2[i] < thr2:
                break
            centers_idx.append(i)
            d2 = np.sum((pts - pts[i]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        if self.verbose:
            print("[FPS] centers=%d (min_point_distance=%.3f, target_pts=%d)"
                  % (len(centers_idx), min_point_distance, len(pts)))
        return pts[centers_idx]

    def select_initial_points_kpconv(self, src_points, ref_points,
                                     mask_radius, min_point_distance,
                                     n_iter=50, step_factor=0.3,
                                     attract=0.3, k_neighbors=13,
                                     seed=0, max_centers=4000):
        np.random.seed(seed)
        pts = np.asarray(ref_points, dtype=np.float64)
        if len(pts) == 0:
            return np.array([np.zeros(3)])

        centers = np.asarray(
            self.select_initial_points_fps(src_points, pts, mask_radius,
                                           min_point_distance, max_centers),
            dtype=np.float64)
        if len(centers) <= 2:
            return centers

        ref_tree = cKDTree(pts)
        step = step_factor * float(min_point_distance)
        eps = 1e-6

        for _ in range(n_iter):
            ctree = cKDTree(centers)
            k = min(len(centers), k_neighbors)
            dists, nbrs = ctree.query(centers, k=k)
            move = np.zeros_like(centers)
            for col in range(1, k):
                d = dists[:, col][:, None] + eps
                move += (centers - centers[nbrs[:, col]]) / (d ** 2)
            mnorm = np.linalg.norm(move, axis=1, keepdims=True) + eps
            centers = centers + step * move / mnorm
            _, idx = ref_tree.query(centers, k=1)
            centers = centers + attract * (pts[idx] - centers)

        _, idx = ref_tree.query(centers, k=1)
        centers = np.unique(pts[idx], axis=0)

        if self.verbose:
            print("[KPConv] centers=%d (min_point_distance=%.3f, n_iter=%d)"
                  % (len(centers), min_point_distance, n_iter))
        return centers

    def generate_masks(self, src_points: np.ndarray, ref_points: np.ndarray,
                       ref_structured_data: np.ndarray,
                       mask_radius_factor: float = 1.0,
                       min_coverage: float = 0.3,
                       min_point_distance_factor: float = 0.8,
                       center_mode: str = "greedy") -> List[MaskResult]:
        if self.verbose:
            print('=== Spherical mask generation strategy ===')

        all_masks = []
        
        centroid = np.mean(ref_points, axis=0)
        max_radius = self.calculate_max_radius(ref_points, centroid)
        full_mask = np.ones(len(ref_points), dtype=bool)
        
        full_mask_result = MaskResult(
            id=0,
            center=centroid,
            radius=max_radius,
            points_count=len(ref_points),
            point_cloud_data=ref_structured_data,
            mask=full_mask,
            strategy='full',
            type='full'
        )
        
        all_masks.append(full_mask_result)
        
        if self.verbose:
            print(f"Created full-data mask (ID=0): contains {len(ref_points)} points")
            print(f"Centroid: {centroid}")
            print(f"Maximum radius: {max_radius:.3f}")

        gyration_radius, src_centroid = self.calculate_gyration_radius(src_points)
        mask_radius = gyration_radius * mask_radius_factor

        min_point_distance = mask_radius * min_point_distance_factor

        min_points_threshold = int(len(src_points) * min_coverage)

        if self.verbose:
            print(f"\n=== Generating local spherical masks ===")
            print(f"Source point cloud contains {len(src_points)} points")
            print(f"Source point-cloud radius of gyration: {gyration_radius:.3f}")
            print(f"Spherical mask radius: {mask_radius:.3f}")
            print(f"Minimum point-count threshold: {min_points_threshold}")

        if center_mode == "fps" and hasattr(self, "select_initial_points_fps"):
            mask_centers = self.select_initial_points_fps(src_points, ref_points, mask_radius, min_point_distance)
        elif center_mode == "kpconv" and hasattr(self, "select_initial_points_kpconv"):
            mask_centers = self.select_initial_points_kpconv(src_points, ref_points, mask_radius, min_point_distance)
        else:
            mask_centers = self.select_initial_points(src_points, ref_points, mask_radius, min_point_distance)

        if self.verbose:
            print(f"Found {len(mask_centers)} potential mask centers")

        mask_id = 1
        
        radius_squared = mask_radius ** 2
        
        for center in mask_centers:
            distances_squared = np.sum((ref_points - center) ** 2, axis=1)
            mask = distances_squared <= radius_squared
            points_count = np.sum(mask)

            if points_count >= min_points_threshold:
                masked_data = ref_structured_data[mask]

                mask_result = MaskResult(
                    id=mask_id,
                    center=center,
                    radius=mask_radius,
                    points_count=points_count,
                    point_cloud_data=masked_data,
                    mask=mask,
                    strategy='spherical',
                    type='sphere'
                )

                all_masks.append(mask_result)
                mask_id += 1

        if self.verbose:
            print(f"Generated {(len(all_masks) - 1)} valid spherical masks")
            print(f"Total masks: {len(all_masks)} (including 1 full mask)")

        return all_masks


def generate_spherical_masks(source_data: Union[str, np.ndarray],
                             target_data: Union[str, np.ndarray],
                             mask_radius_factor: float = 1.0,
                             min_coverage: float = 0.3,
                             min_point_distance_factor: float = 0.8,
                             verbose: bool = True,
                             center_mode: str = "greedy") -> List[MaskResult]:
    generator = SphericalMaskGenerator(verbose=verbose)

    if isinstance(source_data, str):
        src_structured = generator.load_sample_points(source_data)
        src_points = src_structured['point']
    else:
        src_points = source_data
        src_structured = None

    if isinstance(target_data, str):
        tgt_structured = generator.load_sample_points(target_data)
        ref_points = tgt_structured['point']
    else:
        ref_points = target_data
        if target_data.shape[1] == 3:
            n_points = len(target_data)
            dtype = [('index', np.int32),
                     ('point', np.float32, (3,)),
                     ('vector', np.float32, (3,)),
                     ('density', np.float32)]
            tgt_structured = np.zeros(n_points, dtype=dtype)
            tgt_structured['index'] = np.arange(n_points)
            tgt_structured['point'] = target_data
            tgt_structured['vector'] = np.zeros((n_points, 3), dtype=np.float32)
            tgt_structured['density'] = np.ones(n_points, dtype=np.float32)
        else:
            raise ValueError('NumPy coordinate arrays must have shape (N, 3)')

    if verbose:
        print(f"Source point-cloud shape: {src_points.shape}")
        print(f"Target point-cloud shape: {ref_points.shape}")

    masks = generator.generate_masks(
        src_points, ref_points, tgt_structured,
        mask_radius_factor=mask_radius_factor,
        min_coverage=min_coverage,
        min_point_distance_factor=min_point_distance_factor,
        center_mode=center_mode
    )

    return masks


def save_masks(masks: List[MaskResult], output_dir: str,
               save_individual_files: bool = True) -> str:
    os.makedirs(output_dir, exist_ok=True)

    all_centers = np.array([mask.center for mask in masks])
    centers_path = os.path.join(output_dir, "all_mask_centers.txt")
    np.savetxt(centers_path, all_centers,
               header="Mask centers coordinates (x y z)", fmt='%.6f')

    if save_individual_files:
        for mask in masks:
            mask_path = os.path.join(output_dir, f"masked_points_{mask.id:04d}.txt")
            _save_point_cloud_as_txt(mask.point_cloud_data, mask_path)

    report_path = os.path.join(output_dir, "masks_report.txt")
    
    report_lines = [
        '# Spherical mask generation report\n\n',
        f"Total masks: {len(masks)}\n",
        f"First mask (ID=0): full data ({masks[0].points_count} points)\n"
    ]
    
    if len(masks) > 1:
        report_lines.append(f"Spherical mask radius: {masks[1].radius:.3f}\n\n")
    else:
        report_lines.append("\n")
    
    report_lines.append("ID\tType\tCenter_X\tCenter_Y\tCenter_Z\tPoints_Count\tRadius\n")
    
    for mask in masks:
        center = mask.center
        report_lines.append(
            f"{mask.id}\t{mask.type}\t{center[0]:.3f}\t{center[1]:.3f}\t"
            f"{center[2]:.3f}\t{mask.points_count}\t{mask.radius:.3f}\n"
        )
    
    with open(report_path, 'w') as f:
        f.writelines(report_lines)

    print(f"Saved {len(masks)} masks to {output_dir}")
    print(f"  - 1 full-data mask")
    print(f"  - {(len(masks) - 1)} spherical masks")
    return output_dir


def _save_point_cloud_as_txt(structured_data: np.ndarray, output_path: str):
    write_point_cloud(output_path,
                      points=structured_data["point"],
                      vectors=structured_data["vector"],
                      densities=structured_data["density"],
                      indices=structured_data["index"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Spherical mask generation example')
    parser.add_argument("--source", required=True, help='Source point-cloud TXT (chain template)')
    parser.add_argument("--target", required=True, help='Target point-cloud TXT (density map)')
    parser.add_argument("--out", default=None, help='Mask output directory (not saved by default)')
    args = parser.parse_args()

    masks = generate_spherical_masks(
        args.source, args.target,
        mask_radius_factor=1.45,
        min_coverage=0.3,
        min_point_distance_factor=0.45,
        verbose=True
    )

    print('\nGenerated masks:')
    print('masks[0] (full data): %d points, type=%s' % (masks[0].points_count, masks[0].type))
    for i in range(1, len(masks)):
        print('masks[%d] (spherical mask): %d points, radius=%.3f'
              % (i, masks[i].points_count, masks[i].radius))

    if args.out:
        save_masks(masks, args.out)
