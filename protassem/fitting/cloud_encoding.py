'Build reusable point-cloud geometry and encodings.'

import hashlib
import struct
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from pareconv.extensions.pointops.functions import pointops
from pareconv.modules.ops import grid_subsample, point_to_node_partition
from pareconv.utils.data_mask import farthest_point_sampling_gpu

from protassem.fitting.feature_cache import ByteLruCache

GEOMETRY_VERSION = 1
FPS_TARGET_RATIO = 0.25
DETERMINISTIC_SAMPLING = ("voxel",)
ENCODING_CACHE_VERSION = 1


class GeometryError(ValueError):
    pass


@dataclass
class NodePartition:

    masks: torch.Tensor          # (N_c,) bool
    knn_indices: torch.Tensor
    knn_masks: torch.Tensor      # (N_c, K) bool


@dataclass
class CloudGeometry:
    'Point-cloud geometry shared across registrations.'

    points: List[torch.Tensor]
    lengths: List[torch.Tensor]
    features: torch.Tensor                  # (N_0, C)
    voxel_sizes: Tuple[float, ...]
    sampling_method: str
    centroid: Optional[np.ndarray] = None
    neighbors: List[torch.Tensor] = field(default_factory=list)
    subsampling: List[torch.Tensor] = field(default_factory=list)
    upsampling: List[Optional[torch.Tensor]] = field(default_factory=list)
    node_partition: Optional[NodePartition] = None
    _fingerprint: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def device(self):
        return self.points[0].device

    @property
    def num_stages(self):
        return len(self.points)

    @property
    def stage_counts(self):
        return [int(item.shape[0]) for item in self.points]

    @property
    def point_count(self):
        return int(self.points[0].shape[0])

    def fingerprint(self):
        if self._fingerprint is not None:
            return self._fingerprint
        digest = hashlib.sha256()
        digest.update(("geometry_version=%d;" % GEOMETRY_VERSION).encode())
        digest.update(("stages=%d;voxels=%s;sampling=%s;"
                       % (self.num_stages,
                          ",".join("%.6f" % value for value in self.voxel_sizes),
                          self.sampling_method)).encode())
        for index, item in enumerate(self.points):
            array = item.detach().cpu().numpy()
            digest.update(("stage=%d;dtype=%s;shape=%s;" % (index, array.dtype,
                                                            array.shape)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        features = self.features.detach().cpu().numpy()
        digest.update(("features;dtype=%s;shape=%s;" % (features.dtype, features.shape)).encode())
        digest.update(np.ascontiguousarray(features).tobytes())
        self._fingerprint = digest.hexdigest()
        return self._fingerprint


def _check_points(points, features):
    if not isinstance(points, torch.Tensor) or points.dim() != 2:
        raise GeometryError('points must be a two-dimensional tensor (N, D)')
    if points.shape[0] == 0:
        raise GeometryError('points is empty')
    if not isinstance(features, torch.Tensor) or features.dim() != 2:
        raise GeometryError('features must be a two-dimensional tensor (N, C)')
    if features.shape[0] != points.shape[0]:
        raise GeometryError('features rows %d do not match points count %d'
                            % (features.shape[0], points.shape[0]))
    if points.dim() < 2 or points.shape[1] < 3:
        raise GeometryError('points must contain at least 3 coordinate columns')


def _check_sampling(voxel_sizes, sampling_method):
    num_stages = len(voxel_sizes)
    if num_stages < 2:
        raise GeometryError('Stage count %d < 2: multi-scale geometry requires at least 2 stages' % num_stages)
    if sampling_method not in ("voxel", "fps"):
        raise GeometryError('Unknown sampling mode: %r (supported: voxel/fps)' % (sampling_method,))
    return num_stages


def _check_neighbors(num_neighbors, num_stages):
    if len(num_neighbors) != num_stages:
        raise GeometryError('num_neighbors length %d does not match stage count %d'
                            % (len(num_neighbors), num_stages))


def _stage_voxel_size(voxel_sizes, stage, num_stages, sampling_method):
    if stage < len(voxel_sizes) - 1:
        return voxel_sizes[stage]
    if stage == num_stages - 1 and sampling_method == "fps":
        return None
    return voxel_sizes[-1]


def build_stage_points(points, features, voxel_sizes, sampling_method,
                       centroid=None, device="cuda"):
    _check_points(points, features)
    num_stages = _check_sampling(voxel_sizes, sampling_method)
    if points.is_cuda:
        raise GeometryError('build_stage_points requires CPU points (grid_subsampling is a CPU implementation)')

    points_list = [points]
    cur_points = points
    for stage in range(1, num_stages):
        lengths = torch.tensor([cur_points.shape[0]], dtype=torch.long)
        voxel_size = _stage_voxel_size(voxel_sizes, stage, num_stages, sampling_method)
        if voxel_size is None:
            cur_points, _ = farthest_point_sampling_gpu(cur_points, lengths, FPS_TARGET_RATIO)
        else:
            cur_points, _ = grid_subsample(cur_points, lengths, voxel_size)
        points_list.append(cur_points)

    moved = [item.to(device) for item in points_list]
    lengths = [torch.tensor([item.shape[0]], dtype=torch.long, device=device)
               for item in moved]
    return CloudGeometry(points=moved, lengths=lengths, features=features.to(device),
                         voxel_sizes=tuple(float(value) for value in voxel_sizes),
                         sampling_method=sampling_method,
                         centroid=None if centroid is None else np.asarray(centroid),
                         upsampling=[None] * num_stages)


def build_neighbors(geometry, num_neighbors):
    num_stages = geometry.num_stages
    if _check_sampling(geometry.voxel_sizes, geometry.sampling_method) != num_stages:
        raise GeometryError('voxel_sizes length does not match geometry stage count %d' % num_stages)
    _check_neighbors(num_neighbors, num_stages)
    if not geometry.points[0].is_cuda:
        raise GeometryError('build_neighbors requires CUDA points (pointops.knnquery_heap is CUDA-only)')

    neighbors: List[torch.Tensor] = []
    subsampling: List[torch.Tensor] = []
    upsampling: List[Optional[torch.Tensor]] = [None] * num_stages
    for stage in range(num_stages):
        cur = geometry.points[stage][:, :3].contiguous().unsqueeze(0)
        neighbors.append(pointops.knnquery_heap(num_neighbors[stage], cur, cur).squeeze(0))
        if stage < num_stages - 1:
            sub = geometry.points[stage + 1][:, :3].contiguous().unsqueeze(0)
            subsampling.append(pointops.knnquery_heap(num_neighbors[stage], cur, sub).squeeze(0))
            if stage > 0:
                upsampling[stage] = pointops.knnquery_heap(1, sub, cur).squeeze(0)
    geometry.neighbors = neighbors
    geometry.subsampling = subsampling
    geometry.upsampling = upsampling
    return geometry


def build_geometry(points, features, voxel_sizes, sampling_method, num_neighbors,
                   centroid=None, device="cuda"):
    geometry = build_stage_points(points, features, voxel_sizes, sampling_method,
                                  centroid=centroid, device=device)
    return build_neighbors(geometry, num_neighbors)


def node_partition(geometry, num_points_in_patch, fine_stage=1, coarse_stage=-1):
    if geometry.num_stages < 2:
        raise GeometryError('Node partitioning requires at least 2 stages')
    fine = geometry.points[fine_stage][:, :3].contiguous()
    coarse = geometry.points[coarse_stage][:, :3].contiguous()
    _, masks, knn_indices, knn_masks = point_to_node_partition(fine, coarse, num_points_in_patch)
    return NodePartition(masks=masks, knn_indices=knn_indices, knn_masks=knn_masks)


def attach_node_partition(geometry, num_points_in_patch, fine_stage=1, coarse_stage=-1):
    geometry.node_partition = node_partition(geometry, num_points_in_patch,
                                             fine_stage=fine_stage, coarse_stage=coarse_stage)
    return geometry


def backbone_input(geometry, scale, device=None):
    device = geometry.device if device is None else device
    return {
        "points": geometry.points,
        "neighbors": geometry.neighbors,
        "subsampling": geometry.subsampling,
        "upsampling": [item for item in geometry.upsampling if item is not None],
        "scale": torch.as_tensor(scale, device=device),
    }


def offset_indices(indices, offset, source_count):
    if offset == 0:
        return indices
    if source_count >= indices.shape[-1]:
        return indices + offset
    slot = torch.arange(indices.shape[-1], device=indices.device)
    valid = (slot < source_count).unsqueeze(0).expand_as(indices)
    return torch.where(valid, indices + offset, torch.zeros_like(indices))


def join_geometries(ref, src, scale, transform):
    if ref.num_stages != src.num_stages:
        raise GeometryError('Stage counts differ: %d vs %d' % (ref.num_stages, src.num_stages))
    if ref.device != src.device:
        raise GeometryError('Geometry devices differ: %s vs %s' % (ref.device, src.device))
    for name in ("neighbors", "subsampling"):
        if len(getattr(ref, name)) != len(getattr(src, name)):
            raise GeometryError('%s lengths differ: %d vs %d'
                                % (name, len(getattr(ref, name)), len(getattr(src, name))))

    num_stages = ref.num_stages
    device = ref.device
    points, lengths = [], []
    neighbors, subsampling, upsampling = [], [], []
    for stage in range(num_stages):
        ref_points = ref.points[stage]
        src_points = src.points[stage]
        points.append(torch.cat([ref_points, src_points], dim=0))
        lengths.append(torch.tensor([ref_points.shape[0], src_points.shape[0]],
                                    dtype=torch.long, device=device))
        neighbors.append(torch.cat([
            ref.neighbors[stage],
            offset_indices(src.neighbors[stage], ref_points.shape[0], src_points.shape[0])],
            dim=0))
        if stage < num_stages - 1:
            ref_next = ref.points[stage + 1]
            src_next = src.points[stage + 1]
            subsampling.append(torch.cat([
                ref.subsampling[stage],
                offset_indices(src.subsampling[stage], ref_points.shape[0],
                               src_points.shape[0])], dim=0))
            if stage > 0:
                upsampling.append(torch.cat([
                    ref.upsampling[stage],
                    offset_indices(src.upsampling[stage], ref_next.shape[0],
                                   src_next.shape[0])], dim=0))

    return {
        "points": points,
        "lengths": lengths,
        "features": torch.cat([ref.features, src.features], dim=0),
        "neighbors": neighbors,
        "subsampling": subsampling,
        "upsampling": upsampling,
        "transform": transform.to(device) if isinstance(transform, torch.Tensor) else transform,
        "scale": scale,
        "batch_size": 1,
    }


# ======================================================================
# ======================================================================

def geometry_tensors(geometry):
    tensors = list(geometry.points) + list(geometry.lengths) + [geometry.features]
    tensors += list(geometry.neighbors) + list(geometry.subsampling)
    tensors += [item for item in geometry.upsampling if item is not None]
    if geometry.node_partition is not None:
        tensors += [geometry.node_partition.masks, geometry.node_partition.knn_indices,
                    geometry.node_partition.knn_masks]
    return tensors


def geometry_bytes(geometry):
    total = 0
    for tensor in geometry_tensors(geometry):
        total += tensor.numel() * tensor.element_size()
    return total


def geometry_to(geometry, device, copy=False):
    def move(tensor):
        tensor = tensor.to(device)
        return tensor.clone() if copy else tensor

    partition = None
    if geometry.node_partition is not None:
        partition = NodePartition(
            masks=move(geometry.node_partition.masks),
            knn_indices=move(geometry.node_partition.knn_indices),
            knn_masks=move(geometry.node_partition.knn_masks))
    return replace(
        geometry,
        points=[move(item) for item in geometry.points],
        lengths=[move(item) for item in geometry.lengths],
        features=move(geometry.features),
        neighbors=[move(item) for item in geometry.neighbors],
        subsampling=[move(item) for item in geometry.subsampling],
        upsampling=[None if item is None else move(item) for item in geometry.upsampling],
        node_partition=partition)


def geometry_cache_key(points, features, voxel_sizes, sampling_method, num_neighbors,
                       centroid=None):
    digest = hashlib.sha256()
    digest.update(("geometry_version=%d;sampling=%s;voxels=%s;neighbors=%s;"
                   % (GEOMETRY_VERSION, sampling_method,
                      ",".join("%.6f" % float(value) for value in voxel_sizes),
                      ",".join(str(int(value)) for value in num_neighbors))).encode())
    for name, tensor in (("points", points), ("features", features)):
        array = tensor.detach().cpu().numpy()
        digest.update(("%s;dtype=%s;shape=%s;" % (name, array.dtype, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    if centroid is not None:
        digest.update(np.ascontiguousarray(np.asarray(centroid, dtype=np.float32)).tobytes())
    return digest.hexdigest()


@dataclass
class AcquireResult:

    geometry: CloudGeometry
    hit: bool
    cacheable: bool


class GeometryCache:

    def __init__(self, capacity_bytes, device="cuda", name="geometry"):
        self.device = device
        self._cache = ByteLruCache(capacity_bytes, geometry_bytes, name=name)

    @property
    def enabled(self):
        return self._cache.enabled

    @property
    def capacity_bytes(self):
        return self._cache.capacity_bytes

    def get(self, key):
        return self._cache.get(key)

    def put(self, key, geometry):
        return self._cache.put(key, geometry_to(geometry, "cpu", copy=True))

    def snapshot(self):
        return self._cache.snapshot()


def acquire_geometry(cache, points, features, voxel_sizes, sampling_method, num_neighbors,
                     centroid=None, device="cuda"):
    cacheable = sampling_method in DETERMINISTIC_SAMPLING
    key = None
    if cache is not None and cacheable:
        key = geometry_cache_key(points, features, voxel_sizes, sampling_method,
                                 num_neighbors, centroid)
        cached = cache.get(key)
        if cached is not None:
            return AcquireResult(geometry=geometry_to(cached, device),
                                 hit=True, cacheable=cacheable)

    geometry = build_stage_points(points, features, voxel_sizes, sampling_method,
                                  centroid=centroid, device=device)
    build_neighbors(geometry, num_neighbors)
    if key is not None:
        cache.put(key, geometry)

    return AcquireResult(geometry=geometry, hit=False, cacheable=cacheable)


# ======================================================================
# ======================================================================

def object_tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(object_tensor_bytes(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return sum(object_tensor_bytes(getattr(value, name))
                   for name in value.__dataclass_fields__)
    return 0


def encoding_cache_key(geometry_key, scale, model_fingerprint, dtype,
                       version=ENCODING_CACHE_VERSION):
    digest = hashlib.sha256()
    digest.update(("encoding_version=%d;" % version).encode())
    digest.update(("geometry=%s;" % geometry_key).encode())
    digest.update(("scale_bits=%s;" % struct.pack(">f", float(scale)).hex()).encode())
    digest.update(("model=%s;" % model_fingerprint).encode())
    digest.update(("dtype=%s;" % dtype).encode())
    return digest.hexdigest()


class EncodingCache:

    def __init__(self, capacity_bytes, model_fingerprint, name="encoding"):
        self.model_fingerprint = model_fingerprint
        self._cache = ByteLruCache(capacity_bytes, object_tensor_bytes, name=name)

    @property
    def enabled(self):
        return self._cache.enabled

    @property
    def capacity_bytes(self):
        return self._cache.capacity_bytes

    def key_for(self, geometry, scale, dtype="torch.float32"):
        return encoding_cache_key(geometry.fingerprint(), scale, self.model_fingerprint,
                                  dtype)

    def get(self, geometry, scale, dtype="torch.float32"):
        return self._cache.get(self.key_for(geometry, scale, dtype))

    def put(self, geometry, scale, encoded, dtype="torch.float32"):
        return self._cache.put(self.key_for(geometry, scale, dtype), encoded)

    def snapshot(self):
        return self._cache.snapshot()
