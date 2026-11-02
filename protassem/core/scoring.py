'Density-based structure scoring.'

import os

import numpy as np
import mrcfile
from scipy.ndimage import distance_transform_edt, gaussian_filter
from numba import njit
from protassem.core.io import read_structure
from protassem.core.constants import atomic_number_dict, VDW_RADII
from protassem.core.numba_kernels import add_gaussian_to_grid, add_sphere_mask
from protassem.runtime.byte_cache import ByteLruCache, object_bytes
from protassem.runtime.config import DEFAULT_SCORE_CACHE_MB, SCORE_CACHE_ENV

SCORING_VERSION = 1
STRUCTURE_VERSION = 1


def read_mrc_full(filename):
    """Read MRC with axis reordering (handles mapc/mapr/maps)."""
    with mrcfile.open(filename, permissive=True) as mrc:
        data = mrc.data.copy()
        vx = float(mrc.voxel_size.x) or 1.0
        vy = float(mrc.voxel_size.y) or 1.0
        vz = float(mrc.voxel_size.z) or 1.0
        voxel_size = np.array([vx, vy, vz], dtype=np.float32)

        origin = np.array([
            float(mrc.header.origin.x),
            float(mrc.header.origin.y),
            float(mrc.header.origin.z)
        ], dtype=np.float32)

        nstart = np.array([
            int(mrc.header.nxstart),
            int(mrc.header.nystart),
            int(mrc.header.nzstart)
        ], dtype=np.float32)

        mapcrs = np.array([
            int(mrc.header.mapc),
            int(mrc.header.mapr),
            int(mrc.header.maps)
        ], dtype=int)

        sort = np.array([0, 1, 2], dtype=int)
        for i in range(3):
            sort[mapcrs[i] - 1] = i
        nstart = np.array([nstart[i] for i in sort], dtype=np.float32)
        data = np.transpose(data, axes=2 - sort[::-1])

        if np.all(origin == 0):
            origin = origin + nstart * voxel_size

    return data, voxel_size, origin, data.shape


@njit(fastmath=True)
def _pearson(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    sx, sy = 0.0, 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
    mx, my = sx / n, sy / n
    num, vx, vy = 0.0, 0.0, 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        num += dx * dy
        vx += dx * dx
        vy += dy * dy
    denom = np.sqrt(vx * vy)
    if denom == 0.0:
        return 0.0
    return num / denom


def _make_sim_map(origin, voxel_size, box_size, coords, elements, resolution):
    sigma_factor = 1.0 / (np.pi * np.sqrt(2.0))
    sigma_real = resolution * sigma_factor
    sigma_vox = np.array([sigma_real / voxel_size[i] for i in range(3)])

    grid = np.zeros(box_size, dtype=np.float32)

    for coord, elem in zip(coords, elements):
        w = atomic_number_dict.get(elem, 1.0)
        atom_coord = np.array(coord, dtype=np.float64)
        add_gaussian_to_grid(grid, atom_coord, w, origin, voxel_size,
                             sigma_vox, 5.0)

    norm = np.power(2 * np.pi, -1.5) * np.power(sigma_real, -3)
    grid *= norm
    return grid


def _make_phenix_mask(origin, voxel_size, box_size, coords, elements,
                      solvent_radius=1.1, mask_radius=2.0, resolution=None):
    nz, ny, nx = box_size
    vs = [float(voxel_size[i]) for i in range(3)]
    grid = [nx, ny, nz]

    boxes = []
    for coord, elem in zip(coords, elements):
        radius = VDW_RADII.get(elem, 1.70) + solvent_radius
        c = [(coord[i] - origin[i]) / vs[i] for i in range(3)]
        rv = [radius / vs[i] for i in range(3)]
        idx = [(max(0, int(np.floor(c[i] - rv[i]))),
                min(grid[i], int(np.ceil(c[i] + rv[i])) + 1)) for i in range(3)]
        if any(a >= b for a, b in idx):
            continue
        boxes.append((c, radius * radius, idx))

    if not boxes:
        return np.zeros(box_size, dtype=np.bool_)

    sigma = (resolution / 4.0) if resolution else 1.0
    sigma_vox = np.array([sigma / vs[i] for i in range(3)])
    pad = np.ceil(np.array([mask_radius / vs[i] for i in range(3)])
                  + 5.0 * sigma_vox + 2.0).astype(int)
    lo = np.maximum([min(b[2][i][0] for b in boxes) for i in range(3)] - pad, 0)
    hi = np.minimum([max(b[2][i][1] for b in boxes) for i in range(3)] + pad,
                    grid)

    sub = np.zeros((hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]), dtype=np.bool_)
    for c, radius_sq, idx in boxes:
        (i0, i1), (j0, j1), (k0, k1) = idx
        add_sphere_mask(sub, c[0] - lo[0], c[1] - lo[1], c[2] - lo[2],
                        vs[0], vs[1], vs[2], radius_sq,
                        i0 - lo[0], i1 - lo[0], j0 - lo[1], j1 - lo[1],
                        k0 - lo[2], k1 - lo[2])

    avg_vs = float(np.mean(voxel_size))
    dist = distance_transform_edt(~sub) * avg_vs
    expanded = (dist <= mask_radius) | sub
    soft = gaussian_filter(expanded.astype(np.float32), sigma=sigma_vox,
                           mode="constant", cval=0.0)
    if soft.max() > 0:
        soft /= soft.max()
    out = np.zeros(box_size, dtype=np.bool_)
    out[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = soft > 0.5
    return out


class DensityMapContext:

    __slots__ = ("path", "data", "voxel_size", "origin", "shape", "contour",
                 "fingerprint")

    def __init__(self, path, data, voxel_size, origin, contour, fingerprint):
        self.path = path
        self.data = data
        self.voxel_size = voxel_size
        self.origin = origin
        self.shape = data.shape
        self.contour = contour
        self.fingerprint = fingerprint


def _entry_bytes(value):
    if isinstance(value, DensityMapContext):
        return (object_bytes(value.data) + object_bytes(value.voxel_size)
                + object_bytes(value.origin) + 512)
    coords, elements = value
    return object_bytes(coords) + object_bytes(elements) + 256


class _ScoreCache:

    KIND_PREFIX = {"density": "d:", "structure": "s:"}

    def __init__(self, capacity_mb):
        self.capacity_mb = int(capacity_mb)
        capacity_bytes = max(0, self.capacity_mb) * 1024 * 1024
        self.cache = ByteLruCache(capacity_bytes, _entry_bytes, "score")
        self.counters = {kind: {"hits": 0, "misses": 0, "puts": 0, "rejects": 0}
                         for kind in self.KIND_PREFIX}

    @property
    def enabled(self):
        return self.cache.enabled

    def get(self, kind, key):
        value = self.cache.get(self.KIND_PREFIX[kind] + key)
        counter = self.counters[kind]
        if value is None:
            counter["misses"] += 1
        else:
            counter["hits"] += 1
        return value

    def put(self, kind, key, value):
        stored = self.cache.put(self.KIND_PREFIX[kind] + key, value)
        self.counters[kind]["puts" if stored else "rejects"] += 1
        return stored

    def clear(self):
        self.cache.clear()
        for counter in self.counters.values():
            for key in counter:
                counter[key] = 0

    def snapshot(self):
        shared = self.cache.snapshot()
        result = {"score_cache_mb": self.capacity_mb,
                  "capacity_bytes": shared["capacity_bytes"],
                  "entries": shared["entries"], "bytes": shared["bytes"],
                  "peak_bytes": shared["peak_bytes"],
                  "evictions": shared["evictions"]}
        for kind, counter in self.counters.items():
            total = counter["hits"] + counter["misses"]
            result[kind] = dict(counter)
            result[kind]["hit_rate"] = round(float(counter["hits"]) / total, 6) \
                if total else None
            result[kind]["capacity_bytes"] = shared["capacity_bytes"]
        return result


def _env_capacity_mb():
    raw = os.environ.get(SCORE_CACHE_ENV)
    if raw in (None, ""):
        return DEFAULT_SCORE_CACHE_MB
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return DEFAULT_SCORE_CACHE_MB


_CACHE = _ScoreCache(_env_capacity_mb())


def configure_score_cache(capacity_mb):
    global _CACHE
    capacity_mb = max(0, int(capacity_mb))
    if capacity_mb != _CACHE.capacity_mb:
        _CACHE = _ScoreCache(capacity_mb)
    else:
        _CACHE.clear()
    return _CACHE.snapshot()


def apply_score_cache(capacity_mb):
    os.environ[SCORE_CACHE_ENV] = str(int(max(0, int(capacity_mb))))
    return configure_score_cache(capacity_mb)


def score_cache_snapshot():
    return _CACHE.snapshot()


def invalidate_density():
    entries = _CACHE.snapshot()["entries"]
    _CACHE.clear()
    return entries


def density_fingerprint(density_mrc, contour, density_version=None):
    stat = os.stat(str(density_mrc))
    return "|".join([str(SCORING_VERSION), os.path.abspath(str(density_mrc)),
                     str(stat.st_size), str(stat.st_mtime_ns),
                     repr(float(contour)),
                     "-" if density_version is None else str(density_version)])


def load_density_context(density_mrc, contour, density_version=None):
    data, voxel_size, origin, _dims = read_mrc_full(density_mrc)
    data = np.where(data > contour, data, -1.0)
    return DensityMapContext(
        path=os.path.abspath(str(density_mrc)), data=data, voxel_size=voxel_size,
        origin=origin, contour=float(contour),
        fingerprint=density_fingerprint(density_mrc, contour, density_version))


def density_context(density_mrc, contour, density_version=None):
    key = density_fingerprint(density_mrc, contour, density_version)
    cached = _CACHE.get("density", key)
    if cached is not None:
        return cached
    context = load_density_context(density_mrc, contour, density_version)
    _CACHE.put("density", key, context)
    return context


def structure_coords(structure_file):
    path = os.path.abspath(str(structure_file))
    try:
        stat = os.stat(path)
        key = "|".join([str(STRUCTURE_VERSION), path, str(stat.st_size),
                        str(stat.st_mtime_ns)])
    except OSError:
        key = None
    cached = _CACHE.get("structure", key) if key is not None else None
    if cached is not None:
        return cached
    coords, elements = read_structure(str(structure_file))
    value = (list(coords), list(elements))
    if key is not None:
        _CACHE.put("structure", key, value)
    return value


def score_coords(context, coords, elements, resolution):
    exp_map = context.data
    dims = context.shape
    origin, voxel_size = context.origin, context.voxel_size
    elem_list = list(elements)

    sim_map = _make_sim_map(origin, voxel_size, dims, coords, elem_list, resolution)
    mask = _make_phenix_mask(origin, voxel_size, dims, coords, elem_list,
                             resolution=resolution)

    sel = mask
    if np.count_nonzero(sel) == 0:
        return 0.0
    exp_vals = exp_map[sel].astype(np.float64)
    sim_vals = sim_map[sel].astype(np.float64)
    return float(_pearson(exp_vals, sim_vals))


def calculate_cc_mask(density_mrc, structure_file, resolution, contour,
                      density_version=None):
    'Compute masked structure-to-map correlation.'
    context = density_context(density_mrc, contour, density_version)
    coords, elements = structure_coords(structure_file)
    return score_coords(context, coords, elements, resolution)
