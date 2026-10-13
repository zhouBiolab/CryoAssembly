
import collections
import os
import shutil

import mrcfile
import numpy as np

NormalizedMap = collections.namedtuple(
    "NormalizedMap", "path origin previous_origin nstart voxel_size sample_anchor")


def sample_anchor(origin, nstart, voxel_size):
    return (np.asarray(origin, dtype=np.float64)
            + np.asarray(nstart, dtype=np.float64)
            * np.asarray(voxel_size, dtype=np.float64))


def display_origin(origin, nstart, voxel_size):
    origin = np.asarray(origin, dtype=np.float64)
    if np.any(np.abs(origin) > 1e-6):
        return origin
    return (np.asarray(nstart, dtype=np.float64)
            * np.asarray(voxel_size, dtype=np.float64))


def normalize_density_map(source, dest):
    if not os.path.isfile(source):
        raise FileNotFoundError("density map not found: %s" % source)

    with mrcfile.open(source, permissive=True) as mrc:
        origin = np.array([mrc.header.origin.x, mrc.header.origin.y,
                           mrc.header.origin.z], dtype=np.float64)
        nstart = np.array([mrc.header.nxstart, mrc.header.nystart,
                           mrc.header.nzstart], dtype=np.float64)
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y,
                               mrc.voxel_size.z], dtype=np.float64)

    previous_anchor = sample_anchor(origin, nstart, voxel_size)

    if not nstart.any():
        return NormalizedMap(source, origin, origin, nstart, voxel_size,
                             previous_anchor)

    fixed = display_origin(origin, nstart, voxel_size)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    shutil.copyfile(source, dest)
    with mrcfile.open(dest, mode="r+", permissive=True) as mrc:
        mrc.header.origin.x = float(fixed[0])
        mrc.header.origin.y = float(fixed[1])
        mrc.header.origin.z = float(fixed[2])
        mrc.header.nxstart = 0
        mrc.header.nystart = 0
        mrc.header.nzstart = 0
        mrc.flush()
    return NormalizedMap(dest, fixed, origin, nstart, voxel_size,
                         previous_anchor)
