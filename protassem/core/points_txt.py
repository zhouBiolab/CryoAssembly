
import collections

import numpy as np

HEADER_LINES = 5

POINT_DTYPE = [("index", np.int32),
               ("point", np.float32, (3,)),
               ("vector", np.float32, (3,)),
               ("density", np.float32)]

COORD_FIELDS = "index x y z"
VECTOR_FIELDS = "vx vy vz density"

PointCloud = collections.namedtuple(
    "PointCloud",
    "points vectors densities indices sample origin header_lines data_lines")


def read_point_cloud(path):
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if len(lines) < HEADER_LINES:
        raise ValueError("%s: expected %d header lines, got %d"
                         % (path, HEADER_LINES, len(lines)))

    header_lines = lines[:HEADER_LINES]
    sample = _parse_float(path, 1, header_lines[0], "sample")
    origin = _parse_vector(path, 4, header_lines[3], "origin")

    data_lines = list(lines[HEADER_LINES:])
    while data_lines and not data_lines[-1].strip():
        data_lines.pop()
    for offset, line in enumerate(data_lines):
        if not line.strip():
            raise ValueError("%s:%d: blank line inside the data region"
                             % (path, HEADER_LINES + offset + 1))
    if len(data_lines) % 2:
        raise ValueError("%s:%d: data lines must come in pairs "
                         "(coordinate line + vector line), got %d line(s)"
                         % (path, HEADER_LINES + 1, len(data_lines)))

    points, vectors, densities, indices, pairs = [], [], [], [], []
    for offset in range(0, len(data_lines), 2):
        coord_lineno = HEADER_LINES + offset + 1
        vector_lineno = coord_lineno + 1
        coord_parts = _split(path, coord_lineno, data_lines[offset], COORD_FIELDS)
        vector_parts = _split(path, vector_lineno, data_lines[offset + 1], VECTOR_FIELDS)
        voxel = _to_float(path, coord_lineno, coord_parts[1:])
        indices.append(int(coord_parts[0]))
        points.append([voxel[i] * sample + origin[i] for i in range(3)])
        vectors.append(_to_float(path, vector_lineno, vector_parts[:3]))
        densities.append(_to_float(path, vector_lineno, vector_parts[3:])[0])
        pairs.append((data_lines[offset], data_lines[offset + 1]))

    return PointCloud(points=np.array(points, dtype=np.float64),
                      vectors=np.array(vectors, dtype=np.float64),
                      densities=np.array(densities, dtype=np.float64),
                      indices=np.array(indices, dtype=np.int64),
                      sample=sample,
                      origin=np.array(origin, dtype=np.float64),
                      header_lines=header_lines,
                      data_lines=pairs)


def line_mapping(cloud):
    return [(HEADER_LINES + 2 * i, HEADER_LINES + 2 * i + 1)
            for i in range(len(cloud.data_lines))]


def read_point_cloud_file(path):
    cloud = read_point_cloud(path)
    data = np.zeros(len(cloud.points), dtype=POINT_DTYPE)
    data["index"] = cloud.indices
    data["point"] = cloud.points
    data["vector"] = cloud.vectors
    data["density"] = cloud.densities
    return data


def write_filtered(path, cloud, keep_indices):
    with open(path, "w", encoding="utf-8") as handle:
        for line in cloud.header_lines:
            handle.write(line + "\n")
        for index in keep_indices:
            coord_line, vector_line = cloud.data_lines[index]
            handle.write(coord_line + "\n")
            handle.write(vector_line + "\n")


def write_point_cloud(path, points, vectors, densities=None, indices=None,
                      sample=1.0, origin=(0.0, 0.0, 0.0)):
    points = np.asarray(points, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    count = len(points)
    if densities is None:
        densities = np.zeros(count, dtype=np.float64)
    if indices is None:
        indices = np.arange(count, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.float64)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("%.6f\n" % sample)
        handle.write("0 0 0\n")
        handle.write("0.000000 0.000000 0.000000\n")
        handle.write("%.6f %.6f %.6f\n" % (origin[0], origin[1], origin[2]))
        handle.write("0.000000 0.000000 0.000000\n")
        for i in range(count):
            voxel = (points[i] - origin) / sample
            handle.write("%d %.6f %.6f %.6f\n"
                         % (int(indices[i]), voxel[0], voxel[1], voxel[2]))
            handle.write("%.6f %.6f %.6f %.6f\n"
                         % (vectors[i][0], vectors[i][1], vectors[i][2],
                            densities[i]))


def _split(path, lineno, line, expected):
    parts = line.split()
    if len(parts) != 4:
        raise ValueError("%s:%d: expected 4 fields <%s>, got %d"
                         % (path, lineno, expected, len(parts)))
    return parts


def _to_float(path, lineno, parts):
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError("%s:%d: %r is not a number" % (path, lineno, part)) from exc
    return values


def _parse_float(path, lineno, line, name):
    text = line.strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError("%s:%d: %s must be a number, got %r"
                         % (path, lineno, name, text)) from exc


def _parse_vector(path, lineno, line, name):
    parts = line.split()
    if len(parts) != 3:
        raise ValueError("%s:%d: %s must have 3 numbers, got %d"
                         % (path, lineno, name, len(parts)))
    return _to_float(path, lineno, parts)
