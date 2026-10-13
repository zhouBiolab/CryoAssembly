'Runtime settings for inference and CPU workers.'

import json
import os
from dataclasses import dataclass, fields

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

DEFAULT_GEOMETRY_CACHE_MB = 512

DEFAULT_ALLOW_TF32 = None

INFERENCE_MODES = ("joint", "split")
DEFAULT_INFERENCE_MODE = "joint"

DEFAULT_ENCODING_CACHE_MB = 256

DEFAULT_HYPOTHESIS_CHUNK = 64

DEFAULT_TAIL_PIPELINE = False


def effective_allow_tf32(inference_mode, allow_tf32=None):
    if inference_mode not in INFERENCE_MODES:
        raise ValueError('Unknown inference_mode: %r (supported: %s)'
                         % (inference_mode, "/".join(INFERENCE_MODES)))
    if allow_tf32 is None:
        return inference_mode == "joint"
    if inference_mode == "split" and allow_tf32:
        raise ValueError('inference_mode=split requires allow_tf32=false: TF32 changes single-cloud encoding (observed pose difference 0.127)')
    return bool(allow_tf32)


#
DEFAULT_SCORE_CACHE_MB = 1024
SCORE_CACHE_ENV = "PROTASSEM_SCORE_CACHE_MB"

DEFAULT_TM_CACHE = "auto"


def apply_seed(seed):
    import random
    import numpy as np
    random.seed(int(seed))
    np.random.seed(int(seed))
    return {"seed": int(seed)}


def apply_tf32_policy(allow_tf32):
    import torch
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}


@dataclass
class RuntimeConfig:
    'Runtime options for inference and CPU execution.'

    blas_threads: int = 1
    seed: int = 7351
    pool_start_method: str = None
    geometry_cache_mb: int = DEFAULT_GEOMETRY_CACHE_MB
    inference_mode: str = DEFAULT_INFERENCE_MODE
    allow_tf32: bool = DEFAULT_ALLOW_TF32
    encoding_cache_mb: int = DEFAULT_ENCODING_CACHE_MB
    hypothesis_chunk: int = DEFAULT_HYPOTHESIS_CHUNK
    tail_pipeline: bool = DEFAULT_TAIL_PIPELINE
    score_cache_mb: int = DEFAULT_SCORE_CACHE_MB
    tm_cache: str = DEFAULT_TM_CACHE

    def __post_init__(self):
        effective_allow_tf32(self.inference_mode, self.allow_tf32)
        if int(self.hypothesis_chunk) < 0:
            raise ValueError('hypothesis_chunk must be nonnegative: %r' % (self.hypothesis_chunk,))
        if int(self.score_cache_mb) < 0:
            raise ValueError('score_cache_mb must be nonnegative: %r' % (self.score_cache_mb,))
        if not isinstance(self.tm_cache, str) or not self.tm_cache.strip():
            raise ValueError('tm_cache must be a nonempty string (auto/off/path): %r'
                             % (self.tm_cache,))

    def tf32(self):
        return effective_allow_tf32(self.inference_mode, self.allow_tf32)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError("%s: unknown runtime config key(s): %s"
                             % (path, ", ".join(unknown)))
        return cls(**data)

    def apply_thread_env(self):
        value = str(max(1, int(self.blas_threads)))
        for name in THREAD_ENV_VARS:
            os.environ[name] = value
        return value

    def describe_effective_threads(self):
        import threadpoolctl
        import torch
        return {
            "env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
            "blas": [{"internal_api": info.get("internal_api"),
                      "num_threads": info.get("num_threads")}
                     for info in threadpoolctl.threadpool_info()],
            "torch_intra": torch.get_num_threads(),
        }
