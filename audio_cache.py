import glob
import hashlib
import os
import time

import numpy as np

CACHE_DIR = os.environ.get("AUDIO_CACHE_DIR", "cache/audio")
FEATURE_VERSION = 1

GROUPS = [
    ("mfcc_mean", 20, 1.0),
    ("mfcc_std", 20, 0.6),
    ("chroma", 12, 0.7),
    ("contrast", 7, 0.7),
    ("spectral", 10, 0.8),
    ("energy", 2, 0.5),
    ("rhythm", 4, 0.8),
]
DIM = sum(size for _, size, _ in GROUPS)

DONE = 0
DEEZER_MISS = 1
MISS = 2


def empty():
    return (
        np.array([], dtype="U1"),
        np.array([], dtype=np.int8),
        np.zeros((0, DIM), dtype=np.float16),
    )


def load(cache_dir=None):
    cache_dir = cache_dir or CACHE_DIR
    files = sorted(glob.glob(os.path.join(cache_dir, "shard_*.npz")))
    ids, status, feats = [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != FEATURE_VERSION or z["feats"].shape[1] != DIM:
                continue
            ids.append(z["ids"])
            status.append(z["status"])
            feats.append(z["feats"])
    if not ids:
        return empty()
    ids = np.concatenate(ids)
    status = np.concatenate(status)
    feats = np.concatenate(feats)
    _, first = np.unique(ids[::-1], return_index=True)
    keep = np.sort(len(ids) - 1 - first)
    return ids[keep], status[keep], feats[keep]


def write_shard(ids, status, feats, cache_dir=None):
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    stem = time.strftime("shard_%Y%m%d_%H%M%S", time.gmtime())
    path = os.path.join(cache_dir, stem + ".npz")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(cache_dir, f"{stem}_{counter}.npz")
        counter += 1
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            version=np.int32(FEATURE_VERSION),
            ids=np.asarray(ids, dtype=str),
            status=np.asarray(status, dtype=np.int8),
            feats=np.asarray(feats, dtype=np.float16),
        )
    os.replace(tmp, path)
    return path


def digest(ids, status, feats):
    h = hashlib.sha256()
    h.update(np.asarray(ids, dtype=str).tobytes())
    h.update(np.asarray(status, dtype=np.int8).tobytes())
    h.update(np.asarray(feats, dtype=np.float16).tobytes())
    return h.hexdigest()
