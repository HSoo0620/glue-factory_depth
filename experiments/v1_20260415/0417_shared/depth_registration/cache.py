"""Master 캐시: (이미지, 파라미터) fingerprint 기반 npz."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _img_hash(zmap: np.ndarray) -> str:
    return hashlib.sha1(zmap.tobytes()).hexdigest()[:8]


def _params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def master_cache_key(zmap: np.ndarray, descriptor: str, params: dict) -> str:
    return f"master_{_img_hash(zmap)}_{_params_hash(params)}_{descriptor}"


def _path(key: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{key}.npz"


def load_master_cache(key: str, cache_dir: Path) -> Optional[dict]:
    p = _path(key, cache_dir)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    except Exception:
        try:
            p.unlink()
        except OSError:
            pass
        return None


def save_master_cache(key: str, payload: dict, cache_dir: Path) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    p = _path(key, cache_dir)
    np.savez_compressed(p, **payload)
