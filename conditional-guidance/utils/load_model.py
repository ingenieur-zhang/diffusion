import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch as th
from filelock import FileLock
from tqdm.auto import tqdm
def load_checkpoint(
    checkpoint_name: str,
    device: th.device,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> Dict[str, th.Tensor]:
    '''
    if checkpoint_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_PATHS.keys()}."
        )
    path = fetch_file_cached(
        MODEL_PATHS[checkpoint_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    '''
    path = './ckpt/' + checkpoint_name + '.pt'
    print('using ckeckpoint', checkpoint_name)
    return th.load(path, map_location=device)