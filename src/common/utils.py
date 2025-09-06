import logging as log

from typing import Optional
from tqdm.auto import trange


def batch(data, n=128):
    data_len = len(data)
    for ndx in trange(0, data_len, n):
        batch = data[ndx:min(ndx + n, data_len)]
        if type(batch) is not list:
            batch = batch.tolist()
        # log.debug(f"Type: {type(batch)} Batch: {batch}")
        yield batch


try:
    import torch
    import subprocess
    import pandas as pd
    import sys
    from io import StringIO
    
    def cuda_available() -> bool:
        return torch.cuda.is_available()
    
    def get_least_utilized_gpu() -> int:
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode(sys.stdout.encoding)
        gpu_df = pd.read_csv(StringIO(gpu_stats),
                            names=['memory.used', 'memory.free'],
                            skiprows=1)
        log.debug('GPU usage:\n{}'.format(gpu_df))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        idx = gpu_df['memory.free'].astype(float).idxmax()
        log.info('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        return idx
    
    def get_device(device: Optional[str] = None, optimize_gpu_distribution: bool = True) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                if optimize_gpu_distribution:
                    device = f"cuda:{get_least_utilized_gpu()}"
                else:
                    device = "cuda"
            else:
                device = "cpu"
            log.warning(f"Using device: {device}")
        return torch.device(device)
except Exception as e:
    log.info(e)