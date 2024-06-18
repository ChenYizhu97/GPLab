import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from typing import Any
from numpy import (
    int64,
    ndarray,
    dtype,
)

def permutation(
        dataset:Dataset, 
        seed:int=0
) -> ndarray[Any, dtype[int64]]:
    # set fixed seed for pemutating dataset
    rng = np.random.default_rng(seed)
    rnd_idx = rng.permutation(len(dataset))
    
    return rnd_idx

def set_np_and_torch(seed:int=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def seed_worker(worker_id:int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def generate_loader(
        dataset:Dataset, 
        batch_size:int, 
        shuffle:bool=False, 
        seed:int=0
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    data_loader = DataLoader(dataset=dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle,
                             worker_init_fn=seed_worker,
                             generator=g
                            )
    return data_loader

def load_seeds(
        seeds:str, 
        runs:int
) -> list[int]:
    with open(seeds, "a+") as f:
        f.seek(0)
        raw_seeds = f.readline()
        seeds = raw_seeds.split(' ')[:-1]
        seeds = [int(seed) for seed in seeds]
        n_seeds = len(seeds)
        # generate new seeds if there are not enough seeds in file.
        if runs > n_seeds:
            seeds_generated= list(np.random.random_integers(1, 100, runs - n_seeds))
            [print(seed, file=f, end=' ') for seed in seeds_generated]
            seeds.extend(seeds_generated)
        else:
            seeds = seeds[:runs]
    return seeds