import torch
import numpy as np
import random
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

def set_np_and_torch(seed:int=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id:int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_loader(
        dataset:Dataset, 
        batch_size:int, 
        shuffle:bool=False, 
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(0)
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
        #strip space
        raw_seeds = f.readline().strip()
        seeds = raw_seeds.split(' ')

        # situation the seed file is empty
        if seeds[0] == '':
            seeds = []

        n_seeds = len(seeds)
        if n_seeds > 0 :
            seeds = [int(seed) for seed in seeds]
        
        # generate new seeds if there are not enough seeds in file.
        if runs > n_seeds:
            seeds_generated= list(np.random.random_integers(1, 100, runs - n_seeds))
            [print(seed, file=f, end=' ') for seed in seeds_generated]
            seeds.extend(seeds_generated)
        else:
            seeds = seeds[:runs]
    return seeds