import torch
import numpy as np
import random
from pathlib import Path
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
        seed:int=0,
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
    if runs <= 0:
        return []

    seed_path = Path(seeds)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.touch(exist_ok=True)

    raw_tokens = seed_path.read_text().split()
    seed_values = [int(token) for token in raw_tokens]

    if runs > len(seed_values):
        # Keep generation deterministic so repeated bootstrap produces the same sequence.
        # Generate a full deterministic sequence and append only the missing suffix.
        rng = np.random.default_rng(0)
        generated_full = rng.integers(1, 10**9, size=runs, endpoint=False).tolist()
        generated = generated_full[len(seed_values):]
        seed_values.extend(generated)
        seed_path.write_text(" ".join(str(seed) for seed in seed_values) + "\n")

    return seed_values[:runs]
