from torch_geometric.datasets import TUDataset
from .reproducibility import (
    set_np_and_torch, 
    permutation, 
    generate_loader, 
    load_seeds,
)
from torch_geometric.data import  Dataset
from torch_geometric.loader import DataLoader
from typing import Union

TU_DATASET = ["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD"]

def load_dataset(dataset:str) -> Dataset:
    _dataset = None
    if dataset in TU_DATASET: _dataset = TUDataset(root="/tmp/TUDataset", name=dataset, use_node_attr=True)
    return _dataset

def split_dataset(
        r:int, 
        expr_conf, 
        dataset:Dataset
) -> Union[DataLoader, DataLoader, DataLoader]:
    #generate reproducible permutation

    seeds = load_seeds(expr_conf["seeds"], expr_conf["runs"])
    set_np_and_torch(seeds[r-1])
    rnd_idx = permutation(dataset, seed=seeds[r-1])
    #shuffle
    dataset = dataset[list(rnd_idx)]
    #split by the ratio 8:1:1
    train_dataset = dataset[:int(0.8*len(dataset))]
    val_dataset = dataset[int(0.8*len(dataset)):int(0.9*len(dataset))]
    test_dataset = dataset[int(0.9*len(dataset)):]
    
    train_loader = generate_loader(train_dataset, expr_conf["batch_size"], shuffle=True, seed=seeds[r-1])
    val_loader = generate_loader(val_dataset, expr_conf["batch_size"], seed=seeds[r-1])
    test_loader = generate_loader(test_dataset, expr_conf["batch_size"], seed=seeds[r-1])

    return train_loader, val_loader, test_loader
