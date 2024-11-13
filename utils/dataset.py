from torch_geometric.datasets import TUDataset
import numpy as np
from torch_geometric.data import  Dataset
from torch_geometric.loader import DataLoader
from typing import Union

TU_DATASET = ["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD", "NCI1", "COX2"]

def load_dataset(dataset:str) -> Dataset:
    _dataset = None
    if dataset in TU_DATASET: _dataset = TUDataset(root="/tmp/TUDataset", name=dataset, use_node_attr=True)
    return _dataset

def split_dataset(
        dataset:Dataset
) -> Union[DataLoader, DataLoader, DataLoader]:
    #generate reproducible permutation

    rnd_idx = np.random.permutation(len(dataset))
    #shuffle
    dataset = dataset[list(rnd_idx)]
    #split by the ratio 8:1:1
    train_dataset = dataset[:int(0.8*len(dataset))]
    val_dataset = dataset[int(0.8*len(dataset)):int(0.9*len(dataset))]
    test_dataset = dataset[int(0.9*len(dataset)):]
    
    return train_dataset, val_dataset, test_dataset
