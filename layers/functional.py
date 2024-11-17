from torch_geometric.nn import global_add_pool, global_max_pool
import torch

def readout(x, batch=None, size=None):
    x = torch.concat((global_add_pool(x=x, batch=batch, size=size), global_max_pool(x=x, batch=batch, size=size)), dim=-1)
    return x