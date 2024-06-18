from typing import Optional, Union
import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.nn.pool import TopKPooling, SAGPooling
from .SparsePool import SparsePooling
from .PoolAdapter import PoolAdapter


def conv_resolver(layer:str) -> Optional[torch.nn.Module]:
    # add more convolution layer resolvers if use other convolution layers
    if layer == "GCN": return GCNConv
    if layer == "GraphConv": return GraphConv
    return None


def pool_resolver(pool:str, in_channels:int, ratio:float=0.5, avg_node_num:Optional[float]=None, nonlinearity:Union[str, callable]="elu") -> Optional[torch.nn.Module]:
    # if the pooling method is a dense pooling, this func will return the learnable part of the pooling method.
    
    # no pool
    pool_layer = None

    if avg_node_num is not None: k = int(avg_node_num*ratio)

    if pool == "topkpool": pool_layer = TopKPooling(in_channels, ratio=ratio)
    if pool == "sagpool": pool_layer = SAGPooling(in_channels, ratio=ratio)
    if pool == "sparsepool": pool_layer = SparsePooling(in_channels, ratio=ratio)
    #for diffpool, mincutpool, densepool, the learning part  is a linear layer.
    if pool in ["diffpool", "mincutpool", "densepool"]: pool_layer = PoolAdapter(Linear(in_channels, k), pool)
   
    return pool_layer


