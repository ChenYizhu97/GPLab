from typing import Optional, Union
import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, GCNConv, DenseGCNConv, GINConv
from torch_geometric.nn.pool import TopKPooling, ASAPooling
from .pool import LCPooling, SAGPooling, PoolAdapter


def conv_resolver(layer:str) -> Optional[torch.nn.Module]:
    # add more convolution layer resolvers if use other convolution layers
    if layer == "GCN": return GCNConv
    if layer == "GraphConv": return GraphConv
    if layer == "GIN": return lambda in_channel, out_channel: GINConv(nn=Linear(in_channel, out_channel))
    return None


def pool_resolver(pool:str, in_channels:int, ratio:float=0.5, avg_node_num:Optional[float]=None, nonlinearity:Union[str, callable]="relu") -> Optional[torch.nn.Module]:
    # if the pooling method is diffpool, mincutpool, dlspool and densepool, this func will return the learnable part of the pooling method.
    
    # no pool
    pool_layer = None

    if avg_node_num is not None: k = int(avg_node_num*ratio)

    if pool == "topkpool": pool_layer = TopKPooling(in_channels, ratio=ratio)
    if pool == "lcpool": pool_layer = LCPooling(in_channels, ratio=ratio, nonlinearity=nonlinearity)
    if pool == "sagpool": pool_layer = SAGPooling(in_channels, ratio=ratio)
    if pool == "asapool": pool_layer = ASAPooling(in_channels, ratio=ratio)
    #for diffpool, mincutpool, densepool, the learning part  is a linear layer.
    if pool == "mincutpool": pool_layer = PoolAdapter(Linear(in_channels, k), pool)
    if pool == "diffpool": pool_layer = PoolAdapter(DenseGCNConv(in_channels, k), pool, nonlinearity=nonlinearity)
     
    return pool_layer


