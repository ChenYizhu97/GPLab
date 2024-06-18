import torch 
from torch import Tensor
from typing import Optional
from torch_geometric.utils import to_dense_adj, to_dense_batch
from utils.data import to_sparse_batch
from .MP.LSSMP import LSSMP
from ..PoolAdapter import dense_connect

class DLSPooling(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            k:int,
            nonlinearity: str = "elu",
            act: str = "tanh",
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.k = k
        self.lssmp = LSSMP(
            in_channels=in_channels,
            nonlinearity=nonlinearity,
            act=act,
            k=k,
            *args,
            *kwargs
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lssmp.reset_parameters()


    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor]=None,
            batch: Tensor=None,
    ):
        # The situation batch size is 1?
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        

        # lssmp returns assignment matrix
        s = self.lssmp.forward(x, edge_index, edge_attr)


        # transform to dense matrix to generate coarsened graph by matrix multiplication. 
        x, mask = to_dense_batch(x, batch)
        s, _ = to_dense_batch(s, batch)
        adj = to_dense_adj(edge_index, batch, edge_attr)
        
        out, out_adj = dense_connect(x, adj, s, mask)
        
        # transform back to sparse mode
        x, edge_index, batch = to_sparse_batch(out, out_adj)
        
        return (x, edge_index, None, batch, None, None)

        