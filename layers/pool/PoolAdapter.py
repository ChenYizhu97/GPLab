from typing import Optional
import torch
from torch import Tensor
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool
from utils.data import to_sparse_batch
from functional import dense_connect
from typing_extensions import Union
from torch_geometric.nn.resolver import activation_resolver

class PoolAdapter(torch.nn.Module):
    def __init__(
            self, 
            pool:Optional[torch.nn.Module],
            pool_method: str,
            nonlinearity:Union[str, callable]="relu",
            *args, 
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pool = pool
        self.pool_method = pool_method
        self.aux_loss = None
        self.nonlinearity = activation_resolver(nonlinearity)
        self.reset_parameters()
    
    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Tensor
    ):
        x, mask, adj = self._pre_pool(x, edge_index, batch)
        # pool
        if self.pool_method == "diffpool":
            s = self.pool(x, adj, mask)
        if self.pool_method in ["mincutpool", "densepool"]:
            s = self.pool(x)

        # connect
        if self.pool_method == "mincutpool":
            x, adj, l1, l2 = dense_mincut_pool(x, adj, s, mask)
            self.aux_loss = 0.5*l1 + l2
        if self.pool_method == "diffpool":
            x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
            self.aux_loss = 0.1*link_loss + 0.1*ent_loss
        if self.pool_method == "densepool":
            x, adj = dense_connect(x, adj, s, mask)

        x, edge_index, batch = self._post_pool(x, adj)
        return x, edge_index, batch, self.aux_loss

    
    def _pre_pool(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Tensor
    ):
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch)

        return x, mask, adj
        
    def _post_pool(
            self,
            x: Tensor,
            adj: Tensor
    ):
        x, edge_index, batch = to_sparse_batch(x, adj)
        return x, edge_index, batch

    def reset_parameters(self):
        self.pool.reset_parameters()