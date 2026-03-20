from torch import Tensor
import torch
from torch_geometric.data import Data
from typing import Optional


class MODEL(torch.nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, data:Data):
        pass
    
    def _pool(
            self, 
            pool:callable, 
            x:Tensor, 
            edge_index:Tensor, 
            batch:Tensor, 
    ):
        aux_loss = None

        if pool is not None:
            x, edge_index, batch, aux_loss = self._unpack_pool_output(
                pool(x=x, edge_index=edge_index, batch=batch)
            )

        return x, edge_index, batch, aux_loss

    def _unpack_pool_output(self, pool_out):
        out_len = len(pool_out)

        if out_len == 4:
            # Dense adapter path:
            # (x, edge_index, batch, aux_loss)
            x, edge_index, batch, aux_loss = pool_out
            return x, edge_index, batch, aux_loss

        if out_len == 6:
            # Sparse pooling path:
            # (x, edge_index, edge_attr, batch, perm, score)
            x, edge_index, _, batch, _, _ = pool_out
            return x, edge_index, batch, None

        if out_len == 5:
            # ASAPooling style path:
            # (x, edge_index, edge_attr, batch, perm)
            x, edge_index, _, batch, _ = pool_out
            return x, edge_index, batch, None

        raise ValueError(
            "Unsupported pool output format. "
            "Expected tuple length 4 (dense), 5 (ASAP style), or 6 (sparse), "
            f"but got {out_len}."
        )
    
    def _load_from_config(
            self, 
            config:Optional[dict] 
    ):

        #Trainning setting
        self.p_dropout = config["p_dropout"]
        self.hidden_features = config["hidden_features"]
        # the activation should not be learnable
        self.nonlinearity = config["nonlinearity"]

        #Model setting
        self.pre_gnn = [self.n_node_features]
        self.pre_gnn.extend(list(config["pre_gnn"]))

        self.post_gnn = list(config["post_gnn"])
        self.post_gnn.append(self.n_classes)
        self.CONV = config["conv_layer"]
