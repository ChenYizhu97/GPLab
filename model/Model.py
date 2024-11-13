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
            pool_out = pool(x=x, edge_index=edge_index, batch=batch)
            if len(pool_out) == 4:
                # dense pool
                x, edge_index, batch, aux_loss = pool_out
            elif len(pool_out) == 6:
                # sparse pool
                x, edge_index, _, batch, perm, score = pool_out
            else:
                # asapool
                x, edge_index, _, batch, perm = pool_out

        return x, edge_index, batch, aux_loss
    
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
        self.pre_gnn.extend(config["pre_gnn"])

        self.post_gnn = config["post_gnn"]
        self.post_gnn.append(self.n_classes)
        self.CONV = config["conv_layer"]