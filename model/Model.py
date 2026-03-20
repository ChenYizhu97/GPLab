from torch import Tensor
import torch
from torch_geometric.data import Data
from typing import Optional

from layers.pool.contracts import validate_pool_output


class MODEL(torch.nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Flag to track if pool output validation has been performed
        self._pool_validated = False
    
    def forward(self, data:Data):
        pass
    
    def _pool(
            self, 
            pool:callable, 
            x:Tensor, 
            edge_index:Tensor, 
            batch:Tensor, 
    ):
        """
        Execute pooling and return standardized output.
        
        All pooling methods must return a PoolOutput dataclass instance.
        First batch validation is performed to catch contract violations early.
        
        Args:
            pool: The pooling layer (returns PoolOutput)
            x: Node features
            edge_index: Edge indices
            batch: Batch vector
            
        Returns:
            Tuple of (x, edge_index, batch, aux_loss)
        """
        if pool is None:
            return x, edge_index, batch, None

        # Execute pooling - expected to return PoolOutput
        pool_out = pool(x=x, edge_index=edge_index, batch=batch)

        # First-batch validation: catch contract violations early
        if not self._pool_validated:
            validate_pool_output(pool_out, pool.__class__.__name__)
            self._pool_validated = True

        # Extract fields from PoolOutput
        return pool_out.x, pool_out.edge_index, pool_out.batch, pool_out.aux_loss
    
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
