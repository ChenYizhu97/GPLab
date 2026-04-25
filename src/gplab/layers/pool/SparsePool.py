from typing import Optional, Union
import torch 
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.pool.select import SelectOutput
from ..functional import topk

from .contracts import PoolOutput


class SparsePooling(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            ratio: float = 0.5,
            act: str = "tanh",
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.select = SelectSparse(in_channels, ratio=ratio, act=act)
        self.connect = FilterEdges()
        self.reset_parameters()

    def reset_parameters(self):
        self.select.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor]=None,
            batch: Tensor=None,
    ) -> PoolOutput:
        """
        Sparse pooling using linear scoring + topk selection.
        
        Args:
            x: Node features, shape [N, F]
            edge_index: Edge indices, shape [2, E]
            edge_attr: Optional edge features
            batch: Batch vector, shape [N]
            
        Returns:
            PoolOutput with pooled graph structure
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        select_output = self.select.forward(x, batch)
        scores = select_output.weight
        perm = select_output.node_index
        assert scores is not None

        x = x[perm] * scores.unsqueeze(-1)
        connect_out = self.connect.forward(select_output, edge_index, edge_attr, batch)

        return PoolOutput(
            x=x,
            edge_index=connect_out.edge_index,
            batch=connect_out.batch,
            edge_attr=connect_out.edge_attr,
            perm=perm,
            score=scores,
        )


class SelectSparse(torch.nn.Module):
    def __init__(
            self, 
            in_channels: int,
            ratio: Union[float, int] = 0.5,
            act: Union[str, callable] = "tanh",
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.w_linear = Linear(in_channels, 1)
        self.act = activation_resolver(act)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_linear.reset_parameters()

    def forward(
            self, 
            x: Tensor,
            batch: Tensor,
    ) -> SelectOutput:
        scores = self.w_linear.forward(x)
        scores = scores.squeeze(-1)
        
        # NOTE: Original implementation had act saved but not used on scores.
        # For consistency with SAGPooling's SelectSAG, we apply activation here.
        scores = self.act(scores)

        node_index = topk(scores, ratio=self.ratio, batch=batch)

        return SelectOutput(
            node_index=node_index,
            num_nodes=x.size(0),
            cluster_index=torch.arange(node_index.size(0), device=x.device),
            num_clusters=node_index.size(0),
            weight=scores[node_index]
        )
