from typing import Optional
import torch 
from torch import Tensor
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.select import SelectOutput
from .MP.LSSMP import LSSMP

class SLSPooling(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            ratio: float = 0.5,
            nonlinearity: str = "elu",
            act: str = "tanh",
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.lssmp = LSSMP(in_channels, nonlinearity=nonlinearity, act=act, *args, **kwargs)
        self.connect = FilterEdges()
        self.ratio = ratio
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
        

        # lssmp returns score for each node
        scores = self.lssmp.forward(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        scores = scores.squeeze(-1)
        assert scores is not None
        
        node_index = topk(scores, ratio=self.ratio, batch=batch)

        select_output= SelectOutput(
            node_index = node_index,
            num_nodes = x.size(0),
            cluster_index = torch.arange(node_index.size(0), device=x.device),
            num_clusters = node_index.size(0),
            weight=scores[node_index]
        )

        

        x = x * scores.unsqueeze(-1)
        x = x[node_index]
        connect_out = self.connect.forward(select_output, edge_index, edge_attr, batch)

        return (x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, node_index, scores)
    