from typing import Optional, Union
import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, MeanAggregation, MaxAggregation, SoftmaxAggregation
from torch_geometric.nn.resolver import activation_resolver

eps=1e-7

class LSSMP(MessagePassing):
    """Local set score MP that calculates score of local set.

    Args:
        x: Node features.
        edge_index: Sparse adjacency matrix.
        edge_attr: Edge attributes (optional).
    Returns:
    
    """
    def __init__(
            self, 
            in_channels: int,
            in_channels_edge: Optional[int] = None,
            nonlinearity: str = "elu",
            act: Optional[str] = "tanh",
            k: Optional[int] = None,
            *args, 
            **kwargs
        ):
        super().__init__(aggr=None, *args, **kwargs)

        hidden_channels = 1 
        #if k is None else k

        self.w_central = Linear(in_channels, hidden_channels)
        self.w_diff = Linear(in_channels, hidden_channels)

        #nonlinearity
        self.nonlinearity = activation_resolver(nonlinearity)
        self.mean_agg = MeanAggregation()
        self.max_agg = MaxAggregation()
        self.softmax_agg = SoftmaxAggregation(learn=True)

        if in_channels_edge is not None:
            self.w_edge = Linear(in_channels_edge, hidden_channels)
            self.w_localset = Linear(3*hidden_channels, 1) if k is None else Linear(3*hidden_channels, k)
        else:
            self.w_edge = None
            self.w_localset = Linear(2*hidden_channels, 1) if k is None else Linear(2*hidden_channels, k)

        #tanh?
        self.act = activation_resolver(act) if act is not None else None
        
        self.reset_parameters()

    
    def forward(
            self, 
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor],
            *args, 
            **kwargs
        ):
        feat_x = self.w_central(x)
        feat_x = self.nonlinearity(feat_x)

        feat_diff, feat_edge = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x, *args, **kwargs)
        
        
        # concatenate features
        if feat_edge is None:
            feat_localset = torch.concat((feat_x, feat_diff), -1)
        else:
            feat_localset = torch.concat((feat_x, feat_diff, feat_edge), -1)

        scores = self.w_localset(feat_localset)
        if self.act is not None: scores = self.act(scores)

        return scores
        


    def message(self, x_j: Tensor, x_i: Tensor, edge_attr_j:Optional[Tensor]) -> Tensor:
        diff = abs(x_j-x_i)
        #diff_norm = diff/diff.norm(dim=-1, p=2).unsqueeze(-1)
        #diff_norm = torch.norm(diff, p=2, dim=1) + eps
        # print(diff_norm.size())
        #diff = torch.div(diff, diff_norm.unsqueeze(-1))
        #print(diff)
        return diff, edge_attr_j
    
    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:

        diff, edge = inputs
        #diff = self.mean_agg.forward(diff, index=index, ptr=ptr, dim_size=dim_size)
        #diff = self.max_agg.forward(diff, index=index, ptr=ptr, dim_size=dim_size)
        diff = self.max_agg.forward(diff, index=index, ptr=ptr, dim_size=dim_size)
        diff = self.w_diff(diff)
        diff = self.nonlinearity(diff)

        if edge is not None:
            edge = self.mean_agg.forward(edge, index=index, ptr=ptr, dim_size=dim_size)

            edge = self.w_edge(edge)
            edge = self.nonlinearity(edge)
        else:
            edge = None

        return diff, edge
    

    def reset_parameters(self):
        self.w_central.reset_parameters()
        self.w_diff.reset_parameters()
        self.w_localset.reset_parameters()
        self.softmax_agg.reset_parameters()
        if self.w_edge: self.w_edge.reset_parameters()
        