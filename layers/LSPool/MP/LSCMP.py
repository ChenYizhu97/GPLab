from typing import Optional
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import Tanh
from torch_scatter.composite import scatter_softmax
from torch_geometric.nn import MessagePassing, Aggregation, SoftmaxAggregation, SumAggregation
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.inits import uniform

class LSCAggregation(Aggregation):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #self.t = torch.nn.Parameter(torch.empty(1, 1))
        #self.reset_parameters()

    def forward(
        self, 
        x: Tensor,
        attn_score:Tensor, 
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None, 
        dim_size: Optional[int] = None,
        dim: int = -2
    ) -> Tensor:

        attn = scatter_softmax(attn_score, index=index, dim=dim, dim_size=dim_size)
        x = x * attn
        
        return self.reduce(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, reduce="sum")

    def reset_parameters(self):
        #uniform(1, self.t)
        pass

class LSCMP(MessagePassing):
    """Local set collapse MP that collapses local sets into pooled nodes.

    Args:
        scores: Scores produced by LSSMP.
        x: Node features.
        edge_index: Sparse adjancency matrix.
    """
    def __init__(
        self,
        hidden_channels,
        *args,
        **kwargs
    ):
        super().__init__(aggr=None, *args, **kwargs)
        
        #self.w1_attn = Linear(2*hidden_channels, 1, bias=False)
        #self.act = Tanh()
        #self.lsc_agg = LSCAggregation()
        self.lsc_agg = LSCAggregation()
        #self.w2_attn = torch.nn.Parameter(torch.empty(1, 1))

        self.reset_parameters()
    
    def forward(
        self,
        x,
        score,
        edge_index,
        *args, 
        **kwargs
    ):  
        # add self loop
        edge_index, _ = add_self_loops(edge_index)
        x = self.propagate(edge_index, x=x, score=score, *args, **kwargs)
        return x

    """
    def message(self, x_j: Tensor, x_i:Tensor) -> Tensor:
        attn_score = self.w1_attn(torch.concat((x_j, x_i), -1))
        #attn_score = self.act(attn_score)
        #attn_score = attn_score * self.w2_attn
        return x_j, attn_score
    
    """
    def message(self, x_j: Tensor, score_j:Tensor) -> Tensor:
        return x_j, score_j
    
    

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:
        x, score = inputs
        #print(x.size())
        x = self.lsc_agg.forward(x, score, index=index, ptr=ptr, dim_size=dim_size)
        return x

    
    def reset_parameters(self):
        #self.w1_attn.reset_parameters()
        #uniform(1, self.w2_attn)
        self.lsc_agg.reset_parameters()

     