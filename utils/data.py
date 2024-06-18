import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse
from typing import Optional

def to_sparse_batch(
        x:Tensor,
        adj:Tensor,
        mask:Optional[Tensor] = None
) -> tuple[Tensor, Tensor, Tensor]:
    edge_index, _ = dense_to_sparse(adj, mask)
    batch_num = x.size(0)
    node_num = x.size(1)
    x = x.reshape((batch_num*node_num, -1))
    batch = torch.arange(batch_num, device=x.device)
    batch = batch.repeat_interleave(node_num).to(batch.device)

    return x, edge_index, batch
