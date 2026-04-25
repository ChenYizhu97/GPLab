"""
Pooling output contract for Graph Pooling Lab (GPLab).

This module defines the unified output format for all pooling methods,
replacing the fragile tuple-length-based dispatch.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor


@dataclass
class PoolOutput:
    """
    Unified output contract for all graph pooling methods.

    Required fields:
        x: Node features after pooling, shape [N_pooled, F]
        edge_index: Edge indices after pooling, shape [2, E_pooled]
        batch: Batch vector assigning nodes to graphs, shape [N_pooled]

    Optional fields:
        edge_attr: Edge features, shape [E_pooled, F_edge] or None
        edge_weight: Edge weights, shape [E_pooled] or None
        perm: Indices of selected nodes in original graph, shape [N_pooled] or None
        score: Selection scores/weights for pooled nodes, shape [N_pooled] or None
        aux_loss: Auxiliary loss term (e.g., link loss, entropy loss), scalar Tensor or None

    Example:
        >>> pool_out = PoolOutput(
        ...     x=x_pooled,
        ...     edge_index=edge_index_pooled,
        ...     batch=batch_pooled,
        ...     aux_loss=link_loss + ent_loss,
        ... )
    """
    x: Tensor
    edge_index: Tensor
    batch: Tensor
    edge_attr: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None
    perm: Optional[Tensor] = None
    score: Optional[Tensor] = None
    aux_loss: Optional[Tensor] = None


def _check_tensor_field(value, field_name: str, pool_name: str, device: torch.device):
    """Validate that a field is a Tensor on the correct device."""
    if not isinstance(value, Tensor):
        raise TypeError(
            f"Pooling '{pool_name}': {field_name} must be a Tensor or None, "
            f"got {type(value).__name__}"
        )
    if value.device != device:
        raise RuntimeError(
            f"Pooling '{pool_name}': Device mismatch for {field_name}."
        )


def validate_pool_output(pool_out, pool_name: str) -> None:
    """
    Validate that a pooling output conforms to the PoolOutput contract.

    This function performs strong validation on the first batch and should
    raise informative errors for contract violations.

    Args:
        pool_out: The output to validate, expected to be a PoolOutput instance
        pool_name: Name of the pooling method (for error messages)

    Raises:
        TypeError: If pool_out is not a PoolOutput instance
        ValueError: If required fields are missing or have wrong types
        RuntimeError: If tensor shapes/devices are inconsistent
    """
    # Check 1: Must be PoolOutput instance
    if not isinstance(pool_out, PoolOutput):
        raise TypeError(
            f"Pooling method '{pool_name}' must return a PoolOutput instance, "
            f"got {type(pool_out).__name__}. "
            f"Please ensure your pooling layer returns PoolOutput(x=x, edge_index=edge_index, "
            f"batch=batch, aux_loss=...) from its forward() method."
        )

    # Check 2: Required fields must be Tensors
    if not isinstance(pool_out.x, Tensor):
        raise TypeError(
            f"Pooling '{pool_name}': pool_out.x must be a Tensor, got {type(pool_out.x).__name__}"
        )
    if not isinstance(pool_out.edge_index, Tensor):
        raise TypeError(
            f"Pooling '{pool_name}': pool_out.edge_index must be a Tensor, "
            f"got {type(pool_out.edge_index).__name__}"
        )
    if not isinstance(pool_out.batch, Tensor):
        raise TypeError(
            f"Pooling '{pool_name}': pool_out.batch must be a Tensor, got {type(pool_out.batch).__name__}"
        )

    # Check 3: Shape consistency
    if pool_out.x.dim() != 2:
        raise ValueError(
            f"Pooling '{pool_name}': pool_out.x must be 2D [N, F], got shape {list(pool_out.x.shape)}"
        )
    
    if pool_out.edge_index.dim() != 2 or pool_out.edge_index.size(0) != 2:
        raise ValueError(
            f"Pooling '{pool_name}': pool_out.edge_index must have shape [2, E], "
            f"got shape {list(pool_out.edge_index.shape)}"
        )
    
    if pool_out.batch.dim() != 1:
        raise ValueError(
            f"Pooling '{pool_name}': pool_out.batch must be 1D, got shape {list(pool_out.batch.shape)}"
        )
    
    if pool_out.x.size(0) != pool_out.batch.size(0):
        raise RuntimeError(
            f"Pooling '{pool_name}': Shape mismatch between x and batch. "
            f"x has {pool_out.x.size(0)} nodes but batch has {pool_out.batch.size(0)} entries."
        )

    # Check 4: Device consistency
    device = pool_out.x.device
    if pool_out.edge_index.device != device:
        raise RuntimeError(
            f"Pooling '{pool_name}': Device mismatch. x is on {device} but "
            f"edge_index is on {pool_out.edge_index.device}."
        )
    if pool_out.batch.device != device:
        raise RuntimeError(
            f"Pooling '{pool_name}': Device mismatch. x is on {device} but "
            f"batch is on {pool_out.batch.device}."
        )

    # Check 5: Optional fields - type and device consistency
    if pool_out.edge_attr is not None:
        _check_tensor_field(pool_out.edge_attr, "edge_attr", pool_name, device)

    if pool_out.edge_weight is not None:
        _check_tensor_field(pool_out.edge_weight, "edge_weight", pool_name, device)
        if pool_out.edge_weight.dim() != 1:
            raise ValueError(
                f"Pooling '{pool_name}': edge_weight must be 1D [E], "
                f"got shape {list(pool_out.edge_weight.shape)}"
            )

    if pool_out.perm is not None:
        _check_tensor_field(pool_out.perm, "perm", pool_name, device)

    if pool_out.score is not None:
        _check_tensor_field(pool_out.score, "score", pool_name, device)

    # Check 6: aux_loss must be scalar tensor or None
    if pool_out.aux_loss is not None:
        if not isinstance(pool_out.aux_loss, Tensor):
            raise TypeError(
                f"Pooling '{pool_name}': aux_loss must be a scalar Tensor or None, "
                f"got {type(pool_out.aux_loss).__name__}. "
                f"If you have multiple loss terms, sum them into a single scalar."
            )
        if pool_out.aux_loss.numel() != 1:
            raise ValueError(
                f"Pooling '{pool_name}': aux_loss must be a scalar (1 element), "
                f"got shape {list(pool_out.aux_loss.shape)}. "
                f"Use aux_loss = loss_term.sum() or loss_term.mean() to scalarize."
            )
        if pool_out.aux_loss.device != device:
            raise RuntimeError(
                f"Pooling '{pool_name}': Device mismatch for aux_loss."
            )
