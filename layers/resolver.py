from importlib import import_module
from typing import Callable, Optional, Union
import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, GCNConv, DenseGCNConv, GINConv
from torch_geometric.nn.pool import TopKPooling, ASAPooling
from .pool import SAGPooling, SparsePooling, PoolAdapter

BUILTIN_POOLS = (
    "nopool",
    "topkpool",
    "sagpool",
    "asapool",
    "sparsepool",
    "mincutpool",
    "diffpool",
    "densepool",
)


def conv_resolver(layer:str) -> Optional[torch.nn.Module]:
    # add more convolution layer resolvers if use other convolution layers
    if layer == "GCN": return GCNConv
    if layer == "GraphConv": return GraphConv
    if layer == "GIN": return lambda in_channel, out_channel: GINConv(nn=Linear(in_channel, out_channel))
    return None


def list_builtin_pools() -> tuple[str, ...]:
    return BUILTIN_POOLS


def _dense_cluster_size(avg_node_num: Optional[float], ratio: float) -> int:
    if avg_node_num is None:
        raise ValueError("avg_node_num is required for dense pooling methods.")
    return max(1, int(avg_node_num * ratio))


def _load_pool_factory(path: str) -> Callable[..., torch.nn.Module]:
    module_name, sep, factory_name = path.partition(":")
    if sep == "" or not module_name or not factory_name:
        raise ValueError(
            "Custom pool must use '<python_module>:<factory_name>', "
            f"got '{path}'."
        )

    module = import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Cannot find callable pool factory '{factory_name}' in '{module_name}'.")
    return factory


def pool_resolver(pool:str, in_channels:int, ratio:float=0.5, avg_node_num:Optional[float]=None, nonlinearity:Union[str, callable]="relu") -> Optional[torch.nn.Module]:
    # For dense pooling methods, this returns the learnable assignment module wrapped by PoolAdapter.
    if pool in (None, "", "nopool"):
        return None

    if pool == "topkpool":
        return TopKPooling(in_channels, ratio=ratio)
    if pool == "sagpool":
        return SAGPooling(in_channels, ratio=ratio)
    if pool == "asapool":
        return ASAPooling(in_channels, ratio=ratio)
    if pool == "sparsepool":
        return SparsePooling(in_channels, ratio=ratio)

    if pool in ("mincutpool", "diffpool", "densepool"):
        k = _dense_cluster_size(avg_node_num, ratio)
        if pool == "mincutpool":
            return PoolAdapter(Linear(in_channels, k), pool)
        if pool == "diffpool":
            return PoolAdapter(DenseGCNConv(in_channels, k), pool, nonlinearity=nonlinearity)
        return PoolAdapter(Linear(in_channels, k), pool)

    if ":" in pool:
        factory = _load_pool_factory(pool)
        try:
            return factory(
                in_channels=in_channels,
                ratio=ratio,
                avg_node_num=avg_node_num,
                nonlinearity=nonlinearity,
            )
        except TypeError:
            # Backward-compatible fallback for factories that only accept (in_channels, ratio).
            return factory(in_channels, ratio)

    raise ValueError(
        f"Unknown pooling method '{pool}'. "
        f"Built-ins: {', '.join(BUILTIN_POOLS)}. "
        "Or provide a custom factory as '<python_module>:<factory_name>'."
    )

