"""Example custom pooling plugin for GPLab.

Usage:
    python3 main.py --pooling examples.custom_pool_plugin:build_pool --pool-ratio 0.6
"""

from torch_geometric.nn.pool import TopKPooling


def build_pool(
    in_channels: int,
    ratio: float = 0.5,
    avg_node_num=None,
    nonlinearity="relu",
):
    # `avg_node_num` and `nonlinearity` are accepted for compatibility with GPLab's plugin API.
    return TopKPooling(in_channels, ratio=ratio)
