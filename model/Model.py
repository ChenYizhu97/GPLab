from abc import ABC, abstractmethod
import inspect
from typing import Optional

import toml
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MLP, BatchNorm, LayerNorm
from torch_geometric.nn.resolver import activation_resolver

from layers.functional import readout
from layers.pool.contracts import validate_pool_output
from layers.resolver import conv_resolver, pool_resolver

DEFAULT_CONF = "config/model.toml"


class MODEL(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pool_validated = False

    def forward(self, data: Data):
        raise NotImplementedError

    def _pool(
        self,
        pool: callable,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if pool is None:
            return x, edge_index, batch, None, None

        pool_out = pool(x=x, edge_index=edge_index, batch=batch)

        if not self._pool_validated:
            validate_pool_output(pool_out, pool.__class__.__name__)
            self._pool_validated = True

        return (
            pool_out.x,
            pool_out.edge_index,
            pool_out.batch,
            pool_out.edge_weight,
            pool_out.aux_loss,
        )

    def _load_from_config(self, config: dict) -> None:
        self.p_dropout = config["p_dropout"]
        self.hidden_features = config["hidden_features"]
        self.nonlinearity = config["nonlinearity"]

        self.pre_gnn = [self.n_node_features, *list(config["pre_gnn"])]
        self.post_gnn = [*list(config["post_gnn"]), self.n_classes]
        self.CONV = config["conv_layer"]


class GRAPH_CLASSIFIER_BASE(MODEL, ABC):
    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        pool_method: Optional[str] = None,
        ratio: float = 0.5,
        config: Optional[dict] = None,
        avg_node_num: Optional[float] = None,
        norm: str = "layer_norm",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        self.pool_method = pool_method
        self.norm_name = norm

        if config is None:
            print("No config provided to model...Using default config...")
            config = toml.load(DEFAULT_CONF)["model"]

        self._load_from_config(config)

        self.nonlinearity = activation_resolver(self.nonlinearity)
        self.conv_layer = conv_resolver(self.CONV)
        self.norm_layer = self._resolve_norm(norm)

        self.pre_gnn = self._build_pre_gnn()
        self.pool = pool_resolver(
            self.pool_method,
            self.hidden_features,
            ratio=ratio,
            avg_node_num=avg_node_num,
            nonlinearity=self.nonlinearity,
        )
        self.conv1 = self.conv_layer(self.hidden_features, self.hidden_features)
        self.ln_conv1 = self.norm_layer(self.hidden_features)
        self.conv2 = self.conv_layer(self.hidden_features, self.hidden_features)
        self.ln_conv2 = self.norm_layer(self.hidden_features)
        self.global_pool = readout
        self.post_gnn = self._build_post_gnn()

        self.reset_parameters()

    def forward(self, data: Data) -> tuple[Tensor, Optional[Tensor]]:
        x, edge_index, batch, edge_weight = self._unpack_graph(data)

        x = self.pre_gnn(x)
        x = self._apply_conv_block(self.conv1, self.ln_conv1, x, edge_index, edge_weight=edge_weight)

        before_pool = self._readout_before_pool(x, batch)
        x, edge_index, batch, edge_weight, aux_loss = self._pool(self.pool, x=x, edge_index=edge_index, batch=batch)

        x = self._apply_conv_block(self.conv2, self.ln_conv2, x, edge_index, edge_weight=edge_weight)
        after_pool = self.global_pool(x=x, batch=batch)

        graph_embedding = self._merge_graph_embeddings(before_pool, after_pool)
        logits = self.post_gnn(graph_embedding)
        y = F.log_softmax(logits, dim=1)

        return y, aux_loss

    def reset_parameters(self) -> None:
        self.pre_gnn.reset_parameters()
        if self.pool is not None:
            self.pool.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post_gnn.reset_parameters()
        self.ln_conv1.reset_parameters()
        self.ln_conv2.reset_parameters()

    def _resolve_norm(self, norm: str):
        if norm == "layer_norm":
            return LayerNorm
        if norm == "batch_norm":
            return BatchNorm
        raise ValueError(f"Unsupported norm '{norm}'. Use 'layer_norm' or 'batch_norm'.")

    def _build_pre_gnn(self) -> MLP:
        return MLP(
            channel_list=self.pre_gnn,
            act=self.nonlinearity,
            norm=self.norm_name,
            bias=True,
            plain_last=False,
            dropout=self.p_dropout,
        )

    def _build_post_gnn(self) -> MLP:
        bias = [True] * (len(self.post_gnn) - 2) + [False]
        return MLP(
            channel_list=self.post_gnn,
            act=self.nonlinearity,
            norm=self.norm_name,
            bias=bias,
            plain_last=True,
            dropout=self.p_dropout,
        )

    def _unpack_graph(self, data: Data) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        edge_weight = getattr(data, "edge_weight", None)
        return data.x, data.edge_index, batch, edge_weight

    def _apply_conv_block(
        self,
        conv: torch.nn.Module,
        norm: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_weight is not None and self._conv_supports_edge_weight(conv):
            x = conv(x, edge_index, edge_weight=edge_weight)
        else:
            if edge_weight is not None:
                edge_index = self._filter_zero_weight_edges(edge_index, edge_weight)
            x = conv(x, edge_index)
        x = norm(x)
        return self.nonlinearity(x)

    @staticmethod
    def _conv_supports_edge_weight(conv: torch.nn.Module) -> bool:
        params = inspect.signature(conv.forward).parameters
        return "edge_weight" in params

    @staticmethod
    def _filter_zero_weight_edges(edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        keep_mask = edge_weight != 0
        if bool(keep_mask.all()):
            return edge_index
        return edge_index[:, keep_mask]

    def _readout_before_pool(self, x: Tensor, batch: Tensor) -> Optional[Tensor]:
        return None

    @abstractmethod
    def _merge_graph_embeddings(
        self,
        before_pool: Optional[Tensor],
        after_pool: Tensor,
    ) -> Tensor:
        raise NotImplementedError
