from typing import Optional

from torch import Tensor

from .classifier_base import GraphClassifierBase


class GraphClassifierPlain(GraphClassifierBase):
    def __init__(self, *args, norm: Optional[str] = "layer_norm", **kwargs) -> None:
        super().__init__(*args, norm=norm or "layer_norm", **kwargs)

    def _merge_graph_embeddings(self, before_pool: Optional[Tensor], after_pool: Tensor) -> Tensor:
        return after_pool
