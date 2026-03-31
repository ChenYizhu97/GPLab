from typing import Optional

from torch import Tensor

from .Model import GRAPH_CLASSIFIER_BASE


class GRAPH_CLASSIFIER_PLAIN(GRAPH_CLASSIFIER_BASE):
    def __init__(self, *args, norm: Optional[str] = "layer_norm", **kwargs) -> None:
        super().__init__(*args, norm=norm or "layer_norm", **kwargs)

    def _merge_graph_embeddings(self, before_pool: Optional[Tensor], after_pool: Tensor) -> Tensor:
        return after_pool
