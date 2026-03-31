from torch import Tensor

from .Model import GRAPH_CLASSIFIER_BASE


class GRAPH_CLASSIFIER_SUM(GRAPH_CLASSIFIER_BASE):
    def _readout_before_pool(self, x: Tensor, batch: Tensor) -> Tensor:
        return self.global_pool(x=x, batch=batch)

    def _merge_graph_embeddings(self, before_pool: Tensor, after_pool: Tensor) -> Tensor:
        return before_pool + after_pool
