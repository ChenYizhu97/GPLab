import toml
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, BatchNorm
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.data import Data
from layers.resolver import conv_resolver, pool_resolver
from layers.functional import readout
from typing import Optional
from .Model import MODEL
DENSE_POOL = ["mincutpool", "diffpool"]
SPARSE_POOL = ["topkpool", "sagpool", "lspool"]

DEFAULT_CONF = "config/model.toml"


class GRAPH_CLASSIFIER_SUM(MODEL):
    def __init__(
            self, 
            n_node_features:int, 
            n_classes:int, 
            pool_method:Optional[str]=None,
            ratio:float=0.5,
            config:Optional[dict]=None,
            avg_node_num:Optional[float]=None,
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_node_features = n_node_features
        self.n_classes = n_classes

        #load model config from file if no dict given
        if config is None:
            print("No config provided to model...Using default config...")
            config = toml.load(DEFAULT_CONF)["model"]

        self._load_from_config(config)

        self.pool_method = pool_method
        self.nonlinearity = activation_resolver(self.nonlinearity)
        self.CONV = conv_resolver(self.CONV)
        self.pre_gnn = MLP(channel_list=self.pre_gnn, act=self.nonlinearity, norm="batch_norm", dropout=self.p_dropout) 
        self.bn_pre_gnn = BatchNorm(in_channels=self.hidden_features)

        self.pool = pool_resolver(
            self.pool_method, 
            self.hidden_features, 
            ratio=ratio, 
            avg_node_num=avg_node_num, 
            nonlinearity=self.nonlinearity,
        )
        self.bn_pool = BatchNorm(in_channels=self.hidden_features)

        self.conv1 = self.CONV(self.hidden_features, self.hidden_features)
        self.bn_conv1 = BatchNorm(in_channels=self.hidden_features)

        self.conv2 = self.CONV(self.hidden_features, self.hidden_features)
        self.bn_conv2 = BatchNorm(in_channels=self.hidden_features)

        self.global_pool = readout
        self.post_gnn = MLP(channel_list=self.post_gnn, act=self.nonlinearity, norm="batch_norm", dropout=self.p_dropout)

        self.reset_parameters()

    def forward(
            self, 
            data: Data,
    ) -> torch.Tensor:
        
        x, edge_index, batch, _ = data.x, data.edge_index, data.batch, data.edge_attr


        x = self.pre_gnn(x)
        x = self.bn_pre_gnn(x)
        
        x = self.nonlinearity(self.conv1(x, edge_index))
        x = self.bn_conv1(x)

        x_out_before_pool = self.global_pool(x=x, batch=batch)

        #pooling
        x, edge_index, batch, aux_loss = self._pool(self.pool, x=x, edge_index=edge_index, batch=batch)


        x = self.bn_pool(x)

        x = self.nonlinearity(self.conv2(x, edge_index))
        x = self.bn_conv2(x)

        x_out = self.global_pool(x=x, batch=batch)
        
        x = x_out_before_pool + x_out
        x = self.post_gnn(x)
        
        y = F.log_softmax(x, dim=1)

        return y, aux_loss
    
    def reset_parameters(self):
        self.pre_gnn.reset_parameters()
        if self.pool is not None: self.pool.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post_gnn.reset_parameters()

