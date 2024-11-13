import toml
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import MLP 
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.data import Data
#from LSPool.layers.resolver import conv_resolver, pool_resolver
from layers.resolver import conv_resolver, pool_resolver
from typing import Optional
from torch_geometric.nn.pool import global_mean_pool

DENSE_POOL = ["mincutpool", "diffpool"]
SPARSE_POOL = ["topkpool", "sagpool", "lspool"]

DEFAULT_CONF = "config/config.toml"

def _get_convs(
        conv, 
        hidden_features, 
        layers,
    ):

    CONV = conv_resolver(conv)
    module_list = []
    for _ in range(0, layers):
        module_list.append(CONV(hidden_features, hidden_features))
    return ModuleList(module_list)

class GRAPH_CLASSIFIER(torch.nn.Module):
    def __init__(
            self, 
            n_node_features:int, 
            n_classes:int, 
            pool_method:Optional[str]=None,
            config:Optional[dict]=None,
            avg_node_num:Optional[float]=None,
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        #load model config from file
        self._load_from_config(config)
      
        self.pool_method = pool_method
        self.pre_gnn = MLP(channel_list=self.pre_gnn, act=self.nonlinearity, norm=None, dropout=self.p_dropout)  

        self.pre_pool_convs = _get_convs(
            self.conv,
            self.hidden_features,
            self.pre_pool_convs
        )
    
        self.pool = pool_resolver(
            self.pool_method, 
            self.hidden_features, 
            ratio=0.5, 
            avg_node_num=avg_node_num, 
            nonlinearity=self.nonlinearity,
        )

        self.post_pool_convs = _get_convs(
            self.conv,
            self.hidden_features,
            self.post_pool_convs
        ) 
        self.nonlinearity = activation_resolver(self.nonlinearity)
        self.global_pool = global_mean_pool
        self.post_gnn = MLP(channel_list=self.post_gnn, act=self.nonlinearity, norm=None, dropout=self.p_dropout)
        self.aux_loss = 0.0

        self.reset_parameters()

    def forward(
            self, 
            data: Data,
    ) -> torch.Tensor:
        
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.pre_gnn(x)
        
        # pre pool conv layers
        for conv in self.pre_pool_convs:  
            x = conv(x, edge_index)
            x = self.nonlinearity(x)

        #x_out = self.global_pool(x, batch)

        #pooling
        if self.pool is not None:
            pool_out = self.pool(x=x, edge_index=edge_index, batch=batch)
            if len(pool_out) == 4:
                # dense pool
                x, edge_index, batch, self.aux_loss = self.pool(x, edge_index, batch)
            else:
                # sparse pool
                x, edge_index, edge_attr, batch, perm, score = self.pool(x, edge_index=edge_index, batch=batch)
            
        # post pool conv layers
        for conv in self.post_pool_convs: 
            x = conv(x, edge_index)
            x = self.nonlinearity(x)
        
        #readout
        x = self.global_pool(x=x, batch=batch)
        #x = x + x_out
        x = self.post_gnn(x)
        
        y = F.log_softmax(x, dim=1)
        return y, self.aux_loss
    
    def reset_parameters(self):
        self.pre_gnn.reset_parameters()
        [conv.reset_parameters() for conv in self.pre_pool_convs]
        if self.pool is not None: self.pool.reset_parameters()
        [conv.reset_parameters() for conv in self.post_pool_convs]
        self.post_gnn.reset_parameters()

    def _load_from_config(
            self, 
            config:Optional[dict] 
    ):
        # Load trainning and model setting from default config file if no dict provided
        if config is None:
            config = toml.load(DEFAULT_CONF)["model"]

        #Trainning setting
        self.p_dropout = config["p_dropout"]
        self.hidden_features = config["hidden_features"]
        # the activation should not be learnable
        self.nonlinearity = activation_resolver(config["nonlinearity"])

        #Model setting
        self.pre_gnn = [self.n_node_features]
        self.pre_gnn.extend(config["pre_gnn"])

        self.post_gnn = config["post_gnn"]
        self.post_gnn.append(self.n_classes)
        self.pre_pool_convs = config["pre_pool_convs"]
        self.post_pool_convs = config["post_pool_convs"]

        self.conv = config["conv_layer"]

