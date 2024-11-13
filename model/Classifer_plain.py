import toml
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, BatchNorm
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.data import Data
from layers.resolver import conv_resolver, pool_resolver
from layers.functional import readout
from typing import Optional


class GRAPH_CLASSIFIER_PLAIN(torch.nn.Module):
    def __init__(
            self, 
            n_node_features:int, 
            n_classes:int, 
            ratio = 0.5,
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
        self.pre_gnn = MLP(channel_list=self.pre_gnn, act=self.nonlinearity, norm="batch_norm", dropout=self.p_dropout) 
        self.bn_pre_gnn = BatchNorm(in_channels=self.hidden_features)
        self.conv1 = self.CONV(self.hidden_features, self.hidden_features)
        self.bn_conv1 = BatchNorm(in_channels=self.hidden_features)
        self.conv2 = self.CONV(self.hidden_features, self.hidden_features)
        self.bn_conv2 = BatchNorm(in_channels=self.hidden_features)

        self.pool1 = pool_resolver(
            self.pool_method, 
            self.hidden_features, 
            ratio=ratio, 
            avg_node_num=avg_node_num, 
            nonlinearity=self.nonlinearity,
        )
        self.bn_pool1 = BatchNorm(in_channels=self.hidden_features)
    
        self.pool2 = pool_resolver(
            self.pool_method, 
            self.hidden_features, 
            ratio=ratio, 
            avg_node_num=avg_node_num, 
            nonlinearity=self.nonlinearity,
        )
        self.bn_pool2 = BatchNorm(in_channels=self.hidden_features)

        self.nonlinearity = activation_resolver(self.nonlinearity)
        self.global_pool = readout
        self.post_gnn = MLP(channel_list=self.post_gnn, act=self.nonlinearity, norm="batch_norm", dropout=self.p_dropout)
        self.aux_loss = 0.0
        self.reset_parameters()

    def forward(
            self, 
            data: Data,
    ) -> torch.Tensor:
        
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.pre_gnn(x)
        x = self.bn_pre_gnn(x)
        
        x = self.nonlinearity(self.conv1(x, edge_index))
        x = self.bn_conv1(x)


        #pooling
        x, edge_index, batch, aux_loss = GRAPH_CLASSIFIER_PLAIN._pool(self.pool1, x=x, edge_index=edge_index, batch=batch)
        self.aux_loss += aux_loss
        x = self.bn_pool1(x)

        x = self.nonlinearity(self.conv2(x, edge_index))
        x = self.bn_conv2(x)

        x, edge_index, batch, aux_loss = GRAPH_CLASSIFIER_PLAIN._pool(self.pool2, x=x, edge_index=edge_index, batch=batch)
        self.aux_loss += aux_loss
        x = self.bn_pool2(x)

        x = self.global_pool(x=x, batch=batch)
        x = self.post_gnn(x)
        
        y = F.log_softmax(x, dim=1)
        return y, self.aux_loss
    
    def reset_parameters(self):
        self.pre_gnn.reset_parameters()
        if self.pool1 is not None: self.pool1.reset_parameters()
        if self.pool2 is not None: self.pool2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post_gnn.reset_parameters()

    @classmethod
    def _pool(cls, pool:callable, x, edge_index, batch):
        aux_loss = 0.0
        if pool is not None:
            pool_out = pool(x=x, edge_index=edge_index, batch=batch)
            if len(pool_out) == 4:
                # dense pool
                x, edge_index, batch, aux_loss = pool(x, edge_index, batch)
            elif len(pool_out) == 6:
                # sparse pool
                x, edge_index, edge_attr, batch, perm, score = pool(x, edge_index=edge_index, batch=batch)
            else:
                x, edge_index, edge_attr, batch, perm = pool(x, edge_index=edge_index, batch=batch)

        return x, edge_index, batch, aux_loss

    def _load_from_config(
            self, 
            config:dict
    ):
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
        self.CONV = conv_resolver(config["conv_layer"])

