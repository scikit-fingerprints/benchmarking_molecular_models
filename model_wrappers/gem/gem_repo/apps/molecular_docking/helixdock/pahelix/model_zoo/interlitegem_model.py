import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import GraphPool

from pahelix.networks.gnn_block import GIN
from pahelix.networks.compound_encoder import AtomEmbedding, AtomFloatEmbedding, BondEmbedding, \
        BondFloatRBF, BondAngleFloatRBF
from pahelix.utils.compound_tools import CompoundKit
from pahelix.networks.gnn_block import MeanPool, GraphNorm
from pahelix.networks.basic_block import MLP
from pahelix.model_zoo.light_gem_model import LiteGEMConv, norm_layer


class GraphEdgePool(nn.Layer):
    """Implementation of graph pooling

    This is an implementation of graph "edge" pooling

    Args:
        graph: the graph object from (:code:`Graph`)

        feature: A tensor with shape (num_edges, feature_size).

        pool_type: The type of pooling ("sum", "mean" , "min", "max")

    Return:
        A tensor with shape (num_graph, feature_size)
    """

    def __init__(self, pool_type=None):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, graph, feature, pool_type=None):
        if pool_type is not None:
            warnings.warn("The pool_type (%s) argument in forward function " \
                    "will be discarded in the future, " \
                    "please initialize it when creating a GraphPool instance.")
        else:
            pool_type = self.pool_type
        graph_feat = pgl.math.segment_pool(feature, graph.graph_edge_id, pool_type)
        return graph_feat


class EdgeWeightedSumAndMax(nn.Layer):
    """
    FIXME PGL does not support the pooling on edges
    """
    def __init__(self, feat_size):
        super(EdgeWeightedSumAndMax, self).__init__()
        self.atom_weighting = nn.Sequential(
            nn.Linear(feat_size, 1),
            nn.Tanh()
        )
        self.weight_and_sum = EdgeWeightAndSum(feat_size)

    def forward(self, g, edge_feats):
        edge_w = self.atom_weighting(edge_feats)
        ef_temp = edge_feats * edge_w

        # h_g_sum = self.weight_and_sum(bg, edge_feats)  # normal version
        # # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        # with bg.local_scope():
        #     bg.edata['e'] = edge_feats
        #     h_g_max = dgl.max_edges(bg, 'e')
        # h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        # return h_g  # normal version
        # # return h_g, weights  # temporary version


class InterLiteGEM(nn.Layer):
    """
    The GeoGNN Model used in GEM.

    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, model_config={}):
        super().__init__()

        self.model_config = model_config

        self.num_layers = model_config.num_layers
        self.triangle_layers = model_config.triangle_layers
        self.dropout = model_config.dropout
        self.add_virtual_node = model_config.add_virtual_node
        self.add_virtual_edge = model_config.add_virtual_edge
        self.graph_feat_size = model_config.graph_feat_size
        graph_feat_size = model_config.graph_feat_size
        node_feat_size = model_config.node_feat_size
        edge_feat_size = model_config.edge_feat_size

        self.project_node = nn.LayerList([
            nn.Linear(node_feat_size, graph_feat_size),
            nn.Swish()]
        )
        self.project_edge = nn.LayerList([
            nn.Linear(edge_feat_size, graph_feat_size),
            nn.Swish()]
        )
        self.norm = 'batch'
        self.model_config['norm'] = self.norm

        self.atom_names = model_config.embedding_layer.atom_names
        self.atom_float_names = model_config.embedding_layer.atom_float_names
        self.bond_names = model_config.embedding_layer.bond_names
        self.non_cov_names = model_config.embedding_layer.bond_float_names
        self.atom_embedding = AtomEmbedding(self.atom_names, self.graph_feat_size)
        self.atom_float_embedding = AtomFloatEmbedding(self.atom_float_names, self.graph_feat_size)
        self.bond_embedding = BondEmbedding(self.bond_names, self.graph_feat_size)
        self.non_cov_embedding = BondFloatRBF(self.non_cov_names, self.graph_feat_size, rbf_params={
            'dis': (np.arange(0, 6, 0.2), 20.0)
        })

        self.cov_layers = paddle.nn.LayerList()
        self.non_cov_layers = paddle.nn.LayerList()
        self.cov_norms = paddle.nn.LayerList()
        self.non_cov_norms = paddle.nn.LayerList()
        for layer in range(self.num_layers):
            self.cov_layers.append(LiteGEMConv(self.model_config, with_efeat=False, update='node'))
            self.cov_norms.append(norm_layer(self.norm, self.graph_feat_size))
            self.non_cov_layers.append(LiteGEMConv(self.model_config, with_efeat=False, update='edge'))
            self.non_cov_norms.append(norm_layer(self.norm, self.graph_feat_size))

        self.last_cov_norm = norm_layer(self.norm, self.graph_feat_size)
        self.last_non_cov_norm = norm_layer(self.norm, self.graph_feat_size)

        self.pool_sum = GraphPool(pool_type="sum")
        self.pool_max = GraphPool(pool_type="max")
        self.edge_pool_sum = GraphEdgePool(pool_type="sum")
        self.edge_pool_max = GraphEdgePool(pool_type="max")
        # if self.add_virtual_node:
        #     self.virtualnode_embedding = torch.nn.Embedding(1, graph_feat_size)
        #     self.mlp_virtualnode_list = torch.nn.ModuleList()
        #     for layer_index in range(self.num_layers):
        #         self.mlp_virtualnode_list.append(MLP([graph_feat_size]*3, norm=norm))
        # if self.add_virtual_edge:
        #     self.virtualnode_embedding_e = torch.nn.Embedding(1, graph_feat_size)
        #     self.mlp_virtualnode_list_e = torch.nn.ModuleList()
        #     for layer_index in range(self.num_layers):
        #         self.mlp_virtualnode_list_e.append(MLP([graph_feat_size]*3, norm=norm))
        head_input_size = graph_feat_size * 4
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(head_input_size, head_input_size//2),
            nn.BatchNorm1D(head_input_size//2),
            nn.Swish(),
            nn.Linear(head_input_size//2, 1)
        )
        # self.loss_func = nn.SmoothL1Loss()
        self.loss_func = nn.MSELoss()

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.graph_feat_size

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.graph_feat_size

    def forward(self, batched_graph):
        """
        Build the network.
        """
        cov_graph = batched_graph['cov_graph']
        non_cov_graph = batched_graph['non_cov_graph']
        node_hidden = self.atom_embedding(cov_graph.node_feat)
        node_hidden += self.atom_float_embedding(cov_graph.node_feat)
        cov_hidden = self.bond_embedding(cov_graph.edge_feat)
        non_cov_hidden = self.non_cov_embedding(non_cov_graph.edge_feat)

        node_h, _ = self.cov_layers[0](cov_graph, node_hidden, cov_hidden)
        _, non_cov_eh = self.non_cov_layers[0](non_cov_graph, node_hidden, non_cov_hidden)

        for layer_id in range(1, self.num_layers):
            nh_in = self.cov_norms[layer_id](node_hidden)
            nh_in = F.swish(nh_in)
            nh_in = F.dropout(nh_in, p=self.dropout, training=self.training)

            nch_in = self.non_cov_norms[layer_id](non_cov_hidden)
            nch_in = F.swish(nch_in)
            nch_in = F.dropout(nch_in, p=self.dropout, training=self.training)

            nh_temp, _ = self.cov_layers[layer_id](cov_graph, nh_in, cov_hidden)
            _, nch_temp = self.non_cov_layers[layer_id](non_cov_graph, nh_in, nch_in)

            node_h = node_h + nh_temp
            non_cov_eh = non_cov_eh + nch_temp
        
        node_h = self.cov_norms[0](node_h)
        non_cov_eh = self.non_cov_norms[0](non_cov_eh)
        node_h = F.dropout(node_h, p=self.dropout, training=self.training)
        non_cov_eh = F.dropout(non_cov_eh, p=self.dropout, training=self.training)
        
        h_cov_graph = paddle.concat([self.pool_sum(cov_graph, node_h), 
                                     self.pool_max(cov_graph, node_h)], axis=1)
        h_non_cov_graph = paddle.concat([self.edge_pool_sum(non_cov_graph, non_cov_eh),
                                         self.edge_pool_max(non_cov_graph, non_cov_eh)], axis=1)
        pred = self.graph_pred_linear(paddle.concat([h_cov_graph, h_non_cov_graph], axis=1))
        loss = self.loss_func(pred.reshape((-1,)), batched_graph['labels'])
        return {'loss': loss, 'pred': pred}
