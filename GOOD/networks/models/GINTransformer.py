import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_dense_batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINEncoder, GINMolEncoder, GINFeatExtractor
from .Pooling import GlobalAddPool

class GINTransformerFeatExtractor(GINFeatExtractor):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINTransformerFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = GINTransformerMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = GINTransformerEncoder(config, **kwargs)
            self.edge_feat = False


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):
        super(TransformerEncoder, self).__init__()
        num_layer = config.model.model_layer
        self.attns = nn.ModuleList([nn.MultiheadAttention(config.model.dim_hidden, config.model.num_heads, dropout=config.model.dropout_rate, batch_first=True)
                for _ in range(num_layer)
        ])
        self.attn_norms = nn.ModuleList([
                nn.BatchNorm1d(config.model.dim_hidden, track_running_stats=True)
                for _ in range(num_layer)
        ])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(config.model.dim_hidden, config.model.dim_hidden * 2),
                          nn.ReLU(),
                          nn.Linear(config.model.dim_hidden * 2, config.model.dim_hidden))
            for _ in range(num_layer)
        ])

class GINTransformerEncoder(GINEncoder, TransformerEncoder):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINTransformerEncoder, self).__init__(config, **kwargs)
        self.config = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GT encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            graph feature representations
        """
        node_repr = self.get_node_repr(x, edge_index, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GT encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        layer_feat = [x]
        for i, (conv, attn, attn_norm, ff, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.attns, self.attn_norms, self.ff, self.batch_norms, self.relus, self.dropouts)):
            h_local = conv(layer_feat[-1], edge_index)
            h_dense, mask = to_dense_batch(h_local, batch)
            h_attn = attn(h_dense, h_dense, h_dense, attn_mask=None, key_padding_mask=(~mask), need_weights=False)[0][mask]
            h_attn = attn_norm(h_local + h_attn)
            post_conv = batch_norm(ff(h_attn))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
        return layer_feat[-1]


class GINTransformerMolEncoder(GINMolEncoder, TransformerEncoder):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINTransformerMolEncoder, self).__init__(config, **kwargs)
        self.config: Union[CommonArgs, Munch] = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GT encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            graph feature representations
        """
        node_repr = self.get_node_repr(x, edge_index, edge_attr, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GT encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        layer_feat = [self.atom_encoder(x)]
        for i, (conv, attn, attn_norm, ff, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.attns, self.attn_norms, self.ff, self.batch_norms, self.relus, self.dropouts)):
            h_local = conv(layer_feat[-1], edge_index, edge_attr)
            h_dense, mask = to_dense_batch(h_local, batch)
            h_attn = attn(h_dense, h_dense, h_dense, attn_mask=None, key_padding_mask=(~mask), need_weights=False)[0][mask]
            h_attn = attn_norm(h_local + h_attn)
            post_conv = batch_norm(ff(h_attn))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
        return layer_feat[-1]
