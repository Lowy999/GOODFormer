import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_batch, to_dense_adj
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn.conv import MessagePassing

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .MolEncoders import AtomEncoder, BondEncoder
from .Encoders import DummyEdgeEncoder, LapPENodeEncoder
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from .Classifiers import Classifier


@register.model_register
class GOODFormer(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GOODFormer, self).__init__(config)
        self.dataset_type = config.dataset.dataset_type
        if self.dataset_type == "mol":
            self.atom_encoder = AtomEncoder(
                config.model.dim_hidden - config.posenc_LapPE.dim_pe, config
            )
            self.linear_encoder = nn.Linear(
                config.model.dim_hidden - config.posenc_LapPE.dim_pe,
                config.model.dim_hidden,
            )
            self.edge_encoder = BondEncoder(config.model.dim_hidden, config)
        else:
            self.linear_encoder = nn.Linear(
                config.dataset.dim_node, config.model.dim_hidden
            )
            self.edge_encoder = DummyEdgeEncoder(config.model.dim_hidden)

        transformer_layers = []
        for _ in range(max(2, config.model.model_layer - 2)):
            transformer_layers.append(TransformerLayer(config))
        self.transformer_layers = nn.ModuleList(transformer_layers)

        disentanglement_transformer_layers = []
        for i in range(2):
            if i == 0:
                disentanglement_transformer_layers.append(
                    DisentanglementTransformerLayer(config, True)
                )
            else:
                disentanglement_transformer_layers.append(
                    DisentanglementTransformerLayer(config, False)
                )
        self.disentanglement_transformer_layers = nn.ModuleList(
            disentanglement_transformer_layers
        )

        self.encoding_classifier = nn.Linear(config.posenc_LapPE.dim_pe, config.posenc_LapPE.eigen.max_freqs)

        self.readout = GlobalMeanPool()
        self.causal_classifier = Classifier(config)
        self.conf_classifier = Classifier(config)

        self.num_heads = config.model.num_heads
        self.register_buffer(
            "mean_causal_attention_entropy",
            torch.zeros(self.num_heads),
        )
        self.register_buffer(
            "mean_conf_attention_entropy",
            torch.zeros(self.num_heads),
        )
        self.sample_count = 0

    def get_node_rep(self, data):
        if self.dataset_type == "mol":
            data.x = self.atom_encoder(data.x)
            data.edge_attr = self.edge_encoder(data.edge_attr)
        else:
            data = self.edge_encoder(data)
        data.x = self.linear_encoder(data.x)

        for transformer_layer in self.transformer_layers:
            data.x = transformer_layer(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
        return data

    def forward(self, *args, **kwargs):
        r"""
        The GOODFormer model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get("data")
        batch_size = kwargs.get("batch_size") or (data.batch[-1].item() + 1)
        if self.training:
            temperature = torch.ones(
                data.batch[-1].item() + 1,
                self.num_heads,
                1,
                1,
                device=data.x.device,
            )
        else:
            temperature = kwargs.get("temperature")

        data = self.get_node_rep(data)

        causal_attn_entropy_list = []
        conf_attn_entropy_list = []
        for i, disentanglement_transformer_layer in enumerate(
            self.disentanglement_transformer_layers
        ):
            if i == 0:
                (
                    data,
                    causal_x,
                    conf_x,
                    causal_attn_entropy,
                    conf_attn_entropy,
                    causal_encoding,
                    conf_encoding,
                    mpnn_causal_mask,
                ) = disentanglement_transformer_layer(data, temperature)
            else:
                (
                    data,
                    causal_feature,
                    conf_feature,
                    causal_attn_entropy,
                    conf_attn_entropy,
                    _,
                    _,
                    mpnn_causal_mask,
                ) = disentanglement_transformer_layer(data, 1)
            causal_attn_entropy_list.append(causal_attn_entropy)
            conf_attn_entropy_list.append(conf_attn_entropy)
        causal_attn_entropy_list = torch.stack(causal_attn_entropy_list, dim=0)
        conf_attn_entropy_list = torch.stack(conf_attn_entropy_list, dim=0)

        # --- Causal repr ---
        if self.dataset_type == 'syn':
            causal_rep = self.readout(causal_x + causal_feature, data.batch, batch_size)
        else:
            causal_rep = self.readout(causal_x, data.batch, batch_size)
        causal_out = self.causal_classifier(causal_rep)

        tmp_causal_mean = torch.mean(causal_attn_entropy_list, dim=1)[0]  # H
        tmp_conf_mean = torch.mean(conf_attn_entropy_list, dim=1)[0]  # H

        if self.training:
            causal_encoding_out = self.encoding_classifier(causal_encoding)
            conf_encoding_out = self.encoding_classifier(conf_encoding)

            if self.sample_count == 0:
                self.mean_causal_attention_entropy.copy_(
                    tmp_causal_mean.clone().detach()
                )
                self.mean_conf_attention_entropy.copy_(tmp_conf_mean.clone().detach())
            else:
                self.mean_causal_attention_entropy = (
                    0.99 * self.mean_causal_attention_entropy
                    + 0.01 * tmp_causal_mean.clone().detach()
                )
                self.mean_conf_attention_entropy = (
                    0.99 * self.mean_conf_attention_entropy
                    + 0.01 * tmp_conf_mean.clone().detach()
                )
            self.sample_count += batch_size

            # --- Conf repr ---
            if self.dataset_type == 'syn':
                conf_rep = self.readout(conf_x + conf_feature, data.batch, batch_size).detach()
            else:
                conf_rep = self.readout(conf_x, data.batch, batch_size).detach()
            conf_out = self.conf_classifier(conf_rep)

            # --- combine to causal phase (detach the conf phase) ---
            rep_out = []
            for conf in conf_rep:
                rep_out.append(
                    torch.sigmoid(self.conf_classifier(conf).detach())
                    * causal_out
                )
            rep_out = torch.stack(rep_out, dim=0)

            return (
                rep_out,
                causal_out,
                conf_out,
                causal_attn_entropy_list,
                conf_attn_entropy_list,
                causal_encoding_out,
                conf_encoding_out,
                data.EigVecs,
            )
        else:

            return causal_out


class DisentanglementTransformerLayer(nn.Module):
    def __init__(self, config: Union[CommonArgs, Munch], use_encoding):
        super().__init__()
        dim_h = config.model.dim_hidden
        dropout_rate = config.model.dropout_rate

        self.num_heads = config.model.num_heads
        assert dim_h % self.num_heads == 0, "dim_h must be divisible by num_heads"

        self.q_proj = nn.Linear(dim_h, dim_h)
        self.k_proj = nn.Linear(dim_h, dim_h)
        self.scale = (dim_h // self.num_heads) ** -0.5

        self.use_encoding = use_encoding
        self.encoding_dim = config.posenc_LapPE.dim_pe
        if self.use_encoding:
            encoding_mpnn = []
            for _ in range(3):
                encoding_mpnn.append(
                    pyg_nn.GINEConv(
                        nn.Sequential(
                            Linear_pyg(self.encoding_dim, self.encoding_dim * 2),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate),
                            Linear_pyg(self.encoding_dim * 2, self.encoding_dim),
                            nn.Dropout(dropout_rate),
                            nn.BatchNorm1d(self.encoding_dim),
                        ),
                        edge_dim=dim_h,
                    )
                )
            self.encoding_mpnn = nn.ModuleList(encoding_mpnn)
            self.encoding_linear = nn.Linear(dim_h + self.encoding_dim, dim_h)

        self.v_proj = nn.Linear(dim_h, dim_h)
        self.out_proj = nn.Linear(dim_h, dim_h)
        gin_nn = nn.Sequential(
            Linear_pyg(dim_h, dim_h * 2),
            nn.ReLU(),
            Linear_pyg(dim_h * 2, dim_h),
        )
        self.mpnn = pyg_nn.GINEConv(gin_nn)
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_mpnn = nn.Dropout(dropout_rate)
        self.norm_attn = nn.BatchNorm1d(dim_h)
        self.norm_mpnn = nn.BatchNorm1d(dim_h)
        self.norm_mlp = nn.BatchNorm1d(dim_h)
        self.attn_mpnn_linear = nn.Linear(dim_h * 2, dim_h)
        self.mlp = nn.Sequential(
            nn.Linear(dim_h, dim_h * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_h * 2, dim_h),
            nn.Dropout(dropout_rate),
        )

    def get_attention_logits(self, data):
        x_dense, attn_mask = to_dense_batch(
            data.x, data.batch
        )  # (batch_size, node_num, dim_h), (batch_size, node_num)

        q = self.q_proj(x_dense)
        k = self.k_proj(x_dense)
        q = q.view(
            x_dense.shape[0],
            x_dense.shape[1],
            self.num_heads,
            x_dense.shape[2] // self.num_heads,
        ).permute(0, 2, 1, 3)
        k = k.view(
            x_dense.shape[0],
            x_dense.shape[1],
            self.num_heads,
            x_dense.shape[2] // self.num_heads,
        ).permute(0, 2, 1, 3)
        causal_attention_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        conf_attention_logits = -causal_attention_logits

        extended_mask = attn_mask.unsqueeze(1)
        causal_attention_logits = torch.where(
            extended_mask.unsqueeze(-1) & extended_mask.unsqueeze(-2),
            causal_attention_logits,
            torch.tensor(float("-1e9")).to(causal_attention_logits.device),
        )
        conf_attention_logits = torch.where(
            extended_mask.unsqueeze(-1) & extended_mask.unsqueeze(-2),
            conf_attention_logits,
            torch.tensor(float("-1e9")).to(conf_attention_logits.device),
        )

        return x_dense, attn_mask, causal_attention_logits, conf_attention_logits

    def softmax(
        self, attn_mask, causal_attention_logits, conf_attention_logits, temperature
    ):
        causal_attn_weight = F.softmax(causal_attention_logits / temperature, dim=-1)
        causal_attn_entropy = compute_attn_weights_entropy(
            causal_attn_weight, attn_mask
        )
        conf_attn_weight = F.softmax(conf_attention_logits / temperature, dim=-1)
        conf_attn_entropy = compute_attn_weights_entropy(conf_attn_weight, attn_mask)

        return (
            causal_attn_weight,
            causal_attn_entropy,
            conf_attn_weight,
            conf_attn_entropy,
        )

    def forward(self, data, temperature):
        x_dense, attn_mask, causal_attention_logits, conf_attention_logits = (
            self.get_attention_logits(data)
        )

        (
            causal_attn_weight,
            causal_attn_entropy,
            conf_attn_weight,
            conf_attn_entropy,
        ) = self.softmax(
            attn_mask, causal_attention_logits, conf_attention_logits, temperature
        )

        v = self.v_proj(x_dense)
        v = v.view(
            x_dense.shape[0],
            x_dense.shape[1],
            self.num_heads,
            x_dense.shape[2] // self.num_heads,
        ).permute(0, 2, 1, 3)

        attn_causal_feature = torch.matmul(causal_attn_weight, v)
        attn_causal_feature = (
            attn_causal_feature.transpose(1, 2)
            .contiguous()
            .view(
                x_dense.shape[0],
                x_dense.shape[1],
                x_dense.shape[2],
            )
        )
        attn_causal_feature = self.norm_attn(
            self.dropout_attn(self.out_proj(attn_causal_feature[attn_mask]))
        )

        attn_conf_feature = torch.matmul(conf_attn_weight, v)
        attn_conf_feature = (
            attn_conf_feature.transpose(1, 2)
            .contiguous()
            .view(
                x_dense.shape[0],
                x_dense.shape[1],
                x_dense.shape[2],
            )
        )
        attn_conf_feature = self.norm_attn(
            self.dropout_attn(self.out_proj(attn_conf_feature[attn_mask]))
        )

        mpnn_causal_mask = torch.sigmoid(causal_attention_logits).permute(0, 2, 3, 1)
        mpnn_causal_mask = edge_attr_to_sparse(
            mpnn_causal_mask, attn_mask, data.edge_index
        )
        mpnn_conf_mask = torch.sigmoid(conf_attention_logits).permute(0, 2, 3, 1)
        mpnn_conf_mask = edge_attr_to_sparse(mpnn_conf_mask, attn_mask, data.edge_index)

        set_masks(torch.mean(mpnn_causal_mask, dim=-1), self)
        mpnn_causal_feature = self.norm_mpnn(
            self.dropout_mpnn(
                self.mpnn(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                )
            )
        )
        clear_masks(self)
        set_masks(torch.mean(mpnn_conf_mask, dim=-1), self)
        mpnn_conf_feature = self.norm_mpnn(
            self.dropout_mpnn(
                self.mpnn(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                )
            )
        )
        clear_masks(self)

        causal_feature = self.attn_mpnn_linear(
            torch.cat((attn_causal_feature, mpnn_causal_feature), dim=-1)
        )
        causal_feature = self.norm_mlp(self.mlp(causal_feature) + causal_feature)

        conf_feature = self.attn_mpnn_linear(
            torch.cat((attn_conf_feature, mpnn_conf_feature), dim=-1)
        )
        conf_feature = self.norm_mlp(self.mlp(conf_feature) + conf_feature)

        causal_encoding = torch.randn(
                data.x.shape[0], self.encoding_dim, device=mpnn_causal_feature.device
            )
        conf_encoding = torch.randn(
                data.x.shape[0], self.encoding_dim, device=mpnn_conf_feature.device
            )
        if self.use_encoding:
            set_masks(torch.mean(mpnn_causal_mask, dim=-1).detach(), self)
            for layer in self.encoding_mpnn:
                causal_encoding = layer(
                    causal_encoding, data.edge_index, data.edge_attr
                )
            clear_masks(self)
            set_masks(torch.mean(mpnn_conf_mask, dim=-1).detach(), self)
            for layer in self.encoding_mpnn:
                conf_encoding = layer(conf_encoding, data.edge_index, data.edge_attr)
            clear_masks(self)
            causal_feature = self.encoding_linear(
                torch.cat((causal_feature, causal_encoding), dim=-1)
            )
            conf_feature = self.encoding_linear(
                torch.cat((conf_feature, conf_encoding), dim=-1)
            )
            
        data.x = (causal_feature + conf_feature) / 2

        return (
            data,
            causal_feature,
            conf_feature,
            causal_attn_entropy,
            conf_attn_entropy,
            causal_encoding,
            conf_encoding,
            mpnn_causal_mask,
        )


class TransformerLayer(nn.Module):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        dim_h = config.model.dim_hidden
        dropout_rate = config.model.dropout_rate

        self.attn = torch.nn.MultiheadAttention(
            dim_h, config.model.num_heads, batch_first=True
        )
        self.mpnn = pyg_nn.GINEConv(
            nn.Sequential(
                Linear_pyg(dim_h, dim_h * 2),
                nn.ReLU(),
                Linear_pyg(dim_h * 2, dim_h),
            )
        )
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_mpnn = nn.Dropout(dropout_rate)
        self.norm_attn = nn.BatchNorm1d(dim_h)
        self.norm_mpnn = nn.BatchNorm1d(dim_h)
        self.norm_mlp = nn.BatchNorm1d(dim_h)
        self.attn_mpnn_linear = nn.Linear(dim_h * 2, dim_h)
        self.mlp = nn.Sequential(
            nn.Linear(dim_h, dim_h * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_h * 2, dim_h),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        mpnn_mask=None,
    ):
        x_dense, attn_mask = to_dense_batch(
            x, batch
        )  # (batch_size, node_num, dim_h), (batch_size, node_num)

        if mpnn_mask == None:
            attn_feature = self.norm_attn(
                self.dropout_attn(
                    self.attn(
                        x_dense,
                        x_dense,
                        x_dense,
                        key_padding_mask=~attn_mask,
                        need_weights=False,
                    )[0][attn_mask]
                )
                + x
            )

            mpnn_feature = self.norm_mpnn(
                self.dropout_mpnn(self.mpnn(x, edge_index, edge_attr)) + x
            )
        else:
            bias = to_dense_adj(
                edge_index, batch, torch.log(mpnn_mask + 1e-10)
            ).permute(0, 3, 1, 2)
            B, H, N, _ = bias.shape
            bias = bias.reshape(B * H, N, N)
            attn_feature = self.norm_attn(
                self.dropout_attn(
                    self.attn(
                        x_dense,
                        x_dense,
                        x_dense,
                        attn_mask=bias,
                        key_padding_mask=~attn_mask,
                        need_weights=False,
                    )[0][attn_mask]
                )
                + x
            )

            set_masks(torch.mean(mpnn_mask, dim=-1), self)
            mpnn_feature = self.norm_mpnn(
                self.dropout_mpnn(self.mpnn(x, edge_index, edge_attr)) + x
            )
            clear_masks(self)

        feature = self.attn_mpnn_linear(torch.cat((attn_feature, mpnn_feature), dim=-1))
        feature = self.norm_mlp(self.mlp(feature) + feature)

        return feature


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None


def edge_attr_to_sparse(edge_attr, mask, edge_index):
    num_samples = edge_attr.shape[0]
    num_nodes = edge_attr.shape[1]
    dim = edge_attr.shape[3]
    large_matrix_shape = (num_samples * num_nodes, num_samples * num_nodes, dim)
    large_matrix = torch.zeros(large_matrix_shape, device=edge_attr.device)
    for i in range(num_samples):
        offset = i * num_nodes
        small_matrix = edge_attr[i]
        large_matrix[offset : offset + num_nodes, offset : offset + num_nodes].copy_(
            small_matrix
        )
    flattened_mask = mask.view(-1)
    rows_to_keep = flattened_mask.nonzero(as_tuple=True)[0]
    matrix_reduced_rows = large_matrix[rows_to_keep]
    cols_to_keep = flattened_mask.nonzero(as_tuple=True)[0]
    matrix_reduced = matrix_reduced_rows[:, cols_to_keep, :]
    sparse_edge_attr = torch.zeros(edge_index.shape[1], device=edge_attr.device)
    sparse_edge_attr = matrix_reduced[edge_index[0, :], edge_index[1, :]]
    return sparse_edge_attr


def compute_attn_weights_entropy(attn_weights, mask, virtual_node=False):
    B, H, N, _ = attn_weights.shape
    epsilon = 1e-10
    probs = torch.clamp(attn_weights, min=epsilon, max=1.0 - epsilon)
    log_probs = torch.log(probs)
    entropy_last_dim = -probs * log_probs
    mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # B*1*1*N
    mask_expanded = mask_expanded.expand(-1, H, N, -1)  # B*H*N*N
    if virtual_node:
        mask_expanded = torch.cat(
            (
                mask_expanded,
                torch.ones(B, H, N, 1, dtype=torch.bool, device=mask_expanded.device),
            ),
            dim=-1,
        )
    entropy_last_dim = (entropy_last_dim * mask_expanded).sum(dim=-1)  # B*H*N
    mean_entropy = torch.sum(
        entropy_last_dim * mask.unsqueeze(1).expand(-1, H, -1), dim=-1
    ) / mask.sum(dim=-1, keepdim=True).expand(
        -1, H
    )  # B*H
    return mean_entropy
