from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg
from collections import OrderedDict


@register.ood_alg_register
class GOODFormerAlg(BaseOODAlg):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GOODFormerAlg, self).__init__(config)
        self.alpha = 0
        self.beta = 0
        self.delta = 0

        self.rep_out = None
        self.causal_out = None
        self.conf_out = None
        self.causal_attention_entropy = None
        self.conf_attention_entropy = None
        self.causal_encoding_out = None
        self.conf_encoding_out = None
        self.origin_encoding = None

    def stage_control(self, config: Union[CommonArgs, Munch]):
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1
        self.alpha = config.ood.ood_param
        self.beta = config.ood.extra_param[0] / (config.train.epoch + 1)
        self.delta = config.ood.extra_param[1]

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        if isinstance(model_output, tuple):
            (
                self.rep_out,
                self.causal_out,
                self.conf_out,
                self.causal_attention_entropy,
                self.conf_attention_entropy,
                self.causal_encoding_out,
                self.conf_encoding_out,
                self.origin_encoding
            ) = model_output
        else:
            self.causal_out = model_output
            (
                self.rep_out,
                self.conf_out,
                self.causal_attention_entropy,
                self.conf_attention_entropy,
                self.causal_encoding_out,
                self.conf_encoding_out,
                self.origin_encoding
            ) = (None, None, None, None, None, None, None)
        return self.causal_out

    def loss_calculate(
        self,
        raw_pred: Tensor,
        targets: Tensor,
        mask: Tensor,
        node_norm: Tensor,
        config: Union[CommonArgs, Munch],
    ) -> Tensor:
        if self.rep_out is not None:
            self.spec_loss = OrderedDict()
            causal_loss = (
                config.metric.loss_func(raw_pred, targets, reduction="none") * mask
            ).sum() / mask.sum()
            conf_loss = (
                config.metric.loss_func(self.conf_out, targets, reduction="none") * mask
            ).sum() / mask.sum()

            env_loss = torch.tensor([]).to(config.device)

            for rep in self.rep_out:
                tmp = config.metric.loss_func(rep, targets, reduction="none")
                env_loss = torch.cat([env_loss, tmp.unsqueeze(0)])

            self.spec_loss["rep_loss"] = 0.1 * env_loss.mean()
            causal_loss += self.spec_loss["rep_loss"]
            env_loss = self.alpha * ((torch.var(env_loss, dim=0) * mask).sum()) / mask.sum()

            if self.causal_attention_entropy != None and self.conf_attention_entropy != None:
                entropy_loss = self.beta * (
                    torch.mean(self.causal_attention_entropy)
                    + torch.mean(self.conf_attention_entropy)
                )
            else:
                entropy_loss = 0

            if torch.isnan(self.origin_encoding).any():
                encoding_loss = 0
            else:
                encoding_loss = F.l1_loss(self.causal_encoding_out, self.origin_encoding) + F.l1_loss(self.conf_encoding_out, self.origin_encoding)
                encoding_loss = self.delta * encoding_loss

            loss = causal_loss + env_loss + conf_loss + entropy_loss + encoding_loss
            self.mean_loss = causal_loss
            self.spec_loss["env_loss"] = env_loss
            self.spec_loss["conf_loss"] = conf_loss
            self.spec_loss["entropy_loss"] = entropy_loss
            self.spec_loss["encoding_loss"] = encoding_loss
        else:
            causal_loss = (
                config.metric.loss_func(raw_pred, targets, reduction="none") * mask
            ).sum() / mask.sum()

            loss = causal_loss
            self.mean_loss = causal_loss

        return loss

    def loss_postprocess(
        self,
        loss: Tensor,
        data: Batch,
        mask: Tensor,
        config: Union[CommonArgs, Munch],
        **kwargs,
    ) -> Tensor:
        return loss
