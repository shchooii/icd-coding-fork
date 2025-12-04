# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional

from transformers import RobertaModel, AutoConfig

from src.models.modules.attention import LabelAttention

from src.losses.estimator import EstimatorCV
from src.losses.resample2 import ResampleLoss



class PLMICD3(nn.Module):
    def __init__(self, num_classes: int, model_path: str,
                 cls_num_list = None, 
                 head_idx = None, tail_idx = None,
                 co_occurrence_matrix = None,
                 class_freq = None, neg_class_freq = None,
                 **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )
        
        self.roberta = RobertaModel(
            self.config, add_pooling_layer=False
        ).from_pretrained(model_path, config=self.config)
        
        self.attention = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=num_classes,
        )
        
        self.estimator = EstimatorCV(self.config.hidden_size, num_classes) # 추가
            
        # self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        self.slploss = ResampleLoss(
            use_sigmoid=True,
            class_freq=class_freq,           
            neg_class_freq=neg_class_freq,
        )

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask) # 변경
        with torch.no_grad():
            Prop, Cov_pos, Cov_neg, Sigma_cj, Ro_cj, Tao_cj = self.estimator.update_CV(targets, logits)
        loss = self.slploss(
                    norm_prop=Prop,
                    nonzero_var_tensor=Cov_pos,
                    zero_var_tensor=Cov_neg,
                    normalized_sigma_cj=Sigma_cj,
                    normalized_ro_cj=Ro_cj,
                    normalized_tao_cj=Tao_cj,
                    cls_score=logits,
                    label=targets
                )
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}


    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        dev = logits.device
        
        with torch.no_grad():
            Prop    = self.estimator.Prop.to(dev)
            Cov_pos = self.estimator.Cov_pos.to(dev)
            Cov_neg = self.estimator.Cov_neg.to(dev)
            Sigma_cj = self.estimator.Sigma_cj
            Ro_cj = self.estimator.Ro_cj
            Tao_cj = self.estimator.Tao_cj

        loss = self.slploss(
            norm_prop=Prop,
            nonzero_var_tensor=Cov_pos,
            zero_var_tensor=Cov_neg,
            normalized_sigma_cj=Sigma_cj,
            normalized_ro_cj=Ro_cj,
            normalized_tao_cj=Tao_cj,
            cls_score=logits,
            label=targets
        )
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size)
            if attention_mask is not None
            else None,
            return_dict=False,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        logits = self.attention(hidden_output)
        return logits
