###########################################################################################
# Implementation of different typpes of regularization
# Authors: Thomas Warford
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod

import torch


class Regularization(torch.nn.Module):
    def __init__(self, reg_weight=1.0) -> None:
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        return self.reg_weight * self.compute_regularization(model)

    @abstractmethod
    def compute_regularization(self, model: torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}, reg_weight={self.reg_weight:.3f})"


class L2PairwiseRegularization(Regularization):
    def compute_regularization(self, model: torch.nn.Module) -> torch.Tensor:
        head_embs = model.head_embedding.linear.weight_view_for_instruction(0) # len(heads), head_emb_dim
        pairwise_differences = head_embs[:, None, :] - head_embs[None, :, :]
        return torch.sum(torch.pow(pairwise_differences, 2))

class L2Regularization(Regularization):
    def compute_regularization(self, model: torch.nn.Module) -> torch.Tensor:
        head_embs = model.head_embedding.linear.weight_view_for_instruction(0) # len(heads), head_emb_dim
        return torch.sum(torch.pow(head_embs, 2))