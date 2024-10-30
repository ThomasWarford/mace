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

    @abstractmethod
    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}, reg_weight={self.reg_weight:.3f})"


class L2PairwiseRegularization(Regularization):
    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        sum_of_squares = 0.0

        for readout in model.readouts:
            final_weights = list(readout.modules())[-2].weight_view_for_instruction(0)
            pairwise_differences = final_weights[:, None] - final_weights[:, :, None]
            sum_of_squares += torch.pow(pairwise_differences, 2).sum()

        return self.reg_weight * sum_of_squares
