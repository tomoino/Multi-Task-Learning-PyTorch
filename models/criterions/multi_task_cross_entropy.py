# -*- coding: utf-8 -*-
"""Multi Task Cross Entropy Loss"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseModel


class MultiTaskCrossEntropyLoss(nn.Module):
    """Multi Task Cross Entropy Loss"""


    def __init__(self, num_train_task: int) -> None:
        """Initialization

        Args:
            num_train_task: Number of train tasks.

        """

        super().__init__()

        tasks = range(num_train_task)
        loss_ft = torch.nn.ModuleDict({task: nn.CrossEntropyLoss() for task in tasks})

        self.tasks = tasks
        self.loss_ft = loss_ft


    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([out[t] for t in self.tasks]))

        return out