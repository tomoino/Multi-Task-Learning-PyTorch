# -*- coding: utf-8 -*-
"""Multi Task Model"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseModel


class Backbone(nn.Module):
    """Backbone for SimpleCNN"""


    def __init__(self, in_channel, out_channel) -> None:
        """Initialization

        Args:
            in_channel: Channel of input.
            out_channel: Channel of output.

        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.out_channel = out_channel


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, out_channel)

        return x


class Head(nn.Module):
    """Head for SimpleCNN"""


    def __init__(self, in_channel, out_channel) -> None:
        """Initialization

        Args:
            in_channel: Channel of input.
            out_channel: Channel of output.

        """

        super().__init__()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channel)

        self.out_channel = out_channel


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Net(nn.Module):
    """Net for Multi task learning using SimpleCNN"""


    def __init__(self, cfg) -> None:
        """Initialization

        Args:
            cfg: Config.

        """

        super().__init__()

        in_channel = self.cfg.data.dataset.in_channel
        middle_channel = 16 * 5 * 5
        out_channel = self.cfg.data.dataset.num_way
        
        self.backbone = Backbone(in_channel=in_channel, out_channel=middle_channel)
        
        self.tasks = range(cfg.data.dataset.num_train_task)
        self.decoders = torch.nn.ModuleDict(
            {task: Head(in_channel=middle_channel, out_channel=out_channel) for task in self.tasks})


    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        return {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}


class MultiTaskModel(BaseModel):
    """Multi Task Model
    
    This model is for multi task learning.
    This model has shared encoder and task-specific decoders

    """


    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Build model.

        Args:
            cfg: Config.

        """

        super().__init__(cfg)

        self.network = Net(cfg)

        self.build()