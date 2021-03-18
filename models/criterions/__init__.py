# -*- coding: utf-8 -*-
"""Criterions"""

import torch.nn as nn

from configs.supported_info import SUPPORTED_CRITERION
from models.criterions.multi_task_cross_entropy import MultiTaskCrossEntropyLoss

def get_criterion(cfg: object) -> object:
    """Get criterion function

    This is function to get criterion.

    Args:
        cfg: Config of criterion.

    Returns:
        Criterion object.

    Raises:
        NotImplementedError: If the criterion you want to use is not suppoeted.

    """
    
    criterion_name = cfg.name

    if not criterion_name:
        return None

    if criterion_name not in SUPPORTED_CRITERION:
        raise NotImplementedError('The loss function is not supported.')

    if criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()
        
    if criterion_name == "multi_task_cross_entropy":
        return MultiTaskCrossEntropyLoss(cfg.data.dataset.num_train_task)