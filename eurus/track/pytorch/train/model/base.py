from abc import ABCMeta

import torch.nn as nn


class TrackingModel(nn.Module, metaclass=ABCMeta):
    r"""
    """
    def __init__(self):
        super(TrackingModel, self).__init__()
