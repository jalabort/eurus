import torch
import torch.nn as nn
import torch.nn.functional as F

from eurus.track.pytorch.train.model.base import TrackingModel


class ForwardTrackingModel(TrackingModel):
    r"""
    """
    def __init__(self):
        super(ForwardTrackingModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)

        self.bn11 = nn.BatchNorm2d(3)
        self.bn12 = nn.BatchNorm2d(16)
        self.bn13 = nn.BatchNorm2d(16)
        self.bn14 = nn.BatchNorm2d(16)
        self.bn15 = nn.BatchNorm2d(16)

        self.bn21 = nn.BatchNorm2d(3)
        self.bn22 = nn.BatchNorm2d(16)
        self.bn23 = nn.BatchNorm2d(16)
        self.bn24 = nn.BatchNorm2d(16)
        self.bn25 = nn.BatchNorm2d(16)

        self.bn1 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(3712, 512)
        self.fc2 = nn.Linear(512, 1089)

    def forward(self, x1, x2):
        r"""

        Returns
        -------
        response: torch.autograd.variable.Variable
        """
        x1 = self.bn11(x1)
        x1 = F.max_pool2d(F.relu(self.bn12(self.conv1(x1))), 2)
        x1 = F.max_pool2d(F.relu(self.bn13(self.conv2(x1))), 2)
        x1 = F.max_pool2d(F.relu(self.bn14(self.conv3(x1))), 2)
        x1 = F.max_pool2d(F.relu(self.bn15(self.conv4(x1))), 2)

        x1 = x1.view(-1, 16 * 6 * 6)

        x2 = self.bn21(x2)
        x2 = F.max_pool2d(F.relu(self.bn22(self.conv1(x2))), 2)
        x2 = F.max_pool2d(F.relu(self.bn23(self.conv2(x2))), 2)
        x2 = F.max_pool2d(F.relu(self.bn24(self.conv3(x2))), 2)
        x2 = F.max_pool2d(F.relu(self.bn25(self.conv4(x2))), 2)

        x2 = x2.view(-1, 16 * 14 * 14)

        x = torch.cat((x1, x2), 1)

        y = F.relu(self.bn1(self.fc1(x)))
        y = self.fc2(y)

        y = y.view(-1, 1, 33, 33)

        return y
