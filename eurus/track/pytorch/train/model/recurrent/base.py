import torch
import torch.nn as nn
import torch.nn.functional as F

from eurus.track.pytorch.train.model.base import TrackingModel


class RecurrentTrackingModel(TrackingModel):
    r"""
    """
    def __init__(self):
        super(RecurrentTrackingModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn21 = nn.BatchNorm2d(16)
        self.bn22 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3)
        self.bn31 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, 3)
        self.bn41 = nn.BatchNorm2d(16)
        self.bn42 = nn.BatchNorm2d(16)

        self.rnn1 = nn.LSTMCell(576, 256)
        self.rnn2 = nn.LSTMCell(3136, 256)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 4)
        self.fc4 = nn.Linear(128, 65536)

    def forward(self, x1, x2, s1, s2):
        r"""
        """
        x1 = F.max_pool2d(F.relu(self.bn11(self.conv1(x1))), 2)
        x1 = F.max_pool2d(F.relu(self.bn21(self.conv2(x1))), 2)
        x1 = F.max_pool2d(F.relu(self.bn31(self.conv3(x1))), 2)
        x1 = F.max_pool2d(F.relu(self.bn41(self.conv4(x1))), 2)
        x1 = x1.view(-1, 16 * 6 * 6)

        h1, c1 = self.rnn1(x1, s1)

        x2 = F.max_pool2d(F.relu(self.bn12(self.conv1(x2))), 2)
        x2 = F.max_pool2d(F.relu(self.bn22(self.conv2(x2))), 2)
        x2 = F.max_pool2d(F.relu(self.bn32(self.conv3(x2))), 2)
        x2 = F.max_pool2d(F.relu(self.bn42(self.conv4(x2))), 2)
        x2 = x2.view(-1, 16 * 14 * 14)

        h2, c2 = self.rnn2(x2, s2)

        x = torch.cat((h1, h2), 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        y1 = self.fc3(x)

        y2 = self.fc4(x)
        y2 = y2.view(-1, 1, 256, 256)

        return y1, y2, (h1, c1), (h2, c2)
