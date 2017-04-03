import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pycrayon import CrayonClient

from .dataset import create_dataset
from .model import create_tracking_model


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_tracking_model(config):
    r"""
    Training function for :class:`eurus.track.pytorch.train.TrackingModel`.

    Parameters
    ----------
    config : :class:`eurus.track.pytorch.train.TrackingModelConfig`
        The configuration to run the training.
    """
    if config.crayon_config is not None:
        crayon_config = config.crayon_config
        cc = CrayonClient(hostname=crayon_config.server_address)
        crayon_logger = cc.create_experiment(crayon_config.experiment_name)
    else:
        crayon_logger = None

    dataset = create_dataset(config.dataset_config)
    dataloader = DataLoader(dataset, **config.dataloader_config.configuration)

    model = create_tracking_model(config.tracking_model_config)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # TODO: Optimizer config?
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(1, config.n_epochs + 1):
        training_loop(epoch, dataloader, model, criterion, optimizer,
                      crayon_logger)

        logger.info(
            'Epoch: {0:05d} completed'.format(epoch))

        torch.save(model.state_dict(), config.weights_file)


def training_loop(epoch, dataloader, model, criterion, optimizer,
                  crayon_logger):
    r"""
    Training loop for :class:`eurus.track.pytorch.train.TrackingModel`.

    Parameters
    ----------
    epoch : int
        The epoch number.
    dataloader: :class:`torch.utils.data.DataLoader`.
        The data loader.
    model: :class:`eurus.track.pytorch.train.TackingModel`.
        The tracking model.
    optimizer :
        The optimizer.
    criterion :
        The loss function.
    """
    model.train()

    for i, data_sequence in enumerate(dataloader):

        for j, data in enumerate(data_sequence):
            optimizer.zero_grad()

            x1, x2, _, search_box, context_box, t = data

            x1 = Variable(x1)
            x2 = Variable(x2)
            t = Variable(t)

            if torch.cuda.is_available():
                x1 = x1.cuda()
                x2 = x2.cuda()
                t = t.cuda()

            y = model(x1, x2)
            loss = criterion(y, t)

            loss.backward()
            optimizer.step()

        logger.info('Epoch: {0:05d} [{1:06d}/{2:05d} ({3:2.0f}%)]\t'
                    'Loss: {4:4.4f}'.format(
                        epoch,
                        i * x1.size()[0],
                        len(dataloader.dataset),
                        100. * i / len(dataloader),
                        loss.data[0]))

        if crayon_logger is not None:
            crayon_logger.add_scalar_value('loss', loss.data[0])
