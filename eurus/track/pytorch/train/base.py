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
from .utils import AverageMeter


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
        loss_meter, time_meter = training_loop(
            epoch, dataloader, model, criterion, optimizer, crayon_logger)

        logger.info(
            'Epoch: {0:05d} completed \t'
            'Average Loss: {1:4.4f} \t'
            'Total time: {2:4.4f}'.format(
                epoch,
                loss_meter.avg,
                time_meter.sum))

        if crayon_logger is not None:
            crayon_logger.add_scalar_value('loss_epochs', loss_meter.avg)
            crayon_logger.add_scalar_value('time_epochs', time_meter.sum)

        torch.save(model.state_dict(), config.weights_file)


def training_loop(epoch, dataloader, model, criterion, optimizer,
                  crayon_logger, logging_frequency=10):
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
    crayon_logger :
        
    logging_frequency : int, optional
        
    Returns
    -------
    meters: (:class:`eurus.track.pytorch.train.AverageMeter`, 
             :class:`eurus.track.pytorch.train.AverageMeter`)
    """
    loss_meter = AverageMeter()
    time_meter = AverageMeter()

    model.train()

    end = time.time()
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()

        context, search, _, search_box, context_box, response = data

        context = Variable(context)
        search = Variable(search)
        response = Variable(response)

        if torch.cuda.is_available():
            context = context.cuda()
            search = search.cuda()
            response = response.cuda()

        output_response = model(context, search)
        loss = criterion(output_response, response)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.data[0])
        time_meter.update(time.time() - end)
        end = time.time()

        if i % logging_frequency == 0:
            logger.info('Epoch: {0:05d} [{1:05d}/{2:05d} ({3:2.0f}%)]\t'
                        'Loss: {4:5.4f} [{5:5.4f}]\t'
                        'Time: {6:5.4f} [{7:5.4f}]'.format(
                            epoch,
                            i * context.size()[0],
                            len(dataloader.dataset),
                            100. * i / len(dataloader),
                            loss_meter.val,
                            loss_meter.avg,
                            time_meter.val,
                            time_meter.avg))

            if crayon_logger is not None:
                crayon_logger.add_scalar_value('loss_iters', loss_meter.val)
                crayon_logger.add_scalar_value('time_iters', time_meter.val)

    return loss_meter, time_meter
