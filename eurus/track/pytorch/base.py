import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms

from eurus.utils import Box
from eurus.track.base import Tracker

from .model.base import HudlNet
from .model.dataset.utils import crop


class HudlTracker(Tracker):
    r"""
    Tracker class based on Deep Neural Networks.
    
    Parameters
    ----------
    resume :
        
    """
    def __init__(self, resume):
        self.model = HudlNet()

        state_dict = torch.load(resume)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.cuda()

        # TODO: Find a better way to set this up. They need to be the same
        # as the one used to train the model
        self.grid = np.mgrid[:256, :256]
        self.context_factor = 3
        self.search_factor = 2
        self.context_size = (128, 128)
        self.search_size = (256, 256)

        self.box = None
        self.context_center = None
        self.search_crop_size = None
        self.context = None
        self.search_ratio = None
        self.context_state = None
        self.search_state = None

    def initialize(self, context, box):
        r"""
        
        
        Parameters
        ----------
        context :
        
        box :

        """
        # context = Image.fromarray(np.uint8(context[..., ::-1]) * 255)

        self.box = box.to_numpy()

        tl = self.box[:2]
        br = tl + self.box[2:]
        self.context_center = tl + self.box[2:] / 2

        size = np.ceil(br - tl)
        max_size = np.array([max(size)] * 2)
        context_crop_size = max_size * self.context_factor
        self.search_crop_size = context_crop_size * self.search_factor

        context = crop(context, self.context_center, context_crop_size)
        context = context.resize(self.context_size, resample=Image.BICUBIC)
        context = transforms.ToTensor()(context)
        context = Variable(context, volatile=True)[None]
        self.context = context.cuda()

        self.search_ratio = self.search_crop_size / self.search_size

        # TODO: This should be grabbed from the model.
        self.context_state = (
            Variable(torch.zeros(1, 256), volatile=True).cuda(),
            Variable(torch.zeros(1, 256), volatile=True).cuda())
        self.search_state = (
            Variable(torch.zeros(1, 256), volatile=True).cuda(),
            Variable(torch.zeros(1, 256), volatile=True).cuda())

    def track(self, img, current_time):
        r"""
        
        
        Parameters
        ----------
        img : 
        
        current_time :
        

        Returns
        -------

        """
        # search = Image.fromarray(np.uint8(search[..., ::-1]) * 255)

        search = crop(img, self.context_center, self.search_crop_size)
        search = search.resize(self.search_size, resample=Image.BICUBIC)
        search = transforms.ToTensor()(search)
        search = Variable(search, volatile=True)[None]
        search = search.cuda()

        (_, response,
         self.context_state,
         self.search_state) = self.model.forward(self.context, search,
                                                 self.context_state,
                                                 self.search_state)

        response = response.squeeze().cpu().data.numpy()
        peak = np.array(np.unravel_index(np.argmax(response), response.shape))

        # scaled_peak = self.model.module.stride * peak

        offset = peak - self.context_size
        self.context_center = self.context_center + offset[::-1] * self.search_ratio

        self.box[:2] = self.context_center - self.box[2:] / 2

        size = self.box[2:]
        max_size = np.array([max(size)] * 2)
        context_crop_size = max_size * self.context_factor
        self.search_crop_size = context_crop_size * self.search_factor

        context = crop(img, self.context_center, context_crop_size)
        context = context.resize(self.context_size, resample=Image.BICUBIC)
        context = transforms.ToTensor()(context)
        context = Variable(context, volatile=True).unsqueeze(0)
        self.context = context.cuda()

        self.search_ratio = self.search_crop_size / self.search_size

        return self.context_center, response
