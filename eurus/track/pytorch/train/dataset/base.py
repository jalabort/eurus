import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import multivariate_normal

from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

from .utils import crop, display_1d_histogram
from .widgets.exception import IPythonWidgetsMissingError


class TrackingDataset(Dataset, metaclass=ABCMeta):
    r"""
    Base class for tracking datasets.

    Parameters
    ----------
    root : str
        The root path to the dataset.
    transform :

    target_transform :

    context_factor : float, optional

    search_factor : float, optional

    context_size : int | (int, int), optional

    search_size : int | (int, int), optional

    response_size : int | (int, int), optional
        
    """
    def __init__(self, root, transform=None, target_transform=None,
                 context_factor=3, search_factor=2, context_size=128,
                 search_size=256, response_size=33):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.context_factor = context_factor
        self.search_factor = search_factor

        if isinstance(context_size, numbers.Number):
            self.context_size = np.array([context_size, context_size]).astype(
                np.int32)
        elif isinstance(context_size, tuple):
            self.context_size = np.array(context_size)
        else:
            raise ValueError('`context_size` must be `int` or `(int, int)`.')

        if isinstance(search_size, numbers.Number):
            self.search_size = np.array([search_size, search_size]).astype(
                np.int32)
        elif isinstance(search_size, tuple):
            self.search_size = np.array(search_size).astype(np.int32)
        else:
            raise ValueError('`context_size` must be `int` or `(int, int)`.')

        if isinstance(response_size, numbers.Number):
            self.response_size = np.array([response_size,
                                           response_size]).astype(np.int32)
        elif isinstance(response_size, tuple):
            self.response_size = np.array(response_size).astype(np.int32)
        else:
            raise ValueError('`response_size` must be `int` or `(int, int)`.')

    @property
    @abstractmethod
    def _n_elements_per_sequence(self):
        raise NotImplemented

    @property
    def n_shortest(self):
        return np.min(self._n_elements_per_sequence)

    @property
    def n_longest(self):
        return np.max(self._n_elements_per_sequence)

    @property
    def n_average(self):
        return np.mean(self._n_elements_per_sequence)

    @property
    def n_median(self):
        return np.median(self._n_elements_per_sequence)

    @abstractmethod
    def __len__(self):
        raise NotImplemented

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplemented

    def _create_training_pair(self, img0, img1, img0_box, img1_box):
        r"""
        
        
        Parameters
        ----------
        img0 : 
        img1 :
        img0_box :
        img1_box :
 
        Returns
        -------
        training_data : 
        
        """
        if img0.mode == 'L':
            img0 = img0.convert(mode='RGB')

        img0_tl = img0_box[:2]
        img0_sz = img0_box[2:]
        img0_center = img0_tl + img0_sz / 2

        max_sz = max(img0_sz)
        img0_crop_size = np.array([max_sz, max_sz]) * self.context_factor

        img0_center += np.random.randn(2) * 0.01 * img0_crop_size
        img0_crop_size += np.random.randn(2) * 0.05 * img0_crop_size

        context = crop(img0, img0_center, img0_crop_size)
        context = context.resize(self.context_size, resample=Image.BICUBIC)

        ratio0 = img0_crop_size / self.context_size
        corrector = img0_center - img0_crop_size / 2
        context_tl = (img0_box[:2] - corrector) / ratio0
        context_sz = img0_box[2:] / ratio0
        context_box = np.concatenate([context_tl, context_sz])

        if img1.mode == 'L':
            img1 = img0.convert(mode='RGB')

        img1_crop_size = img0_crop_size * self.search_factor

        search = crop(img1, img0_center, img1_crop_size)
        search = search.resize(self.search_size, resample=Image.BICUBIC)

        ratio1 = img1_crop_size / self.search_size

        corrector = img0_center - img1_crop_size / 2
        search_tl = (img1_box[:2] - corrector) / ratio1
        search_sz = img1_box[2:] / ratio1
        search_box = np.concatenate([search_tl, search_sz])

        aux_tl = (img0_box[:2] - corrector) / ratio1
        aux_sz = img0_box[2:] / ratio1
        search_context_box = np.concatenate([aux_tl, aux_sz])

        search_tr = search_tl + np.array([search_sz[0], 0])
        search_br = search_tl + np.array([0, search_sz[1]])
        search_bl = search_tl + search_sz
        box_matrix = np.concatenate([[search_tl], [search_tr],
                                     [search_br], [search_bl]])

        mean = search_tl + search_sz / 2
        cov = np.cov(box_matrix, rowvar=False)

        x, y = np.mgrid[:search.size[1], :search.size[0]]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = y
        pos[:, :, 1] = x
        rv = multivariate_normal(mean, cov)
        score_map = rv.pdf(pos)
        max_score = np.max(score_map)
        min_score = np.min(score_map)
        score_map = 255 * (score_map - min_score) / (max_score - min_score)
        score_map = transforms.ToPILImage()(
            score_map[..., None].astype(np.int32))
        score_map = score_map.resize(self.response_size,
                                     resample=Image.BICUBIC)
        score_map = np.array(score_map)

        context_box = context_box.astype(np.float32)
        search_box = search_box.astype(np.float32)
        search_context_box = search_context_box.astype(np.float32)
        score_map = score_map.astype(np.float32)

        if self.transform is not None:
            context = self.transform(context)
            search = self.transform(search)
        if self.target_transform is not None:
            context_box = self.target_transform(context_box)
            search_box = self.target_transform(search_box)
            search_context_box = self.target_transform(search_context_box)
            score_map = self.transform(score_map)

        return (context, search,
                context_box, search_box,
                search_context_box, score_map)

    def view_original(self):
        r"""
        Visualize the original unprocessed dataset.
        """
        try:
            from .widgets import notebook_view_sequence
            notebook_view_sequence(self.img_list, self.ann_list)
        except ImportError:
            raise IPythonWidgetsMissingError()

    def view_sequence_length_histogram(self):
        r"""
        Visualize the histogram of sequence lengths.
        """
        display_1d_histogram(self._n_elements_per_sequence)

    def __str__(self):
        return ('{} dataset:\n'
                '  - Number of sequences: {}\n'
                '    - Shortest sequence length: {}\n'
                '    - Longest sequence length:  {}\n'
                '    - Average sequence length:  {}\n'
                '    - Median sequence length:   {}\n'
                .format(self.__class__.__name__,
                        len(self),
                        self.n_shortest,
                        self.n_longest,
                        self.n_average,
                        self.n_median))


class PairDataset(TrackingDataset, metaclass=ABCMeta):
    r"""
    Base class for pair datasets.

    Parameters
    ----------
    root : str
        The root path to the dataset.
    transform :

    target_transform :

    context_factor : float, optional

    search_factor : float, optional

    context_size : int | (int, int), optional

    search_size : int | (int, int), optional

    response_size : int | (int, int), optional


    References
    ----------
    M. Mueller, et al. "A Benchmark and Simulator for UAV Tracking". ECCV 2016.
    """

    def __init__(self, root, transform=None, target_transform=None,
                 offset=10, context_factor=3, search_factor=2,
                 context_size=128, search_size=256, response_size=33):

        super(PairDataset, self).__init__(
            root, transform=transform, target_transform=target_transform,
            context_factor=context_factor, search_factor=search_factor,
            context_size=context_size, search_size=search_size,
            response_size=response_size)

        self.offset = offset

    @abstractmethod
    def _get_pair_from_index(self, index):
        r"""
        Get a pair of images and annotations from the same sequence.

        Parameters
        ----------
        index : int
            The index from which to retrieve the image and annotations pairs.

        Returns
        -------
        pair : ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray))

        """
        raise NotImplemented

    def __getitem__(self, index):
        img_pair, ann_pair = self._get_pair_from_index(index)

        return self._create_training_pair(
            Image.open(img_pair[0]),
            Image.open(img_pair[1]),
            ann_pair[0],
            ann_pair[1]
        )


class SequenceDataset(TrackingDataset, metaclass=ABCMeta):
    r"""
    Base class for tracking datasets.

    Parameters
    ----------
    root : str
        The root path to the dataset.
    transform :

    target_transform :

    context_factor : float, optional

    search_factor : float, optional

    context_size : int | (int, int), optional

    search_size : int | (int, int), optional

    response_size : int | (int, int), optional
    

    References
    ----------
    M. Mueller, et al. "A Benchmark and Simulator for UAV Tracking". ECCV 2016.
    """
    def __init__(self, root, transform=None, target_transform=None,
                 sequence_length=None, skip=None, context_factor=3,
                 search_factor=2, context_size=128, search_size=256,
                 response_size=33):

        super(SequenceDataset, self).__init__(
            root, transform=transform, target_transform=target_transform,
            context_factor=context_factor, search_factor=search_factor,
            context_size=context_size, search_size=search_size,
            response_size=response_size)

        self.sequence_length = sequence_length
        if skip is None:
            self.skip = 1

    @abstractmethod
    def _get_sequence_from_index(self, index):
        r"""


        Parameters
        ----------
        index :


        Returns
        -------
        sequences : (list[np.ndarray], list[np.ndarray])

        """
        raise NotImplemented

    def __getitem__(self, index):
        img_sequence, ann_sequence = self._get_sequence_from_index(index)

        for i in range(len(img_sequence) - 1):
            yield self._create_training_pair(
                Image.open(img_sequence[i]),
                Image.open(img_sequence[i + 1]),
                ann_sequence[i],
                ann_sequence[i + 1]
            )
