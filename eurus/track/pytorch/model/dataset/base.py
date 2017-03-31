import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import multivariate_normal

from PIL import Image
from torch.utils.data import Dataset

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

    context_size : int, optional

    search_size : int, optional


    References
    ----------
    M. Mueller, et al. "A Benchmark and Simulator for UAV Tracking". ECCV 2016.
    """
    def __init__(self, root, transform=None, target_transform=None,
                 sequence_length=None, skip=None, context_factor=3,
                 search_factor=2, context_size=128, search_size=256):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length
        if skip is None:
            self.skip = 1

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
            img1 = Image.open(img_sequence[i])
            if img1.mode == 'L':
                img1 = img1.convert(mode='RGB')
            img1_box = ann_sequence[i]

            img1_tl = img1_box[:2]
            img1_sz = img1_box[2:]
            img1_center = img1_tl + img1_sz / 2

            max_sz = max(img1_sz)
            img1_crop_size = np.array([max_sz, max_sz]) * self.context_factor

            img1_center += np.random.randn(2) * 0.01 * img1_crop_size
            img1_crop_size += np.random.randn(2) * 0.05 * img1_crop_size

            context = crop(img1, img1_center, img1_crop_size)
            context = context.resize(self.context_size, resample=Image.BICUBIC)

            ratio1 = img1_crop_size / self.context_size
            corrector = img1_center - img1_crop_size / 2
            context_tl = (img1_box[:2] - corrector) / ratio1
            context_sz = img1_box[2:] / ratio1
            context_box = np.concatenate([context_tl, context_sz])

            img2 = Image.open(img_sequence[i + 1])
            if img2.mode == 'L':
                img2 = img1.convert(mode='RGB')
            img2_box = ann_sequence[i + 1]

            img2_crop_size = img1_crop_size * self.search_factor

            search = crop(img2, img1_center, img2_crop_size)
            search = search.resize(self.search_size, resample=Image.BICUBIC)

            ratio2 = img2_crop_size / self.search_size

            corrector = img1_center - img2_crop_size / 2
            search_tl = (img2_box[:2] - corrector) / ratio2
            search_sz = img2_box[2:] / ratio2
            search_box = np.concatenate([search_tl, search_sz])

            aux_tl = (img1_box[:2] - corrector) / ratio2
            aux_sz = img1_box[2:] / ratio2
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
                score_map = self.target_transform(score_map)

            yield (context, search,
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
