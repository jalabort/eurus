import os
import csv
import numbers

import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from .utils import crop, display_1d_histogram
from .widgets.exception import IPythonWidgetsMissingError


class Uav123(Dataset):
    r"""
    Class for the Unmanned Aerial Vehicles (ALOV) 123 dataset.

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
                 sequence_length=10, skip=25, context_factor=3,
                 search_factor=2, context_size=128, search_size=256):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length
        self.skip = skip

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

        img_path = os.path.join(root, 'data_seq', 'UAV123')
        ann_path = os.path.join(root, 'anno', 'UAV123')

        sequences = [seq[:-4] for seq in sorted(next(os.walk(ann_path))[2])]

        self.ann_list = []
        self.img_list = []

        for seq in sequences:
            if '_' in seq and seq.split('_')[1][0] != 's':
                continue

            with open(os.path.join(ann_path, seq + '.txt'), "r") as f:
                truths = list(csv.reader(f, delimiter=','))
            ann_list2 = []
            for t in truths:
                tl = np.array([float(t[0]), float(t[1])])
                sz = np.array([float(t[2]), float(t[3])])
                ann_list2.append(np.concatenate([tl, sz]))
            self.ann_list.append(ann_list2)
            n_ann = len(ann_list2)

            img_path2, _, files = next(os.walk(os.path.join(img_path, seq)))
            files = sorted(files)
            img_list2 = []
            for file in files[:n_ann]:
                img_list2.append(os.path.join(img_path2, file))
            self.img_list.append(img_list2)

        assert len(self.img_list) == len(self.ann_list), \
            'The number of image ({}) and ann ({}) sequences should be ' \
            'the same.'.format(len(self.img_list), len(self.ann_list))
        for i, (img_list2, ann_list2) in enumerate(zip(self.img_list,
                                                       self.ann_list)):
            assert len(img_list2) == len(ann_list2), \
                'The number of image ({}) and annotations ({}) ' \
                'in sequences {} should be the same.'.format(
                    len(img_list2), len(ann_list2), i)

    @property
    def _n_elements_per_sequence(self):
        return [len(sequence) for sequence in self.img_list]

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

    def __len__(self):
        return len(self.img_list)

    def _get_sequence_from_index(self, index):
        r"""


        Parameters
        ----------
        index :


        Returns
        -------
        sequences : (list[np.ndarray], list[np.ndarray])

        """
        if index < 0 or index > len(self):
            raise ValueError('The requested `index`, {}, is not valid. '
                             'Valid indices go from 0 to {}. '
                             .format(index, len(self)))

        img_sequence = self.img_list[index]
        ann_sequence = self.ann_list[index]

        first = np.random.randint(
            0, len(img_sequence) - self.sequence_length * self.skip)
        last = first + self.sequence_length * self.skip

        return (img_sequence[first:last:self.skip],
                ann_sequence[first:last:self.skip])

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

            if self.transform is not None:
                context = self.transform(context)
                search = self.transform(search)
            if self.target_transform is not None:
                context_box = self.target_transform(context_box)
                search_box = self.target_transform(search_box)

            yield context, search, context_box, search_box, search_context_box

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
                '  - # of sequences: {}\n'
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
