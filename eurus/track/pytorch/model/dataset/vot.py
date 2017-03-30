import os
import csv

import numpy as np

from .base import TrackingDataset


class Vot2016(TrackingDataset):
    r"""
    Class for the Visual Object Tracking (VOT) 2016 dataset.

    Parameters
    ----------
    root : str
        The root path to the dataset.
    transform :

    target_transform :

    sequence_length : int, optional

    search_factor : float, optional

    context_size : int, optional

    search_size : int, optional


    References
    ----------
    M. Kristan, et al. "The Visual Object Tracking VOT2016 challenge results".
    ECCV 2016.
    """
    def __init__(self, root, transform=None, target_transform=None,
                 sequence_length=None, skip=None, context_factor=3,
                 search_factor=2, context_size=128, search_size=256):

        super(Vot2016, self).__init__(
            root, transform=transform, target_transform=target_transform,
            sequence_length=sequence_length, skip=skip,
            context_factor=context_factor, search_factor=search_factor,
            context_size=context_size, search_size=search_size)

        sequences = sorted(next(os.walk(root))[1])
        paths = [os.path.join(root, d) for d in sequences]

        self.ann_list = []
        self.img_list = []

        for p in paths:
            files = sorted(next(os.walk(p))[2])

            with open(os.path.join(p, 'groundtruth.txt'), 'r') as f:
                truths = list(csv.reader(f))
            ann_list2 = []
            for t in truths:
                box = np.array([[float(t[0]), float(t[1])],
                                [float(t[2]), float(t[3])],
                                [float(t[4]), float(t[5])],
                                [float(t[6]), float(t[7])]])
                tl = np.min(box, axis=0)
                br = np.max(box, axis=0)
                sz = br - tl
                ann_list2.append(np.concatenate([tl, sz]))
            self.ann_list.append(ann_list2)

            img_list2 = []
            for f in files:
                if f.split(sep='.')[-1] == 'jpg':
                    img_list2.append(os.path.join(p, f))
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

        if sequence_length is None:
            self.sequence_length = self.n_shortest - 1

    @property
    def _n_elements_per_sequence(self):
        return [len(sequence) for sequence in self.img_list]

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
