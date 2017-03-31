import os
import csv

import numpy as np

from eurus.track.pytorch.train.dataset.base import TrackingDataset


class Uav123(TrackingDataset):
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
                 sequence_length=None, skip=None, context_factor=3,
                 search_factor=2, context_size=128, search_size=256):

        super(Uav123, self).__init__(
            root, transform=transform, target_transform=target_transform,
            sequence_length=sequence_length, skip=skip,
            context_factor=context_factor, search_factor=search_factor,
            context_size=context_size, search_size=search_size)

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
