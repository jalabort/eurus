import os
import csv

import numpy as np

from .base import TrackingDataset
from .widgets.exception import IPythonWidgetsMissingError


class Alov300(TrackingDataset):
    r"""
    Class for the Amsterdam Library of Ordinary Videos (ALOV) 300 dataset.

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
    A. W. Smeulders, et al. "Visual tracking: An experimental survey".
    TPAMI 2013.
    """
    def __init__(self, root, transform=None, target_transform=None,
                 sequence_length=None, skip=None, context_factor=3,
                 search_factor=2, context_size=128, search_size=256):

        super(Alov300, self).__init__(
            root, transform=transform, target_transform=target_transform,
            sequence_length=sequence_length, skip=skip,
            context_factor=context_factor, search_factor=search_factor,
            context_size=context_size, search_size=search_size)

        img_path = os.path.join(root, 'images')
        ann_path = os.path.join(root, 'annotations')

        sequences = sorted(next(os.walk(img_path))[1])

        self.ann_list = []
        self.img_list = []

        for seq in sequences:
            ann_path2, _, files = next(os.walk(os.path.join(ann_path, seq)))
            files = sorted(files)
            ann_list2 = []
            ann_indices = []
            for file in files:
                with open(os.path.join(ann_path2, file), "r") as f:
                    truths = list(csv.reader(f, delimiter=' '))
                ann_list3 = []
                indices = []
                for t in truths:
                    xs = np.array(t[1::2]).astype(np.float32)
                    ys = np.array(t[2::2]).astype(np.float32)
                    tl = np.array([np.min(xs), np.min(ys)])
                    br = np.array([np.max(xs), np.max(ys)])
                    sz = br - tl
                    ann_list3.append(np.concatenate([tl, sz]))
                    indices.append(int(t[0]) - 1)
                ann_list2.append(ann_list3)
                ann_indices.append(indices)
            self.ann_list.append(ann_list2)

            img_path2, dirs, _ = next(os.walk(os.path.join(img_path, seq)))
            dirs = sorted(dirs)
            img_list2 = []
            for d, indices in zip(dirs, ann_indices):
                img_path3, _, files = next(os.walk(os.path.join(img_path2, d)))
                files = sorted(files)
                files = [files[i] for i in indices]
                img_list3 = []
                for file in files:
                    img_list3.append(os.path.join(img_path3, file))
                img_list2.append(img_list3)
            self.img_list.append(img_list2)

        assert len(self.img_list) == len(self.ann_list), \
            'The number of image ({}) and ann ({}) groups should be ' \
            'the same.'.format(len(self.img_list), len(self.ann_list))
        for i, (img_list2, ann_list2) in enumerate(zip(self.img_list,
                                                       self.ann_list)):
            assert len(img_list2) == len(ann_list2), \
                'The number of image ({}) and annotations ({}) sequences ' \
                'in group {} should be the same.'.format(
                    len(img_list2), len(ann_list2), i)
            for j, (img_list3, ann_list3) in enumerate(zip(img_list2,
                                                           ann_list2)):
                assert len(img_list3) == len(ann_list3), \
                    'The number of image ({}) and annotations ({}) in ' \
                    'sequence {} of group {} should be the same.'.format(
                        len(img_list3), len(ann_list3), j, i)

        if sequence_length is None:
            self.sequence_length = self.n_shortest - 1

    @property
    def _n_elements_per_sequence(self):
        return [len(sequence) for group in self.img_list for sequence in group]

    def __len__(self):
        n_sequences = 0
        for img_group in self.img_list:
            n_sequences += len(img_group)
        return n_sequences

    def view_original(self):
        r"""
        Visualize the original unprocessed dataset.
        """
        try:
            from .widgets import notebook_view_group
            notebook_view_group(self.img_list, self.ann_list)
        except ImportError:
            raise IPythonWidgetsMissingError()

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

        aux_index = 0
        found = False
        for img_group, ann_group in zip(self.img_list, self.ann_list):
            for img_sequence, ann_sequence in zip(img_group, ann_group):
                if aux_index == index:
                    found = True
                    break
                aux_index += 1
            if found:
                break

        first = np.random.randint(
            0, len(img_sequence) - self.sequence_length * self.skip)
        last = first + self.sequence_length * self.skip

        return (img_sequence[first:last:self.skip],
                ann_sequence[first:last:self.skip])
