import csv
import os

import numpy as np

from PIL import Image
from torch.utils.data import Dataset

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
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

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

    def __len__(self):
        return len(self.img_list)

    def _get_item(self, index):
        r"""
        Get an item from the dataset.

        Parameters
        ----------
        index : int


        Returns
        -------
        items: (list[np.ndarray], list[np.ndarray])

        """
        if index < 0 or index > len(self):
            raise ValueError('The requested `index`, {}, is not valid. '
                             'Valid indices go from 0 to {}. '
                             .format(index, len(self)))

        img_sequence = self.img_list[index]
        ann_sequence = self.ann_list[index]

        new_img_sequence = [Image.open(img_path) for img_path in img_sequence]

        return new_img_sequence, ann_sequence

    def __getitem__(self, index):
        img_sequence, ann_sequence = self._get_item(index)

        if self.transform is not None:
            img_sequence = [self.transform(img) for img in img_sequence]

        if self.target_transform is not None:
            ann_sequence = [self.target_transform(ann) for ann in ann_sequence]

        return img_sequence, ann_sequence

    def view(self):
        r"""
        Visualize dataset.
        """
        try:
            from .widgets import notebook_view_sequence
            notebook_view_sequence(self.img_list, self.ann_list)
        except ImportError:
            raise IPythonWidgetsMissingError()
