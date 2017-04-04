import random

import numpy as np

from eurus.track.pytorch.train.dataset.base import PairDataset

from .base import Alov


class AlovPair(PairDataset, Alov):
    r"""
    Class for the Amsterdam Library of Ordinary Videos (ALOV) 300 dataset.

    Parameters
    ----------
    root : str
        The root path to the dataset.
    transform :

    target_transform :

    offset : int, optional

    search_factor : float, optional

    context_size : int, optional

    search_size : int, optional


    References
    ----------
    A. W. Smeulders, et al. "Visual tracking: An experimental survey".
    TPAMI 2013.
    """
    def __init__(self, root, transform=None, target_transform=None,
                 offset=3, context_factor=3, search_factor=2,
                 context_size=128, search_size=256, response_size=33):

        super(AlovPair, self).__init__(
            root, transform=transform, target_transform=target_transform,
            offset=offset, context_factor=context_factor,
            search_factor=search_factor, context_size=context_size,
            search_size=search_size, response_size=response_size)

    def __len__(self):
        length = 0
        for img_group in self.img_list:
            for img_sequence in img_group:
                length += len(img_sequence)
        return length

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
        if index < 0 or index > len(self):
            raise ValueError('The requested `index`, {}, is not valid. '
                             'Valid indices go from 0 to {}. '
                             .format(index, len(self)))

        aux_index = 0
        found = False
        for img_group, ann_group in zip(self.img_list, self.ann_list):
            for img_sequence, ann_sequence in zip(img_group, ann_group):
                sequence_length = len(img_sequence)
                for selected in range(len(img_sequence)):
                    if aux_index == index:
                        found = True
                        break
                    aux_index += 1
                if found:
                    break
            if found:
                break

        last = selected + self.offset
        if last > sequence_length:
            last = sequence_length

        first = selected - self.offset
        if first < 0:
            first = 0

        candidates = set(range(first, last)) - {selected}
        chosen = random.sample(candidates, 1)[0]

        return ((img_sequence[selected], img_sequence[chosen]),
                (ann_sequence[selected], ann_sequence[chosen]))
