import numpy as np

from eurus.track.pytorch.train.dataset.base import SequenceDataset

from .base import Alov


class AlovSequence(SequenceDataset, Alov):
    r"""
    Class for the Amsterdam Library of Ordinary Videos (ALOV) 300 dataset.

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
    A. W. Smeulders, et al. "Visual tracking: An experimental survey".
    TPAMI 2013.
    """

    def __init__(self, root, transform=None, target_transform=None,
                 sequence_length=None, skip=None, context_factor=3,
                 search_factor=2, context_size=128, search_size=256,
                 response_size=33):

        super(AlovSequence, self).__init__(
            root, transform=transform, target_transform=target_transform,
            sequence_length=sequence_length, skip=skip,
            context_factor=context_factor, search_factor=search_factor,
            context_size=context_size, search_size=search_size,
            response_size=response_size)

        if sequence_length is None:
            self.sequence_length = self.n_shortest

    def __len__(self):
        length = 0
        for img_group in self.img_list:
            for img_sequence in img_group:
                length += len(img_sequence) - (self.sequence_length - 1)
        return length

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
                for first in range(len(img_sequence) -
                                   (self.sequence_length - 1)):
                    if aux_index == index:
                        found = True
                        break
                    aux_index += 1
                if found:
                    break
            if found:
                break

        last = first + self.sequence_length * self.skip

        return (img_sequence[first:last:self.skip],
                ann_sequence[first:last:self.skip])