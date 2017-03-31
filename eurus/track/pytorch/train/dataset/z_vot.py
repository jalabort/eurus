import csv
import os
import sys
import numbers
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .z_utils import crop, view_pair

if 'ipykernel' in sys.modules:
    from ipywidgets import interact
    import ipywidgets as widgets


class Vot2016(Dataset):
    r"""
    """
    def __init__(self, root, transform=None, target_transform=None, offset=5,
                 context_factor=3, search_factor=2, context_size=128,
                 search_size=256, stride=8):
        self._root = root
        self._transform = transform
        self._target_transform = target_transform
        self._offset = offset
        self._context_factor = context_factor
        self._search_factor = search_factor
        self._stride = stride

        if isinstance(context_size, numbers.Number):
            self._context_size = int(context_size), int(context_size)
        else:
            self._context_size = context_size
        if isinstance(search_size, numbers.Number):
            self._search_size = int(search_size), int(search_size)
        else:
            self._search_size = search_size

        # if not self.check_exists():
        #             warnings.warn("Dataset not found. Downloading it." , RuntimeWarning)
        #             if not self.download():
        #                 raise RuntimeError("Error downloading dataset.")

        directories = next(os.walk(root))[1]
        paths = [root + d + "/" for d in directories]

        self._img_path_pairs = []
        for p in paths:
            files = next(os.walk(p))[2]
            img_files = [f for f in files if f.split(sep=".")[-1] == "jpg"]
            img_file_pairs = [(p + img_files[i], p + img_files[i + offset])
                              for i in range(len(img_files) - offset)]
            self._img_path_pairs += img_file_pairs

        self._box_pairs = []
        for p in paths:
            with open(p + "groundtruth.txt", "r") as f:
                gt = list(csv.reader(f))
            for i in range(len(gt) - offset):
                x11 = float(gt[i][0])
                y11 = float(gt[i][1])
                x12 = float(gt[i][2])
                y12 = float(gt[i][3])
                x13 = float(gt[i][4])
                y13 = float(gt[i][5])
                x14 = float(gt[i][6])
                y14 = float(gt[i][7])

                x21 = float(gt[i + offset][0])
                y21 = float(gt[i + offset][1])
                x22 = float(gt[i + offset][2])
                y22 = float(gt[i + offset][3])
                x23 = float(gt[i + offset][4])
                y23 = float(gt[i + offset][5])
                x24 = float(gt[i + offset][6])
                y24 = float(gt[i + offset][7])

                self._box_pairs += [(
                    [x11, y11, x12, y12, x13, y13, x14, y14],
                    [x21, y21, x22, y22, x23, y23, x24, y24]
                )]

        assert len(self) == len(self._box_pairs), \
            "The number of image ({}) and box ({}) pairs should be the same".format(
                len(self), len(self._box_pairs))

    def _getitem_(self, index):
        box1, box2 = self._box_pairs[index]
        center1 = sum(box1[0::2]) / 4, sum(box1[1::2]) / 4

        tl1 = min(box1[0::2]), min(box1[1::2])
        br1 = max(box1[0::2]), max(box1[1::2])
        size = np.ceil((br1[1] - tl1[1])), np.ceil((br1[0] - tl1[0]))

        size1 = max(size) * self._context_factor, max(
            size) * self._context_factor
        size2 = max(size1) * self._search_factor, max(
            size1) * self._search_factor

        img_path1, img_path2 = self._img_path_pairs[index]
        img1, img2 = Image.open(img_path1), Image.open(img_path2)

        #         size1 = (256, 256)
        #         size2 = size1
        img1 = crop(img1, center1, size1)
        img1 = img1.resize(self._context_size, resample=Image.BICUBIC)
        img2 = crop(img2, center1, size2)
        img2 = img2.resize(self._search_size, resample=Image.BICUBIC)

        ratio1 = size1[0] / self._context_size[0], size1[1] / \
                 self._context_size[1]
        ratio2 = size2[0] / self._search_size[0], size2[1] / self._search_size[
            1]

        x, y = center1
        h, w = size1
        x1 = int(round(x - w / 2.))
        y1 = int(round(y - h / 2.))

        box1 = [(box1[0] - x1) / ratio1[1], (box1[1] - y1) / ratio1[0],
                (box1[2] - x1) / ratio1[1], (box1[3] - y1) / ratio1[0],
                (box1[4] - x1) / ratio1[1], (box1[5] - y1) / ratio1[0],
                (box1[6] - x1) / ratio1[1], (box1[7] - y1) / ratio1[0]]

        x, y = center1
        h, w = size2
        x2 = int(round(x - w / 2.))
        y2 = int(round(y - h / 2.))

        box2 = [(box2[0] - x2) / ratio2[1], (box2[1] - y2) / ratio2[0],
                (box2[2] - x2) / ratio2[1], (box2[3] - y2) / ratio2[0],
                (box2[4] - x2) / ratio2[1], (box2[5] - y2) / ratio2[0],
                (box2[6] - x2) / ratio2[1], (box2[7] - y2) / ratio2[0]]

        labels = np.zeros((int(img2.size[0] / self._stride) + 1,
                           int(img2.size[1] / self._stride) + 1))
        center2 = round(sum(box2[0::2]) / 4 / self._stride), round(
            sum(box2[1::2]) / 4 / self._stride)

        # print("----", index)
        # print(labels.shape)
        # print(center2)

        labels[int(center2[1]), int(center2[0])] = 1

        # print("----end", index)

        return (img1, img2), (box1, box2), labels

    def __getitem__(self, index):
        (img1, img2), (box1, box2), labels = self._getitem_(index)

        if self._transform is not None:
            img1, img2 = self._transform(img1), self._transform(img2)

        if self._target_transform is not None:
            box1, box2 = (self._target_transform(box1),
                          self._target_transform(box2))

        if self._transform is not None:
            labels = self._transform(labels)

        return (img1, img2), (box1, box2), labels

    def __len__(self):
        return len(self._img_path_pairs)

    def view(self):
        if 'ipykernel' in sys.modules:
            self._notebook_view()
        else:
            print("hello")

    def _notebook_view(self):
        def _view_pairs(index):
            img_pair, box_pair, labels = self._getitem_(index)
            view_pair(img_pair, box_pair, labels, self._stride)

        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self),
            step=1,
            description='index:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='i',
            slider_color='white'
        )

        interact(_view_pairs, index=slider)
