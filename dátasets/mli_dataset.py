import os
from PIL import Image
from torch.utils.data import Dataset
from torch import FloatTensor
from .utils import read_lines
from .mti_dataset import MultiTaskImageDataset


class MultiLabelImageDataset(MultiTaskImageDataset):

    def __init__(self, filename, root=None, groups=None, transform=None, target_transform=None):
        """ Multi-Task Image Dataset

        args:
            - filename:
                Data file which contains image paths and annotations
                Only UTF-8 encoded files are available!
                The 1st column should be path, and others should be integer.
                Available extension formats:
                    .csv: following csv 'excel' dialect
                    .tsv: following csv 'excel-tab' dialect
                    .json: list of tuples without header row
                    .txt: split by white-space charaters without header row
            - root:
                Image root directory
                If not given, return value of os.getcwd() will be used.
            - groups:
                Label groups. ex) `[[0, 1, 2], [3, 4], [5, 6, 7, 8]]`
                if groups is none:
                    x, y == PIL, tensor([0., 1., 1., 0., ...])
                else:
                    x, y == PIL, [tensor([0., 1., 1.]), tensor([0., 1.]), ...]
            - transform:
                Image transform function
            - target_transform:
                Annotation transform function
        """
        super().__init__(filename, root, transform, target_transform)
        self.groups = groups

        if groups:
            self.items = [(x, [FloatTensor([y[i] for i in g]) for g in groups]) for x, y in self.items]
        else:
            self.items = [(x, FloatTensor(y)) for x, y in self.items]
