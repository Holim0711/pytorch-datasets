import os
from PIL import Image
from torch.utils.data import Dataset
from .utils import read_lines


class MultiTaskImageDataset(Dataset):

    def __init__(self, filename, root=None, transform=None, target_transform=None):
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
            - transform:
                Image transform function
            - target_transform:
                Annotation transform function
        """
        self.filename = filename
        self.root = root if root is not None else os.getcwd()
        self.transform = transform
        self.target_transform = target_transform

        lines = read_lines(filename)

        self.items = [(str(x), list(map(int, y))) for x, *y in lines]

    def __getitem__(self, idx):
        x, y = self.items[idx]

        x = Image.open(os.path.join(self.root, x))

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.items)
