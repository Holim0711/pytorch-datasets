import os
import csv
import json
from PIL import Image
from torch.utils.data import Dataset


def read_excel(filename, dialect):
    with open(filename, newline='', encoding='utf-8') as file:
        has_header = csv.Sniffer().has_header(file.read(1024))
        file.seek(0)
        lines = [line for line in csv.reader(file, dialect)]
    return lines[1:] if has_header else lines


def read_json(filename):
    with open(filename, encoding='utf-8') as file:
        lines = json.load(file)
    return lines


def read_text(filename):
    with open(filename, encoding='utf-8') as file:
        lines = [line.split() for line in file]
    return lines


def read_lines(filename):
    ext = os.path.splitext(filename)[-1]
    if ext == '.csv':
        lines = read_excel(filename, 'excel')
    elif ext == '.tsv':
        lines = read_excel(filename, 'excel-tab')
    elif ext == '.json':
        lines = read_json(filename)
    elif ext == '.txt':
        lines = read_text(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    return lines


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

