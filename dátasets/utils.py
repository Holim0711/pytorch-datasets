import os
import csv
import json


def read_excel(filename, dialect):
    with open(filename, newline='', encoding='utf-8') as file:
        has_header = csv.Sniffer().has_header(file.read(2048))
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
