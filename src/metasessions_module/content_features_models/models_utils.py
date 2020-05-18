import os
import csv
import sys

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.integrated_interest_marker import load_integrated_markers
from src.metasessions_module.interest_marker.interest_marker_utils import load_normalized_interest_markers, \
    save_normalized_interest_markers


def load_regression_markers(book, with_individual=False):
    markers = load_integrated_markers(book)
    if with_individual:
        markers.update(load_normalized_interest_markers(book))
    return markers


def get_models_path(book):
    dir_path = os.path.join('resources', 'models', str(book))
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_models_stats_path(book):
    dir_path = os.path.join(get_models_path(book), 'stats')
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def update_stats(book, new_row, description, rewrite=False):
    description = '_'.join(description.split(' '))
    description = f'{description}.csv'
    stats_path = os.path.join(get_models_stats_path(book), description)
    rows = []
    if not isinstance(new_row, list):
        new_row = [new_row]
    if os.path.exists(stats_path) and not rewrite:
        with open(stats_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = [row for row in reader]
    rows += new_row
    with open(stats_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=new_row[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    pass
