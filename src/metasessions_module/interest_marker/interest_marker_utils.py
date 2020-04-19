import sys
import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.metasessions_module.interest_marker.config import MarkerManifestation, get_markers_path_for_book
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.user_utils import get_user_document_id

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from tqdm import tqdm
from collections import defaultdict
from src.metasessions_module.utils import get_batch_by_percent, is_int, load_from_pickle

SAVED_MARKERS_PATH = os.path.join('resources', 'all_markers')


def position_to_batch(position, batch_borders):
    i = 0
    while i < len(batch_borders) and batch_borders[i] < position + 1e-9:
        i += 1
    return min(i, len(batch_borders) - 1)


def get_batch(batches_number, batches_borders, position):
    if batches_number is not None:
        return get_batch_by_percent(batches_number, position)
    else:
        return position_to_batch(position / 100, batches_borders)


def get_fragment(fragments_borders, position):
    return position_to_batch(position / 100, fragments_borders)


def load_saved_markers():
    """
    Loads all dumped markers. Returns defaultdict from book_id to a dict from marker type to marker.
    :return:
    """
    # TODO: add speed
    all_markers = defaultdict(lambda: ({'quit_marker': {},
                                        'unusual_hours_marker': {},
                                        're_reading_marker': {},
                                        'reading_interrupt_marker': {}}))

    for ind, filename in tqdm(enumerate(sorted(os.listdir(SAVED_MARKERS_PATH)))):
        file_path = os.path.join(SAVED_MARKERS_PATH, filename)
        with open(file_path, 'rb') as file:
            markers = pickle.load(file)

        prefix = filename.split('_')[0]
        user_id = -1
        book_id = -1
        tokens = filename.split('_')
        for token_ind, token in enumerate(tokens):
            if is_int(token):
                book_id = int(token)
                user_id = tokens[token_ind + 1].split('.')[0]
                break
        if prefix == 'quit':
            all_markers[book_id]['quit_marker'][user_id] = markers
        elif prefix == 'unusual':
            all_markers[book_id]['unusual_hours_marker'][user_id] = markers
        elif prefix == 're':
            all_markers[book_id]['re_reading_marker'][user_id] = markers
        elif prefix == 'reading':
            all_markers[book_id]['reading_interrupt_marker'][user_id] = markers

    return all_markers


def calculate_cumulative_markers(markers_dict):
    """
    Calculates cumulative marker of interest. Markers dict is a dict from user id to actual marker value.
    :param markers_dict:
    :return:
    """
    total_markers = np.zeros(list(markers_dict.values())[0].shape[0], dtype=np.int)
    for marker in markers_dict.values():
        total_markers += marker
    return total_markers


def get_manifestation_color(manifestation):
    if manifestation == MarkerManifestation.INTERESTING:
        return 'r'
    elif manifestation == MarkerManifestation.NON_INTERESTING:
        return 'b'
    return 'w'


def get_binary_color(marker):
    return 'r' if marker else 'b'


def visualize_markers_manifestations(markers, output_path, title='', x_label='', y_label=''):
    """
    Build and saves a plot of a marker manifestations. @markers should be a list of np.array of
    MarkerManifestation. All arrays should be the same length.
    """
    plt.clf()
    plt.axis('off')
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if len(markers) != 0:
        n = markers[0].shape[0]
        ax.set_xlim(0, n)
        ax.set_ylim(0, len(markers))
        for ind, marker in enumerate(markers):
            for place_ind in range(n):
                circle = plt.Rectangle((place_ind, ind), 1, 1, color=get_manifestation_color(marker[place_ind]))
                ax.add_artist(circle)

    plt.savefig(output_path)


def binarize_marker(marker):
    """
    Transforms marker from numerical to a boolean.
    """
    return np.array(marker > 0, dtype=np.bool)


def visualize_binary_markers(markers, output_path, title='', x_label='', y_label='', binarize=False):
    """
    Build and saves a plot of a markers manifestations.
    """
    plt.clf()
    plt.axis('off')
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if len(markers) != 0:
        n = markers[0].shape[0]
        ax.set_xlim(0, n)
        ax.set_ylim(0, len(markers))
        for ind, marker in enumerate(markers):
            if binarize:
                marker = binarize_marker(marker)
            for place_ind in range(n):
                circle = plt.Rectangle((place_ind, ind), 1, 1, color=get_binary_color(marker[place_ind]))
                ax.add_artist(circle)

    plt.savefig(output_path)


def get_cumulative_marker(markers, binarize=False, normalization_weight=None):
    if binarize:
        markers = [binarize_marker(marker) for marker in markers]
    numbers = np.sum(markers, axis=0, dtype=np.int)
    if normalization_weight is not None:
        numbers = np.array(numbers / normalization_weight, dtype=np.float32)
    return numbers


def visualize_cumulative_marker(markers, output_path, title='', binarize=False, normalization_weight=None):
    numbers = get_cumulative_marker(markers, binarize, normalization_weight)
    plt.clf()
    plt.plot(np.arange(markers[0].shape[0]), numbers)
    plt.title(title)
    plt.xlabel('Fragment')
    plt.ylabel('Cumulative marker value')
    plt.savefig(output_path)


def get_read_fragments(book_id, user_id, sessions=None, fragment_borders=None):
    if fragment_borders is None:
        fragment_borders = get_split_text_borders(book_id)
    if sessions is None:
        sessions = get_user_sessions(book_id, get_user_document_id(book_id, user_id), user_id)

    result = np.zeros(len(fragment_borders), dtype=np.bool)
    for session in sessions:
        if 'book_from' in session:
            result[get_fragment(fragment_borders, session['book_from'])] = True
        if 'book_to' in session:
            result[get_fragment(fragment_borders, session['book_to'])] = True
    return result


def calculate_numbers_of_readers_per_fragment(book_id, user_ids, n_fragments):
    result = np.zeros(n_fragments, dtype=np.int)
    for user_id in user_ids:
        result += get_read_fragments(book_id, user_id)

    return result


def save_marker_stats(filename, marker, marker_description='Marker'):
    with open(filename, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Fragment index', marker_description])
        writer.writeheader()
        for ind, marker_value in enumerate(marker):
            writer.writerow({'Fragment index': ind, marker_description: marker_value})


def save_marker_stats_with_normalization(filename, marker, normalization_weight, marker_description='Marker'):
    with open(filename, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Fragment index', marker_description,
                                                      f'Normalized {marker_description}'])
        writer.writeheader()
        for ind, (marker_value, weight) in enumerate(zip(marker, normalization_weight)):
            writer.writerow({'Fragment index': ind, marker_description: marker_value,
                             f'Normalized {marker_description}': 1. * marker_value / weight})


def load_markers_for_user(book_id, user_id):
    markers_path = get_markers_path_for_book(book_id)
    markers = {}
    for dir_name in [name for name in os.listdir(markers_path) if os.path.isdir(os.path.join(markers_path, name))]:
        pkl_path = os.path.join(markers_path, dir_name, f'{user_id}.pkl')
        if os.path.exists(pkl_path):
            markers[dir_name] = load_from_pickle(pkl_path)
    return markers
