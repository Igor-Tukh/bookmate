import math
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

from src.metasessions_module.interest_marker.config import get_book_stats_path
from src.metasessions_module.utils import save_result_to_csv, min_max_scale

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.interest_marker_utils import load_markers_for_user, \
    calculate_numbers_of_readers_per_fragment, load_quit_marker
from src.metasessions_module.user_utils import get_good_users_info
from collections import defaultdict


def calculate_common_ratio(first, second):
    """
    Calculates the ratio of number of common elements in set and number of elements in set union.
    First and second have to be of the same length.
    """
    if first.shape != second.shape:
        logging.error('Can\'t calculate common ration: provided vectors of unequal shape.')
    common = 1. * np.sum(first & second, dtype=np.int)
    total = np.sum(first | second, dtype=np.int)
    return 1. if total == 0 else 1. * common / total


def compare_markers_common_ratio(output_path, markers):
    """
    For each user and each two markers calculates the proportion of a common marker positions. For each pair of markers
    calculates average ration and saves the result to the output_path.
    markers: list of dictionaries from marker_description to marker boolean value, Each list represents one user.
    """
    if len(markers) == 0:
        logging.info('Provided empty list of markers')
        return

    distances = defaultdict(lambda: {})
    for first_marker_type in markers[0].keys():
        for second_marker_type in markers[0].keys():
            distance = np.mean([calculate_common_ratio(m[first_marker_type], m[second_marker_type]) for m in markers])
            distances[first_marker_type][second_marker_type] = distance

    with open(output_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['marker'] + list(markers[0].keys()))
        writer.writeheader()
        for marker, marker_distances in distances.items():
            results = {'marker': marker}
            results.update(marker_distances)
            writer.writerow(results)


def calculate_manifestation_masks_frequencies(user_markers):
    freq = defaultdict(lambda: 0)
    if len(users_markers) == 0:
        return None
    keys = users_markers[0].keys()
    fragments_number = list(users_markers[0].values())[0].shape[0]
    for user_marker in users_markers:
        for ind in range(fragments_number):
            freq[tuple([user_marker[key][ind] for key in keys])] += 1
    results = []
    for mask, count in freq.items():
        result = {'Number': count}
        for key, manifestation in zip(keys, list(mask)):
            result[key] = manifestation
        results.append(result)
    return results


def _fix_description(description):
    result = ' '.join(description.split('_'))
    if 're reading' in result:
        result = 're-reading marker'
    return result


def plot_combined_markers_bars(book, user_ids, users_markers, output_path, scaled=False):
    if len(users_markers) == 0:
        logging.info('Can\'t plot combined markers: no markers provided')
        return
    marker_descriptions = list(users_markers[0].keys())
    fragments_number = list(users_markers[0].values())[0].shape[0]
    markers = defaultdict(lambda: np.zeros(fragments_number, dtype=np.int))
    for marker_description in marker_descriptions:
        for user_markers in users_markers:
            markers[marker_description] += user_markers[marker_description]
    normalization_weight = calculate_numbers_of_readers_per_fragment(book, user_ids, fragments_number)
    if scaled:
        for description in marker_descriptions:
            markers[description] = min_max_scale(markers[description])

    plt.clf()
    fragments = np.arange(fragments_number) * len(marker_descriptions)
    width = 0.9
    fig, ax = plt.subplots(figsize=(35, 20))
    rects = [ax.bar(fragments + width * (ind + 1. / 2), markers[description] / normalization_weight, width,
                    label=_fix_description(description))
             for ind, description in enumerate(marker_descriptions)]

    ax.set_xlabel('Fragments')
    ax.set_ylabel('Interest markers')
    ax.set_xticks(fragments[::20])
    ax.set_xticklabels([str(fragment) for fragment in fragments[::20]])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(output_path, bbox_inches='tight')


def plot_combined_markers(book, user_ids, users_markers, output_path, scaled=False):
    if len(users_markers) == 0:
        logging.info('Can\'t plot combined markers: no markers provided')
        return
    marker_descriptions = list(users_markers[0].keys())
    fragments_number = list(users_markers[0].values())[0].shape[0]
    markers = defaultdict(lambda: np.zeros(fragments_number, dtype=np.int))
    for marker_description in marker_descriptions:
        for user_markers in users_markers:
            markers[marker_description] += user_markers[marker_description]
    normalization_weight = calculate_numbers_of_readers_per_fragment(book, user_ids, fragments_number)
    for description in marker_descriptions:
        markers[description] = 1. * markers[description] / normalization_weight
    if scaled:
        for description in marker_descriptions:
            markers[description] = min_max_scale(markers[description])
    markers['quit_marker'] = load_quit_marker(book)

    plt.clf()
    fragments = np.arange(fragments_number)
    fig, ax = plt.subplots(figsize=(35, 20))
    for description in marker_descriptions:
        ax.plot(fragments, markers[description], label=_fix_description(description))

    ax.set_xlabel('Fragments')
    ax.set_ylabel('Interest markers')
    ax.set_xticks(fragments[::25])
    ax.set_xticklabels([str(fragment) for fragment in fragments[::25]])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(output_path, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    users = list(get_good_users_info(book_id).keys())
    users_markers = [load_markers_for_user(book_id, user) for user in users]

    # compare_markers_common_ratio(get_book_stats_path(book_id, 'common_ratio.csv'), users_markers)
    # save_result_to_csv(calculate_manifestation_masks_frequencies(users_markers),
    #                    get_book_stats_path(book_id, 'manifestation_frequencies.csv'))

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 24}
    plt.rc('font', **font)

    plot_combined_markers(
        book_id, users, users_markers, os.path.join('resources', 'plots', str(book_id), 'combined_markers.png'))
    # plot_combined_markers(
    #     book_id, users, users_markers, os.path.join('resources', 'plots', str(book_id), 'combined_markers_scaled.png'),
    #     scaled=True)
    # plot_combined_markers_bars(
    #     book_id, users, users_markers, os.path.join('resources', 'plots', str(book_id), 'combined_markers_bars.png'))
    # plot_combined_markers_bars(
    #     book_id, users, users_markers, os.path.join('resources', 'plots', str(book_id),
    #                                                 'combined_markers_scaled_bars.png'),
    #     scaled=True)
