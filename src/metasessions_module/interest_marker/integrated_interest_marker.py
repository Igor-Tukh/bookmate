import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

from src.metasessions_module.interest_marker.config import get_book_stats_path
from src.metasessions_module.interest_marker.interest_marker_utils import load_markers_for_user, \
    calculate_numbers_of_readers_per_fragment, smooth_marker, save_marker_stats, load_mark_up, get_custom_path, \
    custom_visualize_marker_with_mark_up
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.user_utils import get_good_users_info
from src.metasessions_module.utils import save_via_pickle, load_from_pickle, min_max_scale, get_stats_path

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.markers_manifestations_analysis import \
    get_manifestations_clustering_results, get_individual_manifestations_clustering_results
from collections import defaultdict
from tqdm import tqdm


def build_integrated_interest_signal_best_manifestations(book, criterion='Number of 0',
                                                         select_biggest=True, std_weighted=False):
    """
    Collects the best manifestation according to @criterion. Calculates cumulative interest signal.
    Use @select_biggest=False to select data with lowest criterion value
    Possible criterion: 'Silhouette Coefficient', 'Calinski-Harabasz Index', 'Davies-Bouldin Index',  'Number of 0'
    """
    best_scores = defaultdict(lambda: (None, None, None))
    shape = None
    for score in get_manifestations_clustering_results():
        shape = score['mask'].shape
        if int(score['book_id']) == book:
            if score[criterion] == 'und':
                continue
            best_score = best_scores[score['marker']][0]
            current_score = score[criterion]
            if best_score is None or ((select_biggest and best_score < current_score) or
                                      (not select_biggest and best_score > current_score)):
                best_scores[score['marker']] = current_score, score['mask'], score['Marker std']

    if shape is None:
        return None
    result = np.zeros(shape, dtype=np.float32)
    total_weight = 0
    for (score, mask, std) in best_scores.values():
        if score is not None:
            current_weight = 1 / std if std_weighted else 1
            total_weight += current_weight
            result += np.array(mask, dtype=np.int) * current_weight

    return 1. * result / total_weight


def build_average_interest_signal_best_manifestations(book, criterion='Number of 0',
                                                      select_biggest=True, std_weighted=False):
    results = {}

    for user_id, individual_results in tqdm(get_individual_manifestations_clustering_results().items()):
        best_scores = defaultdict(lambda: (None, None, None))
        shape = None
        for score in individual_results:
            shape = score['mask'].shape
            if int(score['book_id']) == book:
                if score[criterion] == 'und':
                    continue
                best_score = best_scores[score['marker']][0]
                current_score = score[criterion]
                if best_score is None or ((select_biggest and best_score < current_score) or
                                          (not select_biggest and best_score > current_score)):
                    best_scores[score['marker']] = current_score, score['mask'], score['Marker std']

        if shape is None:
            return None
        result = np.zeros(shape, dtype=np.float32)
        total_weight = 0
        for (score, mask, std) in best_scores.values():
            if score is not None:
                current_weight = 1 / std if std_weighted else 1
                total_weight += current_weight
                result += np.array(mask, dtype=np.int) * current_weight

        results[user_id] = result / total_weight

    print(np.mean(list(results.values()), axis=1))
    return np.mean(list(results.values()), axis=1)


def _plot_integrated_signal(book, signal, output_path, bar=False):
    plt.clf()
    plt.xlabel('Fragment number')
    plt.ylabel('Integrated interest level')
    plt.title(f'Integrated interest signal for book {book}')
    if bar:
        plt.bar(np.arange(signal.shape[0]) + 0.5, signal, width=1)
    else:
        plt.plot(np.arange(signal.shape[0]), signal)
    plt.savefig(output_path)


def _get_plots_directory_path(book):
    directory_path = os.path.join('resources', 'plots', 'markers', str(book), 'integrated')
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def get_integrated_marker_dumps_directory_path(book):
    dir_path = os.path.join('resources', 'integrated_marker', str(book))
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_integrated_marker_plot_path(book, description):
    description = '_'.join(description.split(' '))
    return os.path.join(_get_plots_directory_path(book), f'{description}.png')


def build_average_integrated_interest_signal(book):
    users = list(get_good_users_info(book).keys())
    fragments_borders = get_split_text_borders(book)
    users_markers = [load_markers_for_user(book_id, user) for user in users]
    normalization_weight = calculate_numbers_of_readers_per_fragment(book, users, len(fragments_borders))
    result = np.zeros(len(fragments_borders), dtype=np.float32)
    for user_markers in users_markers:
        n_markers = len(user_markers)
        for marker in user_markers.values():
            result += marker / n_markers
    return result / normalization_weight


def build_average_scaled_integrated_interest_signal(book):
    users = list(get_good_users_info(book).keys())
    fragments_borders = get_split_text_borders(book)
    users_markers = [load_markers_for_user(book_id, user) for user in users]
    normalization_weight = calculate_numbers_of_readers_per_fragment(book, users, len(fragments_borders))
    results = defaultdict(lambda: np.zeros(len(fragments_borders), dtype=np.float32))
    for user_markers in users_markers:
        for description, marker in user_markers.items():
            results[description] += marker / normalization_weight
    result = np.zeros(len(fragments_borders), dtype=np.float32)
    for description in results.keys():
        result += min_max_scale(results[description]) / len(results)
    return result


def build_average_or_integrated_interest_signal(book):
    users = list(get_good_users_info(book).keys())
    fragments_borders = get_split_text_borders(book)
    users_markers = [load_markers_for_user(book_id, user) for user in users]
    normalization_weight = calculate_numbers_of_readers_per_fragment(book, users, len(fragments_borders))
    result = np.zeros(len(fragments_borders), dtype=np.float32)
    for user_markers in users_markers:
        n_markers = len(user_markers)
        current_result = np.zeros_like(result, dtype=np.bool)
        for marker in user_markers.values():
            current_result = current_result | marker
        result += current_result
    return result / normalization_weight


def save_integrated_marker(book, integrated_marker, description):
    description = '_'.join(description.split(' '))
    output_path = os.path.join(get_integrated_marker_dumps_directory_path(book), f'{description}.pkl')
    save_via_pickle(integrated_marker, output_path)


def visualize_integrated_marker(integrated_marker, output_path, title=''):
    plt.clf()
    plt.rcParams["figure.figsize"] = (30, 20)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}
    plt.rc('font', **font)
    plt.plot(np.arange(integrated_marker.shape[0]), integrated_marker)
    title = ' '.join(title.split('_'))
    plt.title(title)
    plt.xlabel('Fragment')
    plt.ylabel('Integrated interest marker')
    plt.savefig(output_path, bbox_inches='tight')


def load_integrated_markers(book):
    dir_path = get_integrated_marker_dumps_directory_path(book)
    signals = {}
    for filename in os.listdir(dir_path):
        description = ' '.join((filename.split('.')[0]).split('_'))
        signals[description] = load_from_pickle(os.path.join(dir_path, filename))
    return signals


NAME_TO_RU = {'average interest signal': 'Интегрированный сигнал',
              'smoothed average scaled interest signal': 'Сглаженный взвешенный интегрированный сигнал',
              'smoothed average interest signal': 'Сглаженный интегрированный сигнал',
              'average scaled interest signal': 'Взвешенный интегрированный сигнал'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    all_mark_up = load_mark_up()
    print(all_mark_up.keys())
    all_markers = load_integrated_markers(book_id)
    # print(all_mark_up['action'])
    for marker_description, current_marker in all_markers.items():
        for smooth in [True]:  # False
            current_output_path = get_custom_path(book_id,
                                                  f'{marker_description}_ru.png' if not smooth else f'{marker_description}_smoothed_hot_scene_ru.png')
            custom_visualize_marker_with_mark_up(current_marker if not smooth else smooth_marker(current_marker),
                                                 {'g': all_mark_up['hot_scene'],
                                                  'r': all_mark_up['musing'] | all_mark_up['correspondence']},
                                                 # 'r': all_mark_up['correspondence'] | all_mark_up['musing'] |
                                                 #      all_mark_up['description']},
                                                 output_path=current_output_path,
                                                 title=NAME_TO_RU[marker_description],
                                                 x_label='Номер фрагмента',
                                                 y_label='Доля заинтересованных пользователей')
    # _plot_integrated_signal(book_id, build_integrated_interest_signal_best_manifestations(book_id),
    #                         os.path.join(_get_plots_directory_path(book_id), 'best_manifestations.png'))
    # _plot_integrated_signal(book_id, build_integrated_interest_signal_best_manifestations(book_id, std_weighted=True),
    #                         os.path.join(_get_plots_directory_path(book_id), 'best_manifestations_std_weighted.png'))
    # _plot_integrated_signal(book_id, build_average_interest_signal_best_manifestations(book_id),
    #                         os.path.join(_get_plots_directory_path(book_id), 'average_best_manifestations.png'))
    # _plot_integrated_signal(book_id, build_average_interest_signal_best_manifestations(book_id, std_weighted=True),
    #                         os.path.join(_get_plots_directory_path(book_id),
    #                                      'average_best_manifestations_std_weighted.png'))
    # average_signal = build_average_integrated_interest_signal(book_id)
    # save_integrated_marker(book_id, average_signal, 'average interest signal')
    # visualize_integrated_marker(average_signal, get_integrated_marker_plot_path(book_id,
    #                                                                             'average interest signal'),
    #                             'Average interest signal')
    # save_marker_stats(get_book_stats_path(book_id, 'average_interest_signal.csv'),
    #                   average_signal, 'Average interest signal')
    #
    # average_signal = smooth_marker(load_integrated_markers(book_id)['average interest signal'])
    # save_integrated_marker(book_id, average_signal, 'smoothed average interest signal')
    # visualize_integrated_marker(average_signal, get_integrated_marker_plot_path(book_id,
    #                                                                             'smoothed average interest signal'),
    #                             'Smoothed average interest signal')
    # save_marker_stats(get_book_stats_path(book_id, 'smoothed_average_interest_signal.csv'),
    #                   average_signal, 'Smoothed average interest signal')
    # average_signal = build_average_or_integrated_interest_signal(book_id)
    # save_integrated_marker(book_id, average_signal, 'average-or interest signal')
    # visualize_integrated_marker(average_signal, get_integrated_marker_plot_path(book_id,
    # 'average-or interest signal'),
    #                             'Average-or interest signal')
    # average_signal = build_average_scaled_integrated_interest_signal(book_id)
    # save_integrated_marker(book_id, average_signal, 'average scaled interest signal')
    # visualize_integrated_marker(average_signal, get_integrated_marker_plot_path(book_id,
    #                                                                             'average scaled interest '
    #                                                                             'signal'),
    #                             'Average scaled interest signal')
    # save_marker_stats(get_book_stats_path(book_id, 'average_scaled_interest_signal.csv'),
    #                   average_signal, 'Average scaled interest signal')
    #
    # average_signal = smooth_marker(load_integrated_markers(book_id)['average scaled interest signal'])
    # save_integrated_marker(book_id, average_signal, 'smoothed average scaled interest signal')
    # visualize_integrated_marker(average_signal, get_integrated_marker_plot_path(book_id,
    #                                                                             'smoothed average scaled interest '
    #                                                                             'signal'),
    #                             'Smoothed average scaled interest signal')
    # save_marker_stats(get_book_stats_path(book_id, 'smoothed_average_scaled_interest_signal.csv'),
    #                   average_signal, 'Smoothed average scaled interest signal')
    # print(load_integrated_markers(book_id).keys())
