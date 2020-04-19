import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    _plot_integrated_signal(book_id, build_integrated_interest_signal_best_manifestations(book_id),
                            os.path.join(_get_plots_directory_path(book_id), 'best_manifestations.png'))
    _plot_integrated_signal(book_id, build_integrated_interest_signal_best_manifestations(book_id, std_weighted=True),
                            os.path.join(_get_plots_directory_path(book_id), 'best_manifestations_std_weighted.png'))
    _plot_integrated_signal(book_id, build_average_interest_signal_best_manifestations(book_id),
                            os.path.join(_get_plots_directory_path(book_id), 'average_best_manifestations.png'))
    _plot_integrated_signal(book_id, build_average_interest_signal_best_manifestations(book_id, std_weighted=True),
                            os.path.join(_get_plots_directory_path(book_id),
                                         'average_best_manifestations_std_weighted.png'))
