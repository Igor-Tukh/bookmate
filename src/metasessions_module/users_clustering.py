import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id, \
    get_user_sessions_by_place_in_book
from src.metasessions_module.sessions_utils import is_target_speed
from src.metasessions_module.utils import save_via_pickle, load_from_pickle
from src.metasessions_module.text_utils import get_chapter_percents
from src.metasessions_module.config import UNKNOWN_SPEED, DOCUMENTS, BOOK_LABELS, BOOKS
from src.metasessions_module.speed_to_color import to_matplotlib_color, \
    get_colors_speed_using_absolute_min_max_scale, get_colors_speed_using_users_min_max_scale
from src.metasessions_module.batches_sorting_utils import AnnealingBatchesSorter

from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from collections import defaultdict

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'users_clustering.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000


def get_batches(book_id, batches_amount):
    logging.info('Creating {} batches for users of book {}'.format(batches_amount, book_id))
    user_ids = list(get_good_users_info(book_id).keys())
    batches = np.full((len(user_ids), batches_amount), UNKNOWN_SPEED, dtype=np.float64)
    batch_percent = 100.0 / batches_amount

    for (user_ind, user_id), _ in zip(enumerate(user_ids), tqdm(range(len(user_ids)))):
        batch_to = batch_percent
        batch_ind = 0
        current_speeds = []

        document_id = get_user_document_id(book_id, user_id)
        unique_sessions = get_user_sessions_by_place_in_book(book_id, document_id, user_id)

        places = list(unique_sessions.keys())
        places.sort(key=lambda val: val[0])
        for place in places:
            session = unique_sessions[place]
            book_from, book_to = place
            while book_from > batch_to and batch_ind < batches_amount:
                if len(current_speeds) > 0:
                    batches[user_ind][batch_ind] = sum(current_speeds) / len(current_speeds)
                    current_speeds = []
                batch_to, batch_ind = batch_to + batch_percent, batch_ind + 1
            if batches_amount > batch_ind and is_target_speed(session['speed']):
                current_speeds.append(session['speed'])
        if len(current_speeds) > 0:
            batches[user_ind][batch_ind] = sum(current_speeds) / len(current_speeds)

    return batches, user_ids


def load_batches(book_id, batches_amount, rebuild=False):
    """
    Uploads batches file for certain book id and batches amount if it exists and flag rebuild is not selected.
    Otherwise splits all user sessions to batches_amount sequent slices.
    :return: tuple of np.array of shape (users_cnt, batches_amount) and np.array of shape (users_cnt) -- batches average
    speeds and user ids respectively
    """
    output_batches_path = os.path.join('resources', 'batches', '{}_{}.pkl'.format(book_id, batches_amount))
    if rebuild or not os.path.isfile(output_batches_path):
        batches_and_users = get_batches(book_id, batches_amount)
        save_via_pickle(batches_and_users, output_batches_path)
        return batches_and_users
    return load_from_pickle(output_batches_path)


def cluster_users_by_batches_speed_sklearn(book_id, batches_amount, algo,
                                           scale,
                                           scale_each,
                                           with_ids=False,
                                           return_old_batches=False):
    batches, user_ids = load_batches(book_id, batches_amount)
    old_batches = batches.copy()
    if scale_each:
        for speeds_ind in range(batches.shape[0]):
            scaler = MinMaxScaler(copy=True)
            batches[speeds_ind] = scaler.fit_transform(batches[speeds_ind].reshape(-1, 1)).reshape(-1)
    elif scale:
        scaler = MinMaxScaler(copy=False)
        scaler.fit_transform(batches)
    labels = algo.fit_predict(batches)
    if return_old_batches:
        batches = old_batches
    if with_ids:
        return batches, labels, user_ids
    return batches, labels


def cluster_users_by_batches_speed_sklearn_k_means(book_id, batches_amount, clusters_amount, scale=True,
                                                   scale_each=False,
                                                   random_state=23923,
                                                   with_ids=False,
                                                   return_old_batches=False):
    return cluster_users_by_batches_speed_sklearn(book_id,
                                                  batches_amount,
                                                  KMeans(n_clusters=clusters_amount,
                                                         random_state=random_state,
                                                         max_iter=1000,
                                                         init='random'),
                                                  scale, scale_each, with_ids, return_old_batches)


def cluster_users_by_batches_speed_sklearn_agglomerative(book_id, batches_amount, clusters_amount, scale=True,
                                                         scale_each=False,
                                                         with_ids=False,
                                                         return_old_batches=False):
    return cluster_users_by_batches_speed_sklearn(book_id,
                                                  batches_amount,
                                                  AgglomerativeClustering(n_clusters=clusters_amount),
                                                  scale, scale_each, with_ids, return_old_batches)


def cluster_users_by_batches_speed_sklearn_spectral(book_id, batches_amount, clusters_amount, scale=True,
                                                    scale_each=False,
                                                    with_ids=False,
                                                    return_old_batches=False):
    return cluster_users_by_batches_speed_sklearn(book_id,
                                                  batches_amount,
                                                  SpectralClustering(n_clusters=clusters_amount),
                                                  scale, scale_each, with_ids, return_old_batches)


def save_clusters(book_id, clusters, filename, resave=False):
    output_file = os.path.join('resources', 'clusters', str(book_id), filename)
    if os.path.isfile(output_file) and not resave:
        logging.info('Clusters for {} have been already saved'.format(filename))
    else:
        save_via_pickle(clusters, output_file)


def load_clusters(book_id, filename):
    return load_from_pickle(os.path.join('resources', 'clusters', str(book_id), filename))


def visualize_batches_speed_clusters(book_id, batches, labels, plot_title, plot_name, colors):
    batches_amount = batches.shape[1]
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.2)
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Users')
    ax.set_title(plot_title)
    ax.set_xlim(0.0, 100.0)
    batch_percent = 100.0 / batches_amount
    ax.set_ylim(batch_percent * batches.shape[0] + batch_percent / 2)

    for user_ind, speeds in tqdm(enumerate(batches)):
        if user_ind != 0 and labels[user_ind] != labels[user_ind - 1]:
            ax.axhline(y=batch_percent * user_ind, color='black', markersize=10)
        batch_from = batch_percent / 2
        for ind in range(batches_amount):
            users_y = batch_percent * user_ind + batch_percent / 2
            circle = plt.Circle((batch_from, users_y), batch_percent / 2,
                                color=to_matplotlib_color(colors[user_ind][ind]))
            ax.add_artist(circle)
            batch_from += batch_percent

    chapters_lens = get_chapter_percents(book_id, DOCUMENTS[book_id][0])
    prev_len = 0
    ticks_pos = []
    for chapter_len in chapters_lens:
        ticks_pos.append((chapter_len + prev_len) / 2)
        prev_len = chapter_len
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(BOOK_LABELS[book_id], rotation=90)

    plot_path = os.path.join('resources', 'plots', 'batches_clusters', str(book_id), plot_name)
    plt.savefig(plot_path)


def sort_by_batches_by_labels(batches, labels):
    indexes = np.argsort(labels)
    return batches[indexes], labels[indexes], indexes


def get_clusters_boundaries(batches, labels):
    prev_label = labels[0]
    boundaries = [0]
    for ind in range(1, labels.shape[0]):
        if labels[ind] != prev_label:
            prev_label = labels[ind]
            boundaries.append(ind)
    return boundaries


def get_scores(X, y):
    return {
        'Silhouette Coefficient': metrics.silhouette_score(X, y, metric='euclidean'),  # [-1; 1] 1 for highly dense
        'Calinski-Harabasz Index': metrics.calinski_harabasz_score(X, y),  # higher score relates to better clusters def
        'Davies-Bouldin Index': metrics.davies_bouldin_score(X, y)  # >= 0, closer to 0 indicates better partition
    }


def get_scores_path():
    return os.path.join('resources', 'scores', 'users_clustering', 'scores.csv')


def save_scores(scores):
    if len(scores) == 0:
        return
    with open(get_scores_path(), 'w') as scores_file:
        logging.info('Saving users clustering scores to {}'.format(get_scores_path()))
        writer = csv.DictWriter(scores_file, scores[0].keys())
        writer.writeheader()
        writer.writerows(scores)


def load_scores():
    if not os.path.isfile(get_scores_path()):
        logging.info('Early scores not found')
        return []

    with open(get_scores_path(), 'r') as scores_file:
        reader = csv.reader(scores_file)
        lines = [row for row in reader]
        scores = [{lines[0][i]: line[i] for i in range(len(line))} for line in lines[1:]]

    return scores


def extend_scores(scores):
    logging.info('Extending users clustering scores')
    if not os.path.isfile(get_scores_path()):
        logging.info('Early scores not found')
        save_scores(scores)
        return
    scores.extend(load_scores())
    save_scores(scores)


def get_gender_stats_path():
    return os.path.join('resources', 'stats', 'users_clustering', 'gender_stats.csv')


def load_gender_stats():
    stats_path = get_gender_stats_path()
    if not os.path.isfile(stats_path):
        logging.info('Early scores not found')
        return []

    with open(stats_path, 'r') as stats_file:
        reader = csv.reader(stats_file)
        lines = [row for row in reader]
        stats = [{lines[0][i]: line[i] for i in range(len(line))} for line in lines[1:]]

    return stats


def save_gender_stats(stats):
    stats_path = get_gender_stats_path()
    if os.path.isfile(stats_path):
        stats.extend(load_gender_stats())

    with open(stats_path, 'w') as stats_file:
        logging.info('Saving users clustering gender stats to {}'.format(stats_path))
        writer = csv.DictWriter(stats_file, stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', help='cluster to N clusters', type=int, metavar='N')
    parser.add_argument('--n_batches', help='split to N batches', type=int, metavar='N')
    parser.add_argument('--algorithm', help='Apply algorithm algo', type=str, metavar='algo')
    parser.add_argument('--search', help='Run with a different clustering configurations', action='store_true')
    parser.add_argument('--save_clusters', help='Save clusters', action='store_true')
    parser.add_argument('--scores', help='Run with a different clustering configurations ans score clusters',
                        action='store_true')
    parser.add_argument('--count_genders', help='Run with a different clustering configurations count genders ratio',
                        action='store_true')
    parser.add_argument('--print_common_users', help='Print common users of book',
                        action='store_true')

    args = parser.parse_args()

    if args.print_common_users:
        user_count = defaultdict(lambda: 0)
        for book_id in BOOKS.values():
            for user_id in list(get_good_users_info(book_id).keys()):
                user_count[user_id] += 1
        count = 0
        for user_id, value in user_count.items():
            if value == len(BOOKS):
                count += 1
        print(count)
        exit(0)

    n_clusters = args.n_clusters if args.n_clusters is not None else 5
    n_batches = args.n_batches if args.n_batches is not None else 100
    algorithm = args.algorithm
    algorithm = 'k_means' if algorithm is None else algorithm

    clustering_scores = []
    gender_stats = []
    search_range = ([[n_batches], [n_clusters], [algorithm]]) if not args.search and not args.scores and not \
        args.count_genders else [[300], [2], ['agglomerative', 'spectral', 'k_means']]

    for n_batches in search_range[0]:
        for n_clusters in search_range[1]:
            for algorithm in search_range[2]:
                # for book in BOOKS.items():
                for book in [('Fifty Shades of Grey', 210901)]:
                    info = get_good_users_info(book[1])
                    if algorithm == 'agglomerative':
                        book_batches, book_labels, book_user_ids = \
                            cluster_users_by_batches_speed_sklearn_agglomerative(book[1],
                                                                                 n_batches,
                                                                                 n_clusters,
                                                                                 scale_each=True,
                                                                                 with_ids=True,
                                                                                 return_old_batches=True)
                    elif algorithm == 'spectral':
                        book_batches, book_labels, book_user_ids = \
                            cluster_users_by_batches_speed_sklearn_spectral(book[1],
                                                                            n_batches,
                                                                            n_clusters,
                                                                            scale_each=True,
                                                                            with_ids=True,
                                                                            return_old_batches=True)
                    else:
                        book_batches, book_labels, book_user_ids = \
                            cluster_users_by_batches_speed_sklearn_k_means(book[1],
                                                                           n_batches,
                                                                           n_clusters,
                                                                           scale_each=True,
                                                                           with_ids=True,
                                                                           return_old_batches=True)
                    if args.scores:
                        current_scores = get_scores(book_batches, book_labels)
                        current_scores['Model'] = algorithm
                        current_scores['Batches amount'] = n_batches
                        current_scores['Clusters amount'] = n_clusters
                        current_scores['Book'] = book[0]
                        clustering_scores.append(current_scores)

                    book_batches, book_labels, indexes = sort_by_batches_by_labels(book_batches, book_labels)
                    book_user_ids = np.array(book_user_ids)
                    book_user_ids = book_user_ids[indexes]
                    boundaries = get_clusters_boundaries(book_batches, book_labels)
                    sorted_batches = None
                    sorted_labels = None
                    current_clusters = []
                    for ind, boundary in enumerate(boundaries):
                        next_boundary = boundaries[ind + 1] if ind < len(boundaries) - 1 else book_labels.shape[0]

                        sorter = AnnealingBatchesSorter(book_batches[boundary:next_boundary],
                                                        book_labels[boundary:next_boundary],
                                                        initial_temperature=100.0,
                                                        min_temperature=0.01,
                                                        random_state=23923)
                        current_batches, current_labels, permutation = sorter.get_sorted_batches_and_labels()
                        ids = book_user_ids[boundary:next_boundary].copy()
                        ids = ids[permutation]

                        if args.count_genders:
                            males_amount = len([user_id for user_id in ids if info[user_id]['gender'] == 'm'])
                            females_amount = len([user_id for user_id in ids if info[user_id]['gender'] == 'f'])
                            current_stats = {
                                'Model': '{}_{}_{}_{}'.format(book[1], n_clusters, n_batches, algorithm),
                                'Cluster size': next_boundary - boundary,
                                'Males': males_amount,
                                'Females': females_amount,
                                'Females percent':
                                    1. * females_amount / (
                                            males_amount + females_amount) if males_amount + females_amount > 0 else 0.
                            }

                            gender_stats.append(current_stats)

                        current_clusters.append((current_batches, ids))
                        if sorted_batches is None:
                            sorted_batches, sorted_labels = current_batches, current_labels
                        else:
                            sorted_batches = np.vstack([sorted_batches, current_batches])
                            sorted_labels = np.hstack([sorted_labels, current_labels])

                    if args.save_clusters:
                        save_clusters(book[1], current_clusters, '{}_{}_{}.pkl'.format(n_clusters,
                                                                                       n_batches,
                                                                                       algorithm))
                    if args.scores or args.search:
                        plot_name = '{}_{}_{}_{}_annealing'.format(book[1], n_clusters, n_batches, algorithm)
                        book_colors = get_colors_speed_using_users_min_max_scale(sorted_batches)
                        visualize_batches_speed_clusters(book[1], sorted_batches, sorted_labels,
                                                         'Book {} readers clusters'.format(book[0]),
                                                         plot_name,
                                                         book_colors)
    if args.scores:
        extend_scores(clustering_scores)
    if args.count_genders:
        save_gender_stats(gender_stats)
