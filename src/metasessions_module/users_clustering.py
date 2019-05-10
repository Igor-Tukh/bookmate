import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def get_batches(book_id, batches_amount):
    logging.info('Creating {} batches for users of book {}'.format(batches_amount, book_id))
    user_ids = list(get_good_users_info(book_id).keys())
    batches = np.full((len(user_ids), batches_amount), UNKNOWN_SPEED, dtype=np.float64)
    batch_percent = 100.0 / batches_amount

    for user_ind, user_id in tqdm(enumerate(user_ids)):
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


def cluster_users_by_batches_speed_sklearn(book_id, batches_amount, algo, scale, scale_each):
    batches, user_ids = load_batches(book_id, batches_amount)
    if scale_each:
        for speeds_ind in range(batches.shape[0]):
            scaler = MinMaxScaler(copy=True)
            batches[speeds_ind] = scaler.fit_transform(batches[speeds_ind].reshape(-1, 1)).reshape(-1)
    elif scale:
        scaler = MinMaxScaler(copy=False)
        scaler.fit_transform(batches)
    labels = algo.fit_predict(batches)
    return batches, labels


def cluster_users_by_batches_speed_sklearn_k_means(book_id, batches_amount, clusters_amount, scale=True,
                                                   scale_each=False,
                                                   random_state=23923):
    return cluster_users_by_batches_speed_sklearn(book_id,
                                                  batches_amount,
                                                  KMeans(n_clusters=clusters_amount,
                                                         random_state=random_state,
                                                         max_iter=1000,
                                                         init='random'),
                                                  scale, scale_each)


def cluster_users_by_batches_speed_sklearn_agglomerative(book_id, batches_amount, clusters_amount, scale=True,
                                                         scale_each=False):
    return cluster_users_by_batches_speed_sklearn(book_id,
                                                  batches_amount,
                                                  AgglomerativeClustering(n_clusters=clusters_amount),
                                                  scale, scale_each)


def cluster_users_by_batches_speed_sklearn_spectral(book_id, batches_amount, clusters_amount, scale=True,
                                                    scale_each=False):
    return cluster_users_by_batches_speed_sklearn(book_id,
                                                  batches_amount,
                                                  SpectralClustering(n_clusters=clusters_amount),
                                                  scale, scale_each)


def visualize_batches_speed_clusters(book_id, batches, labels, plot_title, plot_name, colors):
    batches_amount = batches.shape[1]
    indexes = np.argsort(labels)
    batches = batches[indexes, :]
    colors = colors[indexes, :]
    labels = labels[indexes]
    fig, ax = plt.subplots(figsize=(14, 14))
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

    plot_path = os.path.join('resources', 'plots', 'batches_clusters', plot_name)
    plt.savefig(plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', help='cluster to N clusters', type=int, metavar='N')
    parser.add_argument('--n_batches', help='split to N batches', type=int, metavar='N')
    parser.add_argument('--algorithm', help='Apply algorithm algo', type=str, metavar='algo')
    parser.add_argument('--search', help='Run with a different clustering configurations', action='store_true')
    args = parser.parse_args()

    n_clusters = args.n_clusters if args.n_clusters is not None else 5
    n_batches = args.n_batches if args.n_batches is not None else 100
    algorithm = args.algorithm
    algorithm = 'k_means' if algorithm is None else algorithm

    search_range = ([[n_batches], [n_clusters], [algorithm]]) if not args.search else \
        [[200], [8, 10], ['agglomerative', 'spectral', 'k_means']]
    for n_batches in search_range[0]:
        for n_clusters in search_range[1]:
            for algorithm in search_range[2]:
                for book in BOOKS.items():
                    if algorithm == 'agglomerative':
                        book_batches, book_labels = cluster_users_by_batches_speed_sklearn_agglomerative(book[1],
                                                                                                         n_batches,
                                                                                                         n_clusters,
                                                                                                         scale_each=True)
                    elif algorithm == 'spectral':
                        book_batches, book_labels = cluster_users_by_batches_speed_sklearn_spectral(book[1],
                                                                                                    n_batches,
                                                                                                    n_clusters,
                                                                                                    scale_each=True)
                    else:
                        book_batches, book_labels = cluster_users_by_batches_speed_sklearn_k_means(book[1],
                                                                                                   n_batches,
                                                                                                   n_clusters,
                                                                                                   scale_each=True)
                    plot_name = '{}_{}_{}_{}'.format(book[1], n_clusters, n_batches, algorithm)
                    book_colors = get_colors_speed_using_users_min_max_scale(book_batches)
                    visualize_batches_speed_clusters(book[1], book_batches, book_labels, 'Book {} readers clusters'.format(book[0]),
                                                     plot_name,
                                                     book_colors)
