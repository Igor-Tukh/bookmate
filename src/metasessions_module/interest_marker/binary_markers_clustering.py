import os
import sys
import csv
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.metasessions_module.interest_marker.config import get_stats_path
from src.metasessions_module.interest_marker.interest_marker_utils import load_markers_for_user, get_binary_color, \
    get_read_fragments, get_fragment
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id
from src.metasessions_module.config import RANDOM_SEED
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

CLUSTERING_METRICS = {'Silhouette Coefficient': metrics.silhouette_score,
                      'Calinski-Harabasz Index': metrics.calinski_harabasz_score,
                      'Davies-Bouldin Index': metrics.davies_bouldin_score}

CLUSTERING_METHODS = {
    # **{f'kmeans with {i} clusters': KMeans(n_clusters=i, random_state=RANDOM_SEED) for i in range(2, 6)},
    **{f'agglomerative with {i} clusters': KMeans(n_clusters=i) for i in range(2, 6)}}


def get_clustering_stats_path(book, marker_description):
    dir_path = os.path.join(get_stats_path(), str(book), 'clustering')
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f'{marker_description}.csv')


def try_sklearn_clustering_approach(markers, approach):
    marker_size = len(markers)
    model = approach.fit(markers)
    clusters = np.array([np.arange(marker_size)[model.labels_ == label] for label in np.unique(model.labels_)])
    scores = {description: metrics_method(markers, model.labels_)
              for description, metrics_method in CLUSTERING_METRICS.items()}
    return clusters, scores


def visualize_binary_markers_clusters(markers, clusters, output_path, title='Marker clusters'):
    markers = np.vstack([markers[cluster] for cluster in clusters])

    plt.clf()
    plt.axis('off')
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.2)
    ax.set_xlabel('Fragments')
    ax.set_ylabel('Users')
    ax.set_title(title)

    current_len = 0
    for cluster in clusters:
        current_len += len(cluster)
        ax.axhline(current_len, color='black', linewidth=1)

    if len(markers) != 0:
        n = markers[0].shape[0]
        ax.set_xlim(0, n)
        ax.set_ylim(0, len(markers))
        for ind, one_marker in enumerate(markers):
            for place_ind in range(n):
                circle = plt.Rectangle((place_ind, ind), 1, 1, color=get_binary_color(one_marker[place_ind]))
                ax.add_artist(circle)

    plt.savefig(output_path)


def get_clustering_plot_path(book, description, filename):
    filename = '_'.join(filename.split(' '))
    dir_path = os.path.join('resources', 'plots', str(book), 'clustering', description)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f'{filename}.png')


def try_sklearn_clustering(book, markers, marker_description):
    logging.info(f'Clustering of marker {marker_description} started')

    results = []
    for method_description, clustering_method in CLUSTERING_METHODS.items():
        method_clusters, method_scores = try_sklearn_clustering_approach(markers, clustering_method)
        current_results = {'Method': method_description}
        current_results.update(method_scores)
        results.append(current_results)
        visualize_binary_markers_clusters(markers, method_clusters, get_clustering_plot_path(book,
                                                                                             marker_description,
                                                                                             method_description))

    with open(get_clustering_stats_path(book, marker_description), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    logging.info(f'Clustering of marker {marker_description} ended')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    users = list(get_good_users_info(book_id).keys())
    users_markers = [load_markers_for_user(book_id, user) for user in users]

    if len(users_markers) == 0:
        exit(0)

    for marker in list(users_markers[0].keys())[2:3]:
        marker_values = np.array([user_markers[marker] for user_markers in users_markers])
        try_sklearn_clustering(book_id, marker_values, marker)
