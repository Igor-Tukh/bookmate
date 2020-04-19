import os
import sys
import csv
import numpy as np
import warnings

from src.metasessions_module.interest_marker.config import MANIFESTATIONS_NUMBER_TO_MARKER_MANIFESTATION, \
    InterestMarker, MarkerManifestation

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))
warnings.filterwarnings('ignore')

from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn import metrics
from tqdm import tqdm
from src.metasessions_module.config import RANDOM_SEED
from src.metasessions_module.interest_marker.interest_marker_utils import load_saved_markers, \
    calculate_cumulative_markers, visualize_markers_manifestations
from src.metasessions_module.utils import array_is_trivial


def calculate_binary_marker_manifestations_above_percentile(marker, percentile=75):
    """
    Calculates boolean mask indicating marker positions above given percentile
    """

    return np.array(marker >= np.percentile(marker, percentile), dtype=np.bool)


def calculate_binary_marker_manifestations_kmeans_clustering(marker, clusters_number=2):
    """
    Cluster marker into @cluster_number different clusters. Returns mask of cluster with the biggest center
    """
    kmeans = KMeans(n_clusters=clusters_number, random_state=RANDOM_SEED).fit(marker.reshape(-1, 1))
    max_center = np.argmax(kmeans.cluster_centers_)
    return np.array(kmeans.labels_ == max_center, dtype=np.bool)


BINARY_CLUSTERING_APPROACHES = [(
    lambda m, copy=percentile: calculate_binary_marker_manifestations_above_percentile(m, copy),
    f'{percentile}-th percentile')
                                   for percentile in [50, 60, 70, 75, 80, 90, 95]] + [
                                   (lambda m, copy=n_clusters: calculate_binary_marker_manifestations_kmeans_clustering(
                                       m, copy),
                                    f'Kmeans with {n_clusters} clusters') for n_clusters in [2, 3, 4]]


def calculate_trinity_marker_manifestations_kmeans_clustering(marker, marker_type):
    kmeans = KMeans(n_clusters=3).fit(marker.reshape(-1, 1))
    center_labels = np.argsort(kmeans.cluster_centers_.reshape(-1))
    result = np.zeros(marker.shape[0], dtype=MarkerManifestation)
    for ind, center in enumerate(center_labels):
        result[kmeans.labels_ == center] = MANIFESTATIONS_NUMBER_TO_MARKER_MANIFESTATION[marker_type][ind]
    return result


def calculate_trinity_marker_manifestations_percentile(marker, marker_type, percent_lower=25, percent_higher=75):
    result = np.zeros(marker.shape[0], dtype=MarkerManifestation)
    lower = np.percentile(marker, percent_lower)
    higher = np.percentile(marker, percent_higher)
    lower_mask = marker <= lower
    higher_mask = higher <= marker if not np.isclose(lower, higher) else higher < marker
    middle_mask = ~(lower_mask | higher_mask)
    for ind, mask in enumerate([lower_mask,  middle_mask, higher_mask]):
        result[mask] = MANIFESTATIONS_NUMBER_TO_MARKER_MANIFESTATION[marker_type][ind]
    return result


TRINITY_DISCRETIZATION_APPROACHES = [
    (lambda marker, marker_description: calculate_trinity_marker_manifestations_kmeans_clustering(
        marker, InterestMarker(marker_description)), 'Kmeans with 3 clusters')] + [
    (lambda marker, marker_description, c=f: calculate_trinity_marker_manifestations_percentile(
        marker, InterestMarker(marker_description), c, 100 - c), f'{f}-th and {100 - f}-th percentile')
    for f in [5, 10, 15, 20, 25, 30]
]


def try_identify_marker_manifestations(marker, marker_description, approaches=None):
    """
    Performs different approaches to cluster marker into clusters.
    Evaluates the performance and returns list of results (each approach result represents as a dictionary).
    """
    if approaches is None:
        approaches = TRINITY_DISCRETIZATION_APPROACHES

    results = []
    for approach, approach_description in approaches:
        current_results = {'approach': approach_description}
        mask = approach(marker, marker_description)
        trivial = array_is_trivial(mask)
        current_results['Silhouette Coefficient'] = metrics.silhouette_score(marker.reshape(-1, 1), mask) \
            if not trivial else 'und'
        current_results['Calinski-Harabasz Index'] = metrics.calinski_harabasz_score(marker.reshape(-1, 1), mask) \
            if not trivial else 'und'
        current_results['Davies-Bouldin Index'] = metrics.davies_bouldin_score(marker.reshape(-1, 1), mask) \
            if not trivial else 'und'
        current_results['Number of -1'] = np.sum(np.array(mask == MarkerManifestation.NON_INTERESTING), dtype=np.int)
        current_results['Percent of -1'] = 1. * current_results['Number of -1'] / mask.shape[0]
        current_results['Number of 0'] = np.sum(np.array(mask == MarkerManifestation.NEUTRAL), dtype=np.int)
        current_results['Percent of 0'] = 1. * current_results['Number of 0'] / mask.shape[0]
        current_results['Number of 1'] = np.sum(np.array(mask == MarkerManifestation.INTERESTING), dtype=np.int)
        current_results['Percent of 1'] = 1. * current_results['Number of 1'] / mask.shape[0]
        current_results['Marker std'] = np.std(marker)
        current_results['Marker mean'] = np.mean(marker)
        current_results['mask'] = mask

        results.append(current_results)

    return results


def get_manifestations_clustering_results():
    all_results = []
    for book_id, markers in load_saved_markers().items():
        for marker_description, marker in markers.items():
            cumulative_marker = calculate_cumulative_markers(marker)
            results = try_identify_marker_manifestations(cumulative_marker, marker_description)
            for result in results:
                result['book_id'] = book_id
                result['marker'] = marker_description
            all_results.extend(results)

    return all_results


def get_individual_manifestations_clustering_results():
    all_results = defaultdict(lambda: [])
    for book_id, markers in load_saved_markers().items():
        for marker_description, marker in markers.items():
            for user_id, individual_marker in tqdm(marker.items()):
                results = try_identify_marker_manifestations(individual_marker, marker_description)
                for result in results:
                    result['book_id'] = book_id
                    result['marker'] = marker_description
                all_results[user_id].extend(results)

    return all_results


def save_manifestations_clustering_results(results_path):
    all_results = get_manifestations_clustering_results()

    if len(all_results) == 0:
        return

    for result in all_results:
        del result['mask']

    with open(results_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)


def visualize_individual_manifestations():
    for book_id, markers in load_saved_markers().items():
        for marker_description, marker in markers.items():
            approach_to_markers = defaultdict(lambda: [])
            for single_marker in tqdm(marker.values()):
                for result in try_identify_marker_manifestations(single_marker, marker_description):
                    approach_description = '_'.join(result['approach'].split(' ')).lower()
                    approach_to_markers[approach_description].append(result['mask'])
            for approach_description, approach_markers in approach_to_markers.items():
                dir_path = os.path.join('resources', 'plots', 'manifestations', marker_description)
                os.makedirs(dir_path, exist_ok=True)
                output_path = os.path.join(dir_path, f'{approach_description}.png')
                visualize_markers_manifestations(approach_markers, output_path,
                                                 f'Marker {marker_description} manifestations',
                                                 'Position',
                                                 'Level of manifestation')


if __name__ == '__main__':
    save_manifestations_clustering_results(os.path.join('resources', 'manifestations', 'trinity_scores.csv'))
    # visualize_individual_manifestations()
