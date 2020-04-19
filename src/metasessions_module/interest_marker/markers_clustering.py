import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.markers_visualization import get_re_reading_markers, \
    get_reading_interrupt_markers
from src.metasessions_module.text_utils import get_chapter_percents
from src.metasessions_module.config import *

from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'markers_clustering.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000


def transform_markers_to_bool(all_markers):
    new_markers = np.zeros(shape=all_markers.shape, dtype=np.bool)
    new_markers[all_markers != 0] = True
    return new_markers


def visualize_markers_clusters(book_id, plot_title, plot_name, all_markers, labels):
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.2)
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Users')
    ax.set_title(plot_title)
    ax.set_xlim(0.0, 100.0)
    batch_percent = 100.0 / markers.shape[1]
    ax.set_ylim(0, batch_percent * markers.shape[0])

    counter = 0
    old_counter = 0
    for label in np.unique(labels):
        cluster = []
        for user_ind, speeds in tqdm(enumerate(all_markers)):
            if labels[user_ind] != label:
                continue
            counter += 1
            cluster.append(all_markers[user_ind])
        cluster = greedy_reordering(np.array(cluster))
        for user_ind, values in enumerate(cluster):
            batch_from = batch_percent / 2
            users_y = batch_percent * (user_ind + old_counter) + batch_percent / 2
            for ind in range(batches_number):
                circle = plt.Circle((batch_from, users_y), batch_percent / 2,
                                    color='black' if values[ind] else 'w')
                ax.add_artist(circle)
                batch_from += batch_percent
        if counter != all_markers.shape[0]:
            ax.axhline(batch_percent * counter, color='y', linewidth=1)
        old_counter = counter

    chapters_lens = get_chapter_percents(book_id, DOCUMENTS[book_id][0])
    prev_len = 0
    ticks_pos = []
    for chapter_len in chapters_lens:
        ticks_pos.append((chapter_len + prev_len) / 2)
        prev_len = chapter_len
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(BOOK_LABELS[book_id], rotation=90)

    dir_path = os.path.join('resources', 'plots', 'markers_clusters', str(book_id))
    plot_path = os.path.join(dir_path, plot_name)
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(plot_path)


def greedy_reordering(markers_to_reorder):
    logging.info('Greedy reordering started')
    mask = []
    used = defaultdict(lambda: False)

    all_sums = markers_to_reorder.sum(axis=1, dtype=np.int64)
    first = np.argmax(all_sums)
    mask.append(first)
    used[first] = True

    diff = np.vectorize(lambda f, s : f != s)
    while len(mask) < markers_to_reorder.shape[0]:
        min_dist = None
        next_ind = -1

        for ind in range(markers_to_reorder.shape[0]):
            if not used[ind]:
                dist = np.sum(diff(markers_to_reorder[mask[-1]], markers_to_reorder[ind]), dtype=np.int64)
                if min_dist is None or min_dist > dist:
                    min_dist = dist
                    next_ind = ind

        mask.append(next_ind)
        used[next_ind] = True

    mask = np.array(mask)
    logging.info('Greedy reordering finished')
    return markers_to_reorder[mask, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reading_interrupt', help='Build reading interrupt marker plot with N batches', type=int)
    parser.add_argument('--min_interrupt_time_second', help='Interrupt time in seconds, default 600 (for '
                                                            'reading interrupt)', type=int)
    parser.add_argument('--book_id', help='Book id', type=int)
    parser.add_argument('--re_reading', help='Build re-reading marker plot with N batches', type=int)
    parser.add_argument('--min_delay_time', help='Interrupt time in seconds, default 600 (for re-reading)', type=int)
    args = parser.parse_args()

    book_id = 210901 if not args.book_id else args.book_id
    if args.reading_interrupt:
        interrupt_time = args.min_interrupt_time_second if args.min_interrupt_time_second else 600
        batches_number = args.reading_interrupt
        markers = np.array(get_reading_interrupt_markers(batches_number, interrupt_time, book_id))
        markers = transform_markers_to_bool(markers)
        for model_name, model in [# ('kmeans_2', KMeans(n_clusters=2, random_state=239239)),
                                  ('agglomerative_2', AgglomerativeClustering(n_clusters=2)),
                                  # ('kmeans_3', KMeans(n_clusters=3, random_state=239239)),
                                  # ('agglomerative_3', AgglomerativeClustering(n_clusters=3))
                                  ]:
            logging.info('Reading interrupt clustering started')
            labels = model.fit_predict(markers)
            logging.info('Reading interrupt clustering finished')
            visualize_markers_clusters(book_id, 'Book {} reading interrupt clusters'.format(book_id),
                                       'clusters_reading_interrupt_{}_{}_{}_bin'.format(model_name,
                                                                                        batches_number,
                                                                                        interrupt_time),
                                       markers, labels)
    if args.re_reading:
        delay_time = args.min_delay_time if args.min_delay_time else 600
        batches_number = args.re_reading
        markers = np.array(get_re_reading_markers(batches_number, delay_time, book_id))
        markers = transform_markers_to_bool(markers)
        for model_name, model in [# ('kmeans_2', KMeans(n_clusters=2, random_state=239239)),
                                  # ('agglomerative_2', AgglomerativeClustering(n_clusters=2)),
                                  # ('kmeans_3', KMeans(n_clusters=3, random_state=239239)),
                                  ('agglomerative_3', AgglomerativeClustering(n_clusters=3))
                                  ]:
            logging.info('Re-reading clustering started')
            labels = model.fit_predict(markers)
            logging.info('Re-reading clustering finished')
            visualize_markers_clusters(book_id, 'Book {} re-reading clusters'.format(book_id),
                                       'clusters_re_reading_{}_{}_{}_bin'.format(model_name,
                                                                                 batches_number,
                                                                                 delay_time),
                                       markers, labels)

