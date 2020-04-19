"""
Important: deprecated.
"""

import argparse
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt


from src.metasessions_module.interest_marker.old_quit_marker import OldQuitMarker
from src.metasessions_module.interest_marker.re_reading_marker import ReReadingMarker
from src.metasessions_module.interest_marker.reading_interrupt_marker import ReadingInterruptMarker
from src.metasessions_module.interest_marker.speed_marker import SpeedMarker
from src.metasessions_module.interest_marker.unusual_reading_hours_marker import UnusualReadingHoursMarker
from src.metasessions_module.text_utils import get_chapter_percents, get_split_text_borders
from matplotlib import gridspec

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id, load_users
from src.metasessions_module.config import *
from src.metasessions_module.utils import save_via_pickle, load_from_pickle
from tqdm import tqdm

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'markers_visualization.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000


def visualize_markers(book_id, plot_title, plot_name, markers):
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.2)
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Users')
    ax.set_title(plot_title)
    ax.set_xlim(0.0, 100.0)
    batches_number = markers.shape[1]
    batch_percent = 100.0 / markers.shape[1]
    ax.set_ylim(0, batch_percent * markers.shape[0])

    for user_ind, speeds in tqdm(enumerate(markers)):
        batch_from = batch_percent / 2
        users_y = batch_percent * user_ind + batch_percent / 2
        for ind in range(batches_number):
            circle = plt.Circle((batch_from, users_y), batch_percent / 2,
                                color='b' if markers[user_ind][ind] == 0 else 'r')
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

    dir_path = os.path.join('resources', 'plots', 'markers', str(book_id))
    plot_path = os.path.join(dir_path, plot_name)
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(plot_path)


def visualize_markers_numbers(book_id, plot_title, plot_name, markers, users_number=False, plot_extremums=10):
    total_markers = np.zeros(markers.shape[1])
    for user_markers in markers:
        for ind, number in enumerate(user_markers):
            total_markers[ind] += number if not users_number else min(1, number)

    if plot_extremums != 0:
        maxs = []
        mins = []
        for ind in range(total_markers.shape[0]):
            max_value = total_markers[
                        max(ind - plot_extremums, 0):
                        min(ind + plot_extremums, total_markers.shape[0])].max()
            min_value = total_markers[
                        max(ind - plot_extremums, 0):
                        min(ind + plot_extremums, total_markers.shape[0])].min()
            if total_markers[ind] == max_value:
                maxs.append(ind)
            if total_markers[ind] == min_value:
                mins.append(ind)

    fig, ax = plt.subplots(figsize=(20, 20))
    fig.subplots_adjust(bottom=0.3)
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Users')
    ax.set_title(plot_title)
    lent = total_markers.shape[0]
    ax.set_xlim(0.0, lent)
    ax.set_ylim(0, total_markers.max() * 1.1)

    if plot_extremums != 0:
        for max_pos in maxs:
            ax.axvline(max_pos + 0.2, linestyle='--', color='r', linewidth=0.6)

        for min_pos in mins:
            ax.axvline(min_pos + 0.2, linestyle='--', color='g', linewidth=0.6)

    ax.bar(np.arange(total_markers.shape[0]) + 0.5, total_markers, width=1, align='center')

    chapters_lens = get_chapter_percents(book_id, DOCUMENTS[book_id][0])
    prev_len = 0
    ticks_pos = []
    for chapter_len in chapters_lens:
        ticks_pos.append((chapter_len + prev_len) / 2)
        prev_len = chapter_len
    ax.set_xticks(np.array(ticks_pos) * lent / 100.0)
    ax.set_xticklabels(BOOK_LABELS[book_id], rotation=90)
    for chapter_len in chapters_lens:
        ax.axvline(chapter_len * markers.shape[0] / 100.0, linestyle='--', color='black', linewidth=0.6)

    dir_path = os.path.join('resources', 'plots', 'markers', str(book_id))
    plot_path = os.path.join(dir_path, plot_name)
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(plot_path)


def visualize_combine_markers_numbers(book_id, plot_title, plot_name, all_markers, reverse_markers, users_number=False,
                                      scales=None, plot_extremums=False, common_extremums=False,
                                      plot_chapters_borders=False):
    if scales is None:
        scales = [2., 1.1]
    max_poses = np.zeros(all_markers[0].shape[1])
    min_poses = np.zeros(all_markers[0].shape[1])
    markers_number = len(all_markers)
    total_markers = [np.zeros(all_markers[0].shape[1]) for _ in range(markers_number)]
    for marker_ind, (reverse, markers) in enumerate(zip(reverse_markers, all_markers)):
        for user_markers in markers:
            for ind, number in enumerate(user_markers):
                total_markers[marker_ind][ind] += number if not users_number else min(1, number)
        if reverse:
            max_value = total_markers[marker_ind].max()
            for ind in range(total_markers[marker_ind].shape[0]):
                total_markers[marker_ind][ind] = max_value - total_markers[marker_ind][ind]

    for marker_ind in range(len(total_markers)):
        for ind in range(total_markers[marker_ind].shape[0]):
            if plot_extremums:
                max_value = total_markers[marker_ind][
                            max(ind - plot_extremums, 0):
                            min(ind + plot_extremums, total_markers[marker_ind].shape[0])].max()
                min_value = total_markers[marker_ind][
                            max(ind - plot_extremums, 0):
                            min(ind + plot_extremums, total_markers[marker_ind].shape[0])].min()
                if total_markers[marker_ind][ind] == max_value:
                    max_poses[ind] += 1
                if total_markers[marker_ind][ind] == min_value:
                    min_poses[ind] += 1

    max_values = [total_marker.max() for total_marker in total_markers]

    maxs = []
    mins = []

    for ind in range(total_markers[0].shape[0]):
        threshold = len(total_markers) if common_extremums else 1
        if max_poses[ind] >= threshold:
            maxs.append(ind)
        if min_poses[ind] >= threshold:
            mins.append(ind)

    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(markers_number, 1, height_ratios=[1, 1])
    fig.subplots_adjust(bottom=0.2, hspace=.0)

    chapters_lens = get_chapter_percents(book_id, DOCUMENTS[book_id][0])
    for ind, markers in enumerate(total_markers):
        ax = plt.subplot(gs[ind])
        ax.set_xlabel('Book percent')
        ax.set_ylabel('Users')
        ax.set_xlim(0.0, markers.shape[0])
        ax.set_ylim(0, scales[ind] * max_values[ind])

        if plot_extremums:
            for max_pos in maxs:
                ax.axvline(max_pos + 0.2, linestyle='--', color='r', linewidth=0.6)

            for min_pos in mins:
                ax.axvline(min_pos + 0.2, linestyle='--', color='g', linewidth=0.6)

        if plot_chapters_borders:
            for chapter_len in chapters_lens:
                ax.axvline(chapter_len * markers.shape[0] / 100.0, linestyle='--', color='black', linewidth=0.6)

        ax.bar(np.arange(markers.shape[0]) + 0.5, markers, width=1, align='center')

        if ind == 0:
            ax.set_title(plot_title)
        if ind == len(total_markers) - 1:
            prev_len = 0
            ticks_pos = []
            for chapter_len in chapters_lens:
                ticks_pos.append((chapter_len + prev_len) / 2)
                prev_len = chapter_len
            ax.set_xticks(np.array(ticks_pos) * markers.shape[0] / 100.0)
            ax.set_xticklabels(BOOK_LABELS[book_id], rotation=90)

    dir_path = os.path.join('resources', 'plots', 'markers', str(book_id))
    plot_path = os.path.join(dir_path, plot_name)
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(plot_path)


def build_reading_interrupt_markers(batches_number, interrupt_time, book_id):
    user_ids = list(get_good_users_info(book_id).keys())
    markers = []
    for user_id in tqdm(user_ids):
        document_id = get_user_document_id(book_id, user_id)
        markers.append(ReadingInterruptMarker.get_for_user(book_id, document_id, user_id, batches_number,
                                                           interrupt_skip_seconds=interrupt_time,
                                                           only_consequent=False))
    return np.array(markers)


def get_reading_interrupt_markers_path(batches_number, interrupt_time, book_id):
    return os.path.join('resources', 'markers',
                        'reading_interrupt_{}_{}_{}.pkl'.format(book_id, batches_number, interrupt_time))


def save_reading_interrupt_markers(batches_number, interrupt_time, book_id):
    logging.info('Saving re reading numbers with {} batches and interrupt time {}'.format(batches_number,
                                                                                          interrupt_time))
    save_via_pickle(build_reading_interrupt_markers(batches_number, interrupt_time, book_id),
                    get_reading_interrupt_markers_path(batches_number, interrupt_time, book_id))


def get_reading_interrupt_markers(batches_number, interrupt_time, book_id):
    pkl_path = get_reading_interrupt_markers_path(batches_number, interrupt_time, book_id)
    if not os.path.exists(pkl_path):
        save_reading_interrupt_markers(batches_number, interrupt_time, book_id)
    return load_from_pickle(pkl_path)


def build_re_reading_markers(batches_number, delay_time, book_id):
    user_ids = list(get_good_users_info(book_id).keys())
    markers = []
    for user_id in tqdm(user_ids):
        document_id = get_user_document_id(book_id, user_id)
        markers.append(ReReadingMarker.get_for_user(book_id, document_id, user_id, batches_number, delay_time))
    return np.array(markers)


def get_re_reading_markers_path(batches_number, delay_time, book_id):
    return os.path.join('resources', 'markers',
                        're_reading_{}_{}_{}.pkl'.format(book_id, batches_number, delay_time))


def save_re_reading_markers(batches_number, delay_time, book_id):
    logging.info('Saving re reading numbers with {} batches and interrupt time {}'.format(batches_number,
                                                                                          delay_time))
    save_via_pickle(build_re_reading_markers(batches_number, delay_time, book_id),
                    get_re_reading_markers_path(batches_number, delay_time, book_id))


def get_re_reading_markers(batches_number, delay_time, book_id):
    pkl_path = get_re_reading_markers_path(batches_number, delay_time, book_id)
    if not os.path.exists(pkl_path):
        save_re_reading_markers(batches_number, delay_time, book_id)
    return load_from_pickle(pkl_path)


if __name__ == '__main__':
    # 300 600 1800
    # 60 120 300
    parser = argparse.ArgumentParser()
    parser.add_argument('--reading_interrupt', help='Build reading interrupt marker plot with N batches', type=int,
                        metavar='N')
    parser.add_argument('--min_interrupt_time_second', help='Interrupt time in seconds, default 600 (for '
                                                            'reading interrupt)', type=int,
                        metavar='N')
    parser.add_argument('--unusual_reading_hours', action='store_true', help='Build urh marker plots')
    parser.add_argument('--reverse_interrupts', action='store_true')
    parser.add_argument('--re_reading', help='Build re-reading marker plot with N batches', type=int,
                        metavar='N')
    parser.add_argument('--min_delay_time', help='Interrupt time in seconds, default 600 (for re-reading)', type=int,
                        metavar='N')
    parser.add_argument('--visualize_all', help='Visualize provided markers', action='store_true')
    parser.add_argument('--plot_extremums', type=int)
    parser.add_argument('--plot_chapters_borders', action='store_true')
    parser.add_argument('--users_number', action='store_true')
    parser.add_argument('--book_id', help='Book id', type=int)
    parser.add_argument('--common_extremums', action='store_true')
    parser.add_argument('--quit_markers', action='store_true')
    parser.add_argument('--speed', action='store_true')
    args = parser.parse_args()

    book_id = 210901 if not args.book_id else args.book_id
    all_markers = []
    reverse_markers = []
    combine_name = str(book_id)
    combine_title = 'Book {}'.format(book_id)
    if args.reading_interrupt:
        interrupt_time = args.min_interrupt_time_second if args.min_interrupt_time_second else 600
        batches_number = args.reading_interrupt
        markers = get_reading_interrupt_markers(batches_number, interrupt_time, book_id)
        all_markers.append(markers)
        combine_name += '_reading_interrupt_{}_{}'.format(batches_number, interrupt_time)
        combine_title += ' interrupts (at least {} seconds)'.format(interrupt_time)
        reverse_markers.append(args.reverse_interrupts is not None)
        # visualize_markers(book_id,
        #                   'Book {} interrupts for at least {} seconds'.format(book_id, interrupt_time),
        #                   'interrupts_{}_{}.png'.format(batches_number, interrupt_time),
        #                   np.array(markers))
        visualize_markers_numbers(book_id,
                                  'Book {} interrupts for at least {} seconds'.format(book_id, interrupt_time),
                                  'interrupts_total_{}_{}.png'.format(batches_number, interrupt_time),
                                  np.array(markers))
        visualize_markers_numbers(book_id,
                                  'Book {} interrupts for at least {} seconds'.format(book_id, interrupt_time),
                                  'interrupts_total_users_{}_{}.png'.format(batches_number, interrupt_time),
                                  np.array(markers), users_number=True)
    if args.re_reading:
        delay_time = args.min_delay_time if args.min_delay_time else 600
        batches_number = args.re_reading
        markers = get_re_reading_markers(batches_number, delay_time, book_id)
        all_markers.append(markers)
        combine_name += '_re_reading_{}_{}'.format(batches_number, delay_time)
        combine_title += ' ' if len(combine_title) == len('Book {}'.format(book_id)) else ', '
        combine_title += 're-reading (delay at least {} seconds)'.format(delay_time)
        reverse_markers.append(False)
        # visualize_markers(book_id,
        #                   'Book {} re-readings (delay at least {} seconds)'.format(book_id, delay_time),
        #                   're_readings_{}_{}.png'.format(batches_number, delay_time),
        #                   np.array(markers))
        visualize_markers_numbers(book_id,
                                  'Book {} re-readings (delay at least {} seconds)'.format(book_id, delay_time),
                                  're_readings_total_{}_{}.png'.format(batches_number, delay_time),
                                  np.array(markers))
        visualize_markers_numbers(book_id,
                                  'Book {} re-readings (delay at least {} seconds)'.format(book_id, delay_time),
                                  're_readings_total_users_{}_{}.png'.format(batches_number, delay_time),
                                  np.array(markers), users_number=True)
    if args.unusual_reading_hours:
        threshold_percentile = 30
        batches_borders = get_split_text_borders(book_id)
        user_ids = list(get_good_users_info(book_id).keys())
        markers = []
        for user_id in tqdm(user_ids[:20]):
            document_id = get_user_document_id(book_id, user_id)
            markers.append(UnusualReadingHoursMarker.get_for_user(book_id, document_id, user_id, len(batches_borders),
                                                                  batches_borders=batches_borders,
                                                                  threshold_percentile=threshold_percentile))
        markers = np.array(markers)
        visualize_markers(book_id,
                          f'Book {book_id} unusual reading hours',
                          f'unusual_reading_hours_{book_id}_{threshold_percentile}.png',
                          markers, )
    if args.speed:
        user_ids = list(get_good_users_info(book_id).keys())
        batches_borders = get_split_text_borders(book_id)
        result = np.zeros(len(batches_borders), dtype=np.int)
        for user_id in tqdm(user_ids):
            document_id = get_user_document_id(book_id, user_id)
            result += SpeedMarker.get_for_user(book_id, document_id, user_id, batches_borders=batches_borders)
        print(np.arange(result.shape[0])[np.argsort(result) < 10], result[np.argsort(result) < 10])
        plt.clf()
        plt.bar(0.5 + np.arange(0, result.shape[0]), result, width=1)
        plt.savefig(os.path.join('resources', 'plots', 'markers', str(book_id), 'speed', 'speed.png'))

    if args.plot_chapters_borders:
        combine_name += "_with_borders"
    if args.users_number:
        combine_name += "_users_number"
    if args.common_extremums:
        combine_name += "_common_extremums"
    if args.visualize_all:
        visualize_combine_markers_numbers(book_id, combine_title, combine_name, all_markers, reverse_markers,
                                          plot_extremums=args.plot_extremums,
                                          plot_chapters_borders=args.plot_chapters_borders,
                                          users_number=args.users_number,
                                          common_extremums=args.common_extremums)
    if args.quit_markers:
        user_ids = list(load_users(book_id))
        markers = []
        marker = OldQuitMarker(os.path.join('resources', 'chunks_poses.pkl'))
        values = []
        n = marker.get_chunks_number()
        for user_id in tqdm(user_ids):
            document_id = get_user_document_id(book_id, user_id)
            value = marker.get_marker(book_id, document_id, user_id)
            if value is not None:  # and value != n - 1:
                values.append(value)
        plot_path = os.path.join('resources', 'plots', 'markers', f'{book_id}', 'quit', 'quit.png')
        values = np.array(values)
        heights = np.zeros(n)
        for i in range(n):
            heights[i] = np.sum(values == i, dtype=int)
        plt.bar(0.5 + np.arange(0, n), heights[:], width=1)
        plt.savefig(plot_path)
