import sys
import os
import argparse
import csv

import numpy as np

from src.metasessions_module.interest_marker.config import get_marker_plots_path, get_marker_dumps_path
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.abstract_interest_marker import AbstractInterestMarker
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.utils import date_from_timestamp, save_via_pickle, get_stats_path
from src.metasessions_module.interest_marker.interest_marker_utils import get_fragment, visualize_binary_markers, \
    visualize_cumulative_marker, calculate_numbers_of_readers_per_fragment, get_cumulative_marker, save_marker_stats, \
    get_read_fragments
from collections import defaultdict
from tqdm import tqdm


class SpeedMarker(AbstractInterestMarker):
    @staticmethod
    def get_marker_title():
        return 'speed_marker'

    @staticmethod
    def get_for_user(book_id, document_id, user_id, fragments_borders=None, percentile=80, speed_threshold=50,
                     extended_results=False):
        """
        Checks if speed is at least percentile-th percentile or at least speed_threshold symbols per second.
        """
        if fragments_borders is None:
            fragments_borders = get_split_text_borders(book_id)
        markers = np.zeros(len(fragments_borders), dtype=np.bool)
        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions.sort(key=lambda value: date_from_timestamp(value['read_at']))
        speeds = defaultdict(lambda: [])
        infinite_speeds = np.zeros(markers.shape[0], dtype=np.int)
        all_speeds = []
        for ind, session in enumerate(sessions[1:]):
            if 'book_to' not in sessions[ind] or 'book_from' not in session or 'book_to' not in session:
                continue
            time_from_last_session = (date_from_timestamp(session['read_at']) -
                                      date_from_timestamp(sessions[ind]['read_at'])).total_seconds()
            start_pos = get_fragment(fragments_borders, session['book_from'])
            end_pos = get_fragment(fragments_borders, session['book_to'])
            if np.isclose(time_from_last_session, 0):
                infinite_speeds[start_pos] += 1
                if end_pos != start_pos:
                    infinite_speeds[end_pos] += 1
            else:
                speed = 1. * sessions[ind]['size'] / time_from_last_session
                all_speeds.append(speed)
                speeds[start_pos].append(speed)
                if end_pos != start_pos:
                    speeds[end_pos].append(speed)

        speed_threshold = min(np.percentile(all_speeds, percentile), speed_threshold)

        for fragment_ind, fragment_speeds in speeds.items():
            markers[fragment_ind] = np.max(fragment_speeds) >= speed_threshold

        result = ~markers & (get_read_fragments(book_id, user_id))
        return (result,
                ~markers & (get_read_fragments(book_id, user_id)),
                np.array(infinite_speeds >= 1, dtype=np.bool) & (get_read_fragments(book_id, user_id)))\
            if extended_results else result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', required=True, type=int)
    parser.add_argument('--extended_results', required=False, action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    book_id = args.book_id
    fragment_borders = get_split_text_borders(book_id)
    users = list(get_good_users_info(book_id).keys())
    all_markers = []
    for user in tqdm(users):
        all_markers.append(SpeedMarker.get_for_user(book_id, get_user_document_id(book_id, user),
                                                    user, fragment_borders, extended_results=args.extended_results))
        if args.save:
            save_via_pickle(all_markers[-1], get_marker_dumps_path(SpeedMarker.get_marker_title(), book_id, user))
    plot_path = get_marker_plots_path(book_id, 'speed')
    if args.extended_results:
        visualize_binary_markers([m[0] for m in all_markers], os.path.join(plot_path, 'speed.png'),
                                 x_label='Positions', y_label='Users', title='Speed marker')
        visualize_binary_markers([m[1] for m in all_markers], os.path.join(plot_path, 'speed_without_infinite.png'),
                                 x_label='Positions', y_label='Users', title='Speed marker')
        visualize_binary_markers([m[2] for m in all_markers], os.path.join(plot_path, 'speed_infinite.png'),
                                 x_label='Positions', y_label='Users', title='Speed marker')
        visualize_cumulative_marker([marker[0] for marker in all_markers],
                                     os.path.join(plot_path, 'cumulative.png'), 'Cumulative speed marker')
        with open(os.path.join('resources', 'stats', 'speed_marker_stats.csv'), 'w') as csv_file:
            total_results = [np.zeros(all_markers[0][0].shape[0], dtype=np.int) for _ in range(3)]
            for r, m, i in all_markers:
                total_results[0] += r
                total_results[1] += m
                total_results[2] += i
            writer = csv.DictWriter(csv_file, fieldnames=['Fragment index',
                                                          'Total number',
                                                          'Total number without infinite',
                                                          'Total infinite number'])
            writer.writeheader()
            for ind, (f, s, t) in enumerate(zip(total_results[0], total_results[1], total_results[2])):
                writer.writerow({'Fragment index': ind,
                                 'Total number': f,
                                 'Total number without infinite': s,
                                 'Total infinite number': t})
    else:
        visualize_cumulative_marker(all_markers, os.path.join(plot_path, 'cumulative.png'), 'Cumulative speed marker')
        normalization_weight = calculate_numbers_of_readers_per_fragment(book_id, users, all_markers[0].shape[0]) + 1
        cumulative_marker = get_cumulative_marker(all_markers, binarize=True, normalization_weight=normalization_weight)
        save_marker_stats(get_stats_path('normalized_speed_marker_stats.csv'), cumulative_marker,
                          'Normalized speed marker')
        cumulative_marker = get_cumulative_marker(all_markers, binarize=True)
        save_marker_stats(get_stats_path('speed_marker_stats.csv'), cumulative_marker,
                          'Speed marker')
        visualize_cumulative_marker(all_markers, os.path.join(plot_path, 'cumulative_normalized.png'),
                                    'Cumulative speed marker', normalization_weight=normalization_weight)
