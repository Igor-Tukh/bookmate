import os
import sys
import argparse
import numpy as np

from src.metasessions_module.interest_marker.config import get_marker_dumps_path, get_marker_plots_path
from src.metasessions_module.interest_marker.interest_marker_utils import get_fragment, visualize_cumulative_marker, \
    calculate_numbers_of_readers_per_fragment, get_cumulative_marker, save_marker_stats, get_read_fragments, \
    save_marker_stats_with_normalization
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.abstract_interest_marker import AbstractInterestMarker
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.utils import date_from_timestamp, save_via_pickle, get_stats_path
from tqdm import tqdm


class ScrollingMarker(AbstractInterestMarker):
    @staticmethod
    def get_for_user(book_id, document_id, user_id, fragments_borders=None, min_await_threshold=3):
        if fragments_borders is None:
            fragments_borders = get_split_text_borders(book_id)
        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions.sort(key=lambda value: date_from_timestamp(value['read_at']))
        infinite_speeds = np.zeros(len(fragments_borders), dtype=np.int)

        for ind, session in enumerate(sessions[1:]):
            if 'book_to' not in sessions[ind] or 'book_from' not in sessions[ind]:
                continue
            time_from_last_session = (date_from_timestamp(session['read_at']) -
                                      date_from_timestamp(sessions[ind]['read_at'])).total_seconds()
            start_pos = get_fragment(fragments_borders, sessions[ind]['book_from'])
            end_pos = get_fragment(fragments_borders, sessions[ind]['book_to'])
            if np.less_equal(time_from_last_session, min_await_threshold):
                infinite_speeds[start_pos] += 1
                if end_pos != start_pos:
                    infinite_speeds[end_pos] += 1

        return (infinite_speeds < 1) & (get_read_fragments(book_id, user_id))

    @staticmethod
    def get_marker_title():
        return 'scrolling_marker'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', required=True, type=int)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    book_id = args.book_id
    users = list(get_good_users_info(book_id).keys())
    fragments_borders = get_split_text_borders(book_id)
    markers = []
    for user in tqdm(users):
        markers.append(ScrollingMarker.get_for_user(book_id, get_user_document_id(book_id, user), user,
                       fragments_borders=fragments_borders))
        if args.save:
            save_via_pickle(markers[-1], get_marker_dumps_path(ScrollingMarker.get_marker_title(), book_id, user))
    plot_path = get_marker_plots_path(book_id, ScrollingMarker.get_marker_title())
    normalization_weight = calculate_numbers_of_readers_per_fragment(book_id, users, markers[0].shape[0]) + 1
    visualize_cumulative_marker(markers, os.path.join(plot_path, 'cumulative.png'), 'Cumulative scrolling marker')
    visualize_cumulative_marker(markers, os.path.join(plot_path, 'normalized_cumulative.png'),
                                'Normalized cumulative scrolling marker', normalization_weight=normalization_weight)
    cumulative_marker = get_cumulative_marker(markers, binarize=True)
    save_marker_stats_with_normalization(get_stats_path('scrolling_marker_stats.csv'), cumulative_marker,
                                         normalization_weight, 'Scrolling Marker')
