import os
import sys
import logging
import numpy as np
import math
import argparse

from src.metasessions_module.interest_marker.config import get_marker_dumps_path, get_marker_plots_path
from src.metasessions_module.interest_marker.interest_marker_utils import get_fragment, visualize_binary_markers, \
    visualize_cumulative_marker, save_marker_stats, calculate_numbers_of_readers_per_fragment, get_cumulative_marker, \
    get_read_fragments
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.interest_marker.abstract_interest_marker import AbstractInterestMarker
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.utils import date_from_timestamp, get_batch_by_percent, save_via_pickle, get_stats_path
from src.metasessions_module.text_utils import get_split_text_borders, load_chapters_mask
from tqdm import tqdm


class ReadingInterruptMarker(AbstractInterestMarker):
    @staticmethod
    def get_marker_title():
        return 'reading_interrupt_marker'

    @staticmethod
    def get_for_user(book_id, document_id, user_id, interrupt_skip_seconds=600,
                     only_consequent=True, fragments_borders=None, without_chapters_borders=False):
        logging.info('Collection reading interrupt markers for user {} of book {}'.format(user_id, book_id))
        if fragments_borders is None:
            fragments_borders = get_split_text_borders(book_id)
        markers = np.zeros(len(fragments_borders), dtype=np.int)
        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions.sort(key=lambda value: date_from_timestamp(value['read_at']))
        for ind, session in enumerate(sessions[1:]):
            prev_session = sessions[ind]
            skip_time = (date_from_timestamp(session['read_at']) - date_from_timestamp(prev_session['read_at'])).\
                total_seconds()
            if (not only_consequent or np.isclose(session['book_from'], prev_session['book_to'])) and \
                    skip_time >= interrupt_skip_seconds and not math.isnan(prev_session['book_to']):
                markers[get_fragment(fragments_borders, prev_session['book_to'])] += 1
        if without_chapters_borders:
            markers[load_chapters_mask(book_id)] = 0
        return np.array((markers < 1) & get_read_fragments(book_id, user_id), dtype=np.bool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', required=True, type=int)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    book_id = args.book_id
    fragment_borders = get_split_text_borders(book_id)
    users = list(get_good_users_info(book_id).keys())
    all_markers = []
    all_markers_without_borders = []
    for user in tqdm(users):
        all_markers.append(ReadingInterruptMarker.get_for_user(book_id, get_user_document_id(book_id, user), user,
                                                               fragments_borders=fragment_borders))
        all_markers_without_borders.append(ReadingInterruptMarker.get_for_user(book_id,
                                                                               get_user_document_id(book_id, user),
                                                                               user,
                                                                               fragments_borders=fragment_borders,
                                                                               without_chapters_borders=True))
        if args.save:
            save_via_pickle(all_markers[-1], get_marker_dumps_path(ReadingInterruptMarker.get_marker_title(), book_id,
                                                                   user))
    plot_path = get_marker_plots_path(book_id, 'reading_interrupt')

    # visualize_binary_markers(all_markers, os.path.join(plot_path, 'reading_interrupt.png'),
    #                          x_label='Positions', y_label='Users', title='Reading interrupt marker', binarize=True)
    # visualize_binary_markers(all_markers, os.path.join(plot_path, 'reading_interrupt_without_chapter_borders.png'),
    #                          x_label='Positions', y_label='Users', title='Reading interrupt marker', binarize=True)
    #
    # visualize_cumulative_marker(all_markers, os.path.join(plot_path, 'cumulative.png'),
    #                             'Cumulative reading interrupt marker')
    # normalization_weight = calculate_numbers_of_readers_per_fragment(book_id, users, all_markers[0].shape[0]) + 1
    # visualize_cumulative_marker(all_markers, os.path.join(plot_path, 'cumulative_normalized.png'),
    #                             'Cumulative reading interrupt marker', normalization_weight=normalization_weight)
    #
    # visualize_cumulative_marker(all_markers_without_borders,
    #                             os.path.join(plot_path, 'cumulative_without_chapter_borders.png'),
    #                             'Cumulative reading interrupt marker')
    # normalization_weight = calculate_numbers_of_readers_per_fragment(book_id, users, all_markers[0].shape[0]) + 1
    # visualize_cumulative_marker(all_markers_without_borders,
    #                             os.path.join(plot_path, 'cumulative_without_chapter_borders_normalized.png'),
    #                             'Cumulative reading interrupt marker', normalization_weight=normalization_weight)
