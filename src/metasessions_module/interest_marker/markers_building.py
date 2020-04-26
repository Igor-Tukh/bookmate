import os
import sys
import argparse
import logging

from src.metasessions_module.interest_marker.quit_marker import QuitMarker
from src.metasessions_module.interest_marker.re_reading_marker import ReReadingMarker
from src.metasessions_module.interest_marker.reading_interrupt_marker import ReadingInterruptMarker
from src.metasessions_module.interest_marker.scrolling_marker import ScrollingMarker
from src.metasessions_module.interest_marker.speed_marker import SpeedMarker
from src.metasessions_module.interest_marker.unusual_hours_marker import UnusualHoursMarker

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.interest_marker_utils import calculate_numbers_of_readers_per_fragment, \
    visualize_cumulative_marker, get_cumulative_marker, save_marker_stats_with_normalization, get_read_fragments, \
    fix_anomaly_in_marker, detect_anomaly_in_read_moments
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id, get_users
from src.metasessions_module.utils import save_via_pickle, get_stats_path, load_from_pickle
from src.metasessions_module.interest_marker.config import get_marker_dumps_path, \
    get_marker_plots_path, InterestMarker, get_book_stats_path, get_markers_path_for_book

from tqdm import tqdm

INTEREST_MARKERS_CLASSES = {
    InterestMarker.SPEED_MARKER: SpeedMarker,
    InterestMarker.SCROLLING_MARKER: ScrollingMarker,
    InterestMarker.RE_READING_MARKER: ReReadingMarker,
    InterestMarker.READING_INTERRUPT_MARKER: ReadingInterruptMarker,
    InterestMarker.UNUSUAL_HOURS_MARKER: UnusualHoursMarker,
    InterestMarker.QUIT_MARKER: QuitMarker
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=str, required=True)
    args = parser.parse_args()
    book_id = args.book_id

    # markers_path = os.path.join(get_markers_path_for_book(book_id), 'quit_marker')
    # users = [int(filename.split('.')[0]) for filename in os.listdir(markers_path)]
    users = list(get_good_users_info(book_id).keys())
    fragments_borders = get_split_text_borders(book_id)
    normalization_weight = calculate_numbers_of_readers_per_fragment(book_id, users, len(fragments_borders)) + 1
    anomaly_mask = {user: detect_anomaly_in_read_moments(book_id, user) for user in users}
    for marker_type, marker_class in INTEREST_MARKERS_CLASSES.items():
        if marker_type == InterestMarker.QUIT_MARKER:
            continue
        marker_title = marker_class.get_marker_title()
        logging.info(f'Building {marker_title}')
        markers = []
        for user in tqdm(users):
            current_marker = marker_class.get_for_user(book_id, get_user_document_id(book_id, user), user,
                                                       fragments_borders=fragments_borders)
            current_marker = fix_anomaly_in_marker(anomaly_mask[user], current_marker)
            markers.append(current_marker)
            # save_via_pickle(markers[-1], get_marker_dumps_path(marker_class.get_marker_title(), book_id, user))
        plot_path = get_marker_plots_path(book_id, marker_title)
        visualize_cumulative_marker(markers, os.path.join(plot_path, 'cumulative.png'), f'Cumulative {marker_title}')
        visualize_cumulative_marker(markers, os.path.join(plot_path, 'normalized_cumulative.png'),
                                    f'Normalized {marker_title}', normalization_weight=normalization_weight)
        cumulative_marker = get_cumulative_marker(markers, binarize=True)
        save_marker_stats_with_normalization(get_book_stats_path(book_id, f'{marker_title}_stats.csv'),
                                             cumulative_marker,
                                             normalization_weight, marker_title)
