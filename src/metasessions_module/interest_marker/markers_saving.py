import sys
import os
import logging
import argparse

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.quit_marker import QuitMarker
from src.metasessions_module.interest_marker.re_reading_marker import ReReadingMarker
from src.metasessions_module.interest_marker.reading_interrupt_marker import ReadingInterruptMarker
from src.metasessions_module.interest_marker.unusual_reading_hours_marker import UnusualReadingHoursMarker
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id
from src.metasessions_module.utils import save_via_pickle

from tqdm import tqdm

log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
root_logger = logging.getLogger()

file_handler = logging.FileHandler(os.path.join('logs', 'markers_saving.log'), 'a')
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(log_formatter)
root_logger.addHandler(consoleHandler)
root_logger.setLevel(logging.INFO)
log_step = 100000


def get_default_markers_path():
    return os.path.join('resources', 'all_markers')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=str, help='ID of book to save', required=True)
    parser.add_argument('--dir', type=str, help='Directory to save', required=False, default=get_default_markers_path())
    parser.add_argument('--min_interrupt_time', help='Interrupt time in seconds', type=int, required=False, default=600)
    parser.add_argument('--min_delay_time', help='Delay time in seconds', type=int, required=False, default=300)
    parser.add_argument('--unusual_hours_percentile', help='UH threshold', type=int, required=False, default=10)
    args = parser.parse_args()

    marker_build_args = {ReReadingMarker: {'delay_seconds': args.min_delay_time},
                         ReadingInterruptMarker: {'interrupt_skip_seconds': args.min_interrupt_time},
                         UnusualReadingHoursMarker: {'threshold_percentile': args.unusual_hours_percentile},
                         QuitMarker: {}}

    book_id = args.book_id
    batches_borders = get_split_text_borders(book_id)
    users = list(get_good_users_info(book_id).keys())

    for user_id in tqdm(users[31:31]):
        document_id = get_user_document_id(book_id, user_id)
        for marker_builder in [UnusualReadingHoursMarker]:
            builder_args = marker_build_args[marker_builder]
            s = f'_{list(builder_args.values())[0]}' if len(builder_args) > 0 else ''
            marker_path = os.path.join(args.dir, f'{marker_builder.get_marker_title()}_{book_id}_{user_id}{s}.pkl')
            if os.path.exists(marker_path):
                continue
            markers = marker_builder.get_for_user(book_id, document_id, user_id, fragments_borders=batches_borders,
                                                  **builder_args)
            logging.info(f'Saving marker to {marker_path}')
            save_via_pickle(markers, marker_path)
