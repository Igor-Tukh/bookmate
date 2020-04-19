import math
import os
import sys
import logging
import numpy as np


sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.interest_marker.abstract_interest_marker import AbstractInterestMarker
from src.metasessions_module.session_time_utils import get_user_time_distribution, session_to_time_range_number
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.interest_marker.interest_marker_utils import position_to_batch, get_read_fragments

from tqdm import tqdm


class UnusualHoursMarker(AbstractInterestMarker):
    @staticmethod
    def get_marker_title():
        return 'unusual_hours_marker'

    @staticmethod
    def get_for_user(book_id, document_id, user_id, threshold_percentile=20,
                     fragments_borders=None):
        logging.info('Collecting unusual reading hours for user {} of book {}'.format(user_id, book_id))
        if fragments_borders is None:
            fragments_borders = get_split_text_borders(book_id)
        markers = np.zeros(len(fragments_borders), dtype=np.int)
        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions = [session for session in sessions if not math.isnan(session['book_from'])]
        reading_hours_distribution = get_user_time_distribution(user_id)
        threshold = np.percentile(reading_hours_distribution, threshold_percentile)
        for session in tqdm(sessions):
            time_range_number = session_to_time_range_number(session)
            if reading_hours_distribution[time_range_number] < threshold + 1e-9:
                markers[position_to_batch(session['book_from'] / 100, fragments_borders)] += 1

        return np.array(markers >= 1, dtype=np.bool)
