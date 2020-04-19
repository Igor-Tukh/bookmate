import os
import sys
import logging
import numpy as np
import math

from src.metasessions_module.interest_marker.interest_marker_utils import get_fragment, get_read_fragments
from src.metasessions_module.text_utils import get_split_text_borders

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.interest_marker.abstract_interest_marker import AbstractInterestMarker
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.utils import date_from_timestamp


class ReReadingMarker(AbstractInterestMarker):
    @staticmethod
    def get_marker_title():
        return 're_reading_marker'

    @staticmethod
    def is_re_reading(delay, same_position_sessions):
        first_time = None
        last_time = None
        for session in same_position_sessions:
            session_time = date_from_timestamp(session['read_at'])
            if first_time is None or session_time < first_time:
                first_time = session_time
            if last_time is None or session_time > last_time:
                last_time = session_time
        if first_time is None or last_time is None:
            return False
        return (last_time - first_time).seconds >= delay

    @staticmethod
    def get_for_user(book_id, document_id, user_id, delay_seconds=300, fragments_borders=None):
        logging.info('Collection re-reading markers for user {} of book {}'.format(user_id, book_id))
        if fragments_borders is None:
            fragments_borders = get_split_text_borders(book_id)
        markers = np.zeros(len(fragments_borders), dtype=np.int)
        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions = [session for session in sessions if not math.isnan(session['book_from'])]
        sessions.sort(key=lambda value: value['book_from'])
        same_position_sessions = [sessions[0]]
        for ind, session in enumerate(sessions[1:]):
            if math.isnan(session['book_from']):
                continue
            previous_position = same_position_sessions[-1]['book_from']
            if np.isclose(previous_position, session['book_from']):
                same_position_sessions.append(session)
            else:
                if ReReadingMarker.is_re_reading(delay_seconds, same_position_sessions):
                    markers[get_fragment(fragments_borders, previous_position)] += 1
                same_position_sessions = [session]
        previous_position = same_position_sessions[-1]['book_from']
        if ReReadingMarker.is_re_reading(delay_seconds, same_position_sessions):
            markers[get_fragment(fragments_borders, previous_position)] += 1
        return np.array((markers >= 1) & get_read_fragments(book_id, user_id), dtype=np.bool)
