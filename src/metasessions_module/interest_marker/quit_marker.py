import os
import sys
from datetime import datetime

import numpy as np

from src.metasessions_module.interest_marker.interest_marker_utils import get_fragment, get_read_fragments

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.abstract_interest_marker import AbstractInterestMarker
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.utils import date_from_timestamp


class QuitMarker(AbstractInterestMarker):
    @staticmethod
    def get_marker_title():
        return 'quit_marker'

    @staticmethod
    def get_for_user(book_id, document_id, user_id, fragments_borders=None):
        if fragments_borders is None:
            fragments_borders = get_split_text_borders(book_id)
        result = np.zeros(len(fragments_borders), dtype=np.int)

        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions.sort(key=lambda value: date_from_timestamp(value['read_at']))
        last_session_time = date_from_timestamp(sessions[-1]['read_at'])
        session_ind = -1
        if last_session_time <= datetime(year=2015, month=9, day=1):
            while session_ind > max(-len(sessions), -10) and 'book_to' not in sessions[session_ind].keys():
                session_ind -= 1
            result[get_fragment(fragments_borders, sessions[session_ind]['book_to'])] += 1
        return np.array((result < 1) & get_read_fragments(book_id, user_id), dtype=np.bool)

