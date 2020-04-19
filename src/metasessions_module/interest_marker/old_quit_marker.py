import pickle

from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.utils import date_from_timestamp
from datetime import datetime


class OldQuitMarker(object):
    def __init__(self, poses_path):
        with open(poses_path, 'rb') as poses_file:
            self.poses = pickle.load(poses_file)

    def get_chunk_ind_by_pos(self, pos):
        for ind, (pos_begin, pos_end) in enumerate(self.poses):
            if pos_begin <= pos <= pos_end + 1e-9:
                return ind
        return -1

    def get_chunks_number(self):
        return len(self.poses)

    def get_marker(self, book_id, document_id, user_id):
        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions.sort(key=lambda value: date_from_timestamp(value['read_at']))
        last_session_time = date_from_timestamp(sessions[-1]['read_at'])
        if last_session_time > datetime(year=2015, month=9, day=1):
            return None
        return self.get_chunk_ind_by_pos(sessions[-1]['book_to'] * 1. / 100) if 'book_to' in sessions[-1].keys() \
            else None
