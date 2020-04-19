import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.interest_marker.abstract_interest_marker import AbstractInterestMarker
from src.metasessions_module.text_utils import get_split_text_borders
from src.metasessions_module.interest_marker.interest_marker_utils import get_batch
from src.metasessions_module.sessions_utils import get_user_sessions
from src.metasessions_module.user_utils import get_good_users_info, get_user_document_id
from src.metasessions_module.utils import date_from_timestamp

from tqdm import tqdm

EPS = 1e-3


class OldSpeedMarker(AbstractInterestMarker):
    @staticmethod
    def get_for_user(book_id, document_id, user_id, batches_number=None, batches_borders=None, max_delay_in_minutes=10):
        if batches_number is None and batches_borders is None:
            batches_borders = get_split_text_borders(book_id)
        result = np.zeros(batches_number if batches_number is not None else len(batches_borders), dtype=np.int)

        sessions = get_user_sessions(book_id, document_id, user_id)
        sessions.sort(key=lambda value: date_from_timestamp(value['read_at']))
        target_sessions = []
        speeds = []
        for ind, session in enumerate(sessions[1:]):
            if 'book_to' not in sessions[ind] or 'book_from' not in session or \
                    abs(session['book_from'] - sessions[ind]['book_to']) / 100 > EPS:
                continue
            time_from_last_session = (date_from_timestamp(session['read_at']) -
                                      date_from_timestamp(sessions[ind]['read_at'])).total_seconds()
            if time_from_last_session + EPS < max_delay_in_minutes * 60:
                start_pos = get_batch(batches_number, batches_borders, sessions[ind]['book_from']) \
                    if 'book_from' in sessions[ind] else None
                end_pos = get_batch(batches_number, batches_borders, sessions[ind]['book_to']) \
                    if 'book_to' in sessions[ind] else None
                if start_pos is None or end_pos is None or abs(time_from_last_session) < EPS:
                    continue
                speeds.append(1. * sessions[ind]['size'] / time_from_last_session)
                target_sessions.append(
                    (session, speeds[-1], start_pos, end_pos))

        anomaly_mask = OldSpeedMarker._get_anomaly_mask(user_id, speeds)
        for ind, (session, speed, start_pos, end_pos) in enumerate(target_sessions):
            if anomaly_mask[ind]:
                if start_pos != end_pos:
                    result[end_pos] += 1
                result[start_pos] += 1
        return result

    @staticmethod
    def get_marker_title():
        return 'old_speed_marker'

    @staticmethod
    def _get_anomaly_mask(user_id, speeds, bins_number=100):
        speeds = np.array(speeds)
        normal_speeds = [speed for speed in speeds if speed < 100 + EPS]
        hist, edges = np.histogram(normal_speeds, bins=bins_number)
        lower_threshold = OldSpeedMarker._get_lower_threshold(hist)
        upper_threshold = OldSpeedMarker._get_upper_threshold(hist)
        plt.clf()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.hist(normal_speeds, bins=bins_number)
        ax.set_title(f'User {user_id} speed distribution')
        ax.set_xlabel('Speed, characters / second')
        ax.set_ylabel('Number of sessions')
        ax.set_xlim(0, np.max(normal_speeds))
        ax.add_patch(plt.Rectangle((0, lower_threshold), np.max(normal_speeds), upper_threshold - lower_threshold,
                                   fill=True, color='g', alpha=0.5))
        plt.savefig(os.path.join('resources', 'plots', 'speed_distributions', f'{user_id}.png'))
        normal_speed_mask = np.array([OldSpeedMarker._is_normal_speed(lower_threshold, upper_threshold, speed, hist, edges)
                                      for speed in speeds], dtype=np.bool)
        return ~normal_speed_mask

    @staticmethod
    def _get_lower_threshold(speed_hist, percent=0.1):
        threshold = 0
        total_sum = np.sum(speed_hist)
        for value in np.sort(speed_hist):
            lower_sum = np.sum(speed_hist[speed_hist <= value])
            if 1. * lower_sum / total_sum < percent:
                threshold = value
        return threshold

    @staticmethod
    def _get_upper_threshold(speed_hist, percent=0.1):
        threshold = 1e9
        total_sum = np.sum(speed_hist)
        for value in reversed(np.sort(speed_hist)):
            upper_sum = np.sum(speed_hist[speed_hist >= value])
            if 1. * upper_sum / total_sum < percent:
                threshold = value
        return threshold

    @staticmethod
    def _is_normal_speed(lower, upper, speed, hist, edges):
        ind = 0
        batch_number = 0
        while ind < len(edges) and edges[ind] <= speed:
            batch_number = ind
            ind += 1

        if batch_number >= hist.shape[0]:
            batch_number = hist.shape[0] - 1
        return (lower < hist[batch_number] < upper) or (speed > 100 - EPS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int)
    args = parser.parse_args()
    book_id = args.book_id
    user_ids = list(get_good_users_info(book_id).keys())
    for user_id in tqdm(user_ids[:80]):
        document_id = get_user_document_id(book_id, user_id)
        markers = OldSpeedMarker.get_for_user(book_id, document_id, user_id)
