import os
import sys
import numpy as np
import argparse

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))


from src.metasessions_module.interest_marker.interest_marker_utils import detect_anomaly_in_read_moments, \
    get_read_fragments
from src.metasessions_module.user_utils import get_good_users_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    users = list(get_good_users_info(book_id).keys())
    for user in users:
        anomaly_mask = detect_anomaly_in_read_moments(book_id, user)
