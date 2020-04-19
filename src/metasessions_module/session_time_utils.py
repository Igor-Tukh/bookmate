import numpy as np
import logging
import os
import sys
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from tqdm import tqdm

from src.metasessions_module.sessions_utils import get_all_user_sessions
from src.metasessions_module.utils import date_from_timestamp, date_to_percent_of_day, save_via_pickle, load_from_pickle
from src.metasessions_module.user_utils import get_good_users_info

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'sessions_time.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000


def session_to_time_range_number(session, time_ranges_number=48):
    range_percent = 1. / time_ranges_number
    session_time = date_from_timestamp(session['read_at'])
    session_percent = date_to_percent_of_day(session_time)
    return min(int(session_percent / range_percent), time_ranges_number - 1)


def build_user_time_distribution(user_id, time_ranges_number, book_id=None):
    """
    Builds the distribution of all the users' sessions time. Divides day to the time_ranges_number consequent equal
    parts and then calculates frequencies of the each part (i.e. percentage of all sessions with a timestamp of this
    part of day).
    :param book_id: book_id to build the distribution for a specific book None otherwise
    :param user_id: id of user1
    :param time_ranges_number: number of parts, default value is 48, i.e. each part corresponds to a half of hour
    :return: np.array of ratios with a size equal to time_ranges_number. Sum of its value is equal to 1.
    """
    logging.info(f'Building sessions\' time distribution for users {user_id}')
    sessions = get_all_user_sessions(user_id)
    if book_id is not None:
        sessions = [session for session in sessions if str(session['book_id']) == str(book_id)]
    distribution = np.zeros(time_ranges_number, dtype=np.float32)
    range_percent = 1. / time_ranges_number
    for session in tqdm(sessions):
        session_time = date_from_timestamp(session['read_at'])
        session_percent = date_to_percent_of_day(session_time)
        distribution[min(int(session_percent / range_percent), time_ranges_number - 1)] += 1
    distribution /= distribution.sum()
    return distribution


def get_user_time_distribution_path(user_id, book_id):
    if book_id is None:
        return os.path.join('resources', 'user_time_distribution', f'{user_id}.pkl')
    else:
        dir_path = os.path.join('resources', 'user_time_distribution', str(book_id))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return os.path.join(dir_path, f'{user_id}.pkl')


def get_user_time_distribution(user_id, time_ranges_number=48, book_id=None):
    path = get_user_time_distribution_path(user_id, book_id)
    if os.path.exists(path):
        d = load_from_pickle(path)
    else:
        d = save_via_pickle(build_user_time_distribution(user_id, time_ranges_number, book_id), path)
    return d


def plot_user_time_distribution(user_id, distribution, time_ranges_number=48, title_suffix='', filename_prefix=''):
    plt.clf()
    plt.bar(np.arange(time_ranges_number) + 0.5, distribution, width=1)
    plt.xlim(0, time_ranges_number)
    plt.title(f'Reading hours distribution for user {user_id}{title_suffix}')
    hour_dist = 1. * time_ranges_number / 24
    plt.xticks(np.arange(0, time_ranges_number, hour_dist), labels=[f'{hour}:00' for hour in range(0, 24)],
               fontsize=8, rotation='vertical')
    plt.savefig(os.path.join('resources', 'plots', 'reading_hours_distributions', f'{filename_prefix}_{user_id}.png') if
                filename_prefix != '' else f'{user_id}.png')


def plot_combined_time_distribution(user_id, book_id, time_ranges_number=48):
    all_d = get_user_time_distribution(user_id)
    d = get_user_time_distribution(user, book_id=book_id)
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.bar(np.arange(time_ranges_number) + 0.25, all_d, width=0.5, label='All books', edgecolor='black')
    plt.bar(np.arange(time_ranges_number) + 0.75, d, width=0.5, label=f'Book {book_id}', edgecolor='black')
    plt.xlim(0, time_ranges_number)
    plt.title(f'Reading hours distribution for user {user_id} and book {book_id}')
    hour_dist = 1. * time_ranges_number / 24
    plt.xticks(np.arange(0, time_ranges_number, hour_dist), labels=[f'{hour}:00' for hour in range(0, 24)],
               fontsize=20, rotation='vertical')
    plt.legend()
    plt.savefig(os.path.join('resources', 'plots', 'reading_hours_distributions', f'combined_{user_id}_{book_id}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=str, help='book id', required=True)
    args = parser.parse_args()
    book = args.book_id
    users = list(get_good_users_info(book).keys())
    logging.info(f'Found {len(users)} users')
    for user in tqdm(users[20:30]):
        plot_combined_time_distribution(user, book)
