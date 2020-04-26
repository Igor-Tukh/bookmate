import pickle
import os
import logging
import math
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from src.metasessions_module.utils import connect_to_mongo_database, date_from_timestamp, load_from_pickle, \
    save_via_pickle
from src.metasessions_module.item_utils import get_items
from src.metasessions_module.config import *

log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
root_logger = logging.getLogger()

file_handler = logging.FileHandler(os.path.join('logs', 'sessions_utils.log'), 'a')
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)


def is_target_speed(speed):
    return speed is not None and not math.isnan(speed) and speed != INFINITE_SPEED and speed != UNKNOWN_SPEED


def load_sessions(book_id):
    logging.info('Loading sessions of book {}'.format(book_id))
    sessions_path = os.path.join('resources', 'sessions', '{book_id}.pkl'.format(book_id=book_id))
    try:
        with open(sessions_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error('Loading sessions for book {} failed (file {} not found)'.format(book_id, sessions_path))
        raise ValueError('Unable to load sessions for book {} (file {} not found)'
                         .format(book_id, sessions_path))


def save_sessions(book_ids):
    logging.info('Books loading start')
    logging.info('Books: ' + ', '.join([str(book_id) for book_id in book_ids]))

    db = connect_to_mongo_database('sessions')
    db_work = connect_to_mongo_database('bookmate_work')
    books_to_save = [book_id for book_id in book_ids
                     if 'sessions_{book_id}'.format(book_id=book_id) not in db_work.collection_names()]
    logging.info('Books to save: {}'.format(str(books_to_save)))
    sessions = []
    if len(books_to_save) > 0:
        logging.info('Search for sessions of books {}'.format(str(books_to_save)))
        sessions = list(db['sessions'].find({"$or": [{'book_id': int(book_id)} for book_id in books_to_save]}))
    logging.info('Books loading finished')

    for book_id in books_to_save:
        collection_name = 'sessions_{book_id}'.format(book_id=book_id)
        book_sessions = [session for session in sessions if session['book_id'] == book_id]
        db_work[collection_name].insert_many(book_sessions)

    for book_id in book_ids:
        sessions_path = os.path.join('resources', 'sessions', '{book_id}.pkl'.format(book_id=book_id))
        if not os.path.isfile(sessions_path):
            logging.info('Saving sessions for book {} to {}'.format(book_id, sessions_path))
            book_sessions = list(db_work['sessions_{book_id}'.format(book_id=book_id)].find({}))
            # os.makedirs(sessions_path)
            with open(sessions_path, 'wb') as file:
                pickle.dump(book_sessions, file)
            logging.info('book {book_id}: done'.format(book_id=book_id))


def load_user_sessions(book_id, document_id, user_id):
    db_work = connect_to_mongo_database('bookmate_work')
    collection_name = 'sessions_{}'.format(book_id)
    sessions = db_work[collection_name].find({'user_id': user_id, 'document_id': document_id})
    return list(sessions)


def get_user_sessions(book_id, document_id, user_id):
    sessions_path = os.path.join('resources', 'sessions_book_document_user', '{}_{}_{}.pkl'.format(book_id,
                                                                                                   document_id,
                                                                                                   user_id))
    if os.path.exists(sessions_path):
        return load_from_pickle(sessions_path)
    sessions = load_user_sessions(book_id, document_id, user_id)
    save_via_pickle(sessions, sessions_path)
    return sessions


def load_document_sessions(book_id, document_id):
    db_work = connect_to_mongo_database('bookmate_work')
    collection_name = 'sessions_{}'.format(book_id)
    sessions = db_work[collection_name].find({'document_id': document_id})
    return list(sessions)


def save_user_sessions_speed(book_id, document_id, user_id):
    logging.info('Calculating user {} sessions speed for document {} of book {}'.format(user_id, document_id, book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    db_work = connect_to_mongo_database('bookmate_work')
    collection_name = 'sessions_{}'.format(book_id)
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    db_work[collection_name].update({'_id': user_sessions[0]['_id']}, {'$set': {'speed': UNKNOWN_SPEED}})
    for ind, session in enumerate(user_sessions[1:], 1):
        previous_session = user_sessions[ind - 1]
        time = (date_from_timestamp(session['read_at']) - date_from_timestamp(previous_session['read_at'])) \
            .total_seconds()
        if time < 0.0001:
            speed = INFINITE_SPEED
            logging.info('Found session {} with infinite speed'.format(session['_id']))
        else:
            speed = previous_session['size'] * 60 / time
        db_work[collection_name].update({'_id': session['_id']}, {'$set': {'speed': speed}})
    logging.info('Calculating sessions speed over')


def save_book_sessions(book_id):
    save_sessions([book_id])


def calculate_session_percents(book_id, document_ids):
    collection_name = 'sessions_{}'.format(book_id)
    db_work = connect_to_mongo_database('bookmate_work')
    if collection_name not in db_work.collection_names():
        logging.error('Unable to find sessions for book {} in bookmate_work collection'.format(book_id))
    sessions = db_work[collection_name].find({"$or": [{'document_id': int(document_id)}
                                                      for document_id in document_ids]})
    items = defaultdict(lambda: None)
    for document_id in document_ids:
        document_items = get_items(document_id)
        for document_item in document_items:
            items[document_item['id']] = document_item

    sessions = list(sessions)
    for session, _ in zip(sessions, tqdm(range(len(sessions)))):
        session_item = items[session['item_id']]
        if session_item is None:
            logging.error(
                'Item {} for session {} not found, session skipped'.format(session['item_id'], session['_id']))
            continue
        try:
            if np.isclose(float(session_item['_to']), float(session_item['_from'])):
                session_book_from = float(session_item['_from'])
                session_book_to = session_book_from
            else:
                session_book_from = float(session_item['_from']) + \
                                    (float(session_item['_to']) - float(session_item['_from'])) * \
                                    float(session['_from']) / 100
                session_book_to = float(session_item['_from']) + \
                                  (float(session_item['_to']) - float(session_item['_from'])) * float(
                    session['_to']) / 100
            db_work[collection_name].update({'_id': session['_id']}, {'$set': {'book_from': session_book_from,
                                                                               'book_to': session_book_to}})
        except Exception:
            logging.error('Session {} skipped due to the internal problem'.format(session['_id']))
    logging.info('Sessions book_from and book_to fields added')


def get_book_percent(book_id, document_id, user_id):
    sessions = load_user_sessions(book_id, document_id, user_id)
    sessions = [session for session in sessions if 'book_from' in session]
    sessions.sort(key=lambda session: session['book_from'])
    logging.info('Found {} sessions for user {}'.format(len(sessions), user_id))

    eps = 1e-5
    current_end_percent = 0
    current_start_percent = 0
    read_percent = 0
    for session in sessions:
        if 'book_from' not in session or 'book_to' not in session:
            continue
        if math.isnan(session['book_from']) or math.isnan(session['book_to']):
            continue
        if session['book_from'] < current_end_percent + eps:
            current_end_percent = max(session['book_to'], current_end_percent)
        else:
            read_percent += current_end_percent - current_start_percent
            current_start_percent = session['book_from']
            current_end_percent = session['book_to']

    return read_percent + current_end_percent - current_start_percent


def get_all_user_sessions_path(user_id):
    return os.path.join('resources', 'all_user_sessions', f'{user_id}.pkl')


def collect_all_user_sessions(user_id):
    logging.info(f'Collecting all user sessions for user {user_id} started')
    db = connect_to_mongo_database('sessions')
    sessions = list(db['sessions'].find({'user_id': int(user_id)}))
    logging.info(f'Collecting all user sessions for user {user_id} finished')
    return sessions


def get_all_user_sessions(user_id):
    sessions_path = get_all_user_sessions_path(user_id)
    if not os.path.exists(sessions_path):
        sessions = save_via_pickle(collect_all_user_sessions(user_id), sessions_path)
    else:
        sessions = load_from_pickle(sessions_path)
    logging.info(f'All user sessions for user {user_id} loaded')
    # logging.info(f'Totally found {len(sessions)} sessions for user {user_id} from '
    #              f'{len(set([session["book_id"] for session in sessions]))} books')
    return sessions


def collect_users_sessions(user_ids):
    logging.info('Collecting user sessions started')
    db = connect_to_mongo_database('sessions')
    sessions = {}
    for user_id in tqdm(user_ids):
        sessions[user_id] = list(db['sessions'].find({'user_id': int(user_id)}))
    logging.info('Collecting user sessions finished')
    return sessions


def get_users_sessions_path(users_group_name):
    return os.path.join('resources', 'users_sessions', f'{users_group_name}.pkl')


def get_users_sessions(user_ids, users_group_name=''):
    """
    :return: dict from user_id to the sessions with this user_id
    """
    logging.info(f'Collecting user sessions for {users_group_name} started')
    sessions_path = get_users_sessions_path(users_group_name)
    if os.path.exists(sessions_path):
        return load_from_pickle(sessions_path)
    else:
        return save_via_pickle(collect_users_sessions(user_ids), sessions_path)


def save_all_sessions_by_user():
    logging.info('Collecting all sessions by user started')
    db = connect_to_mongo_database('sessions')
    sessions = list(db['sessions'].find({}))
    logging.info('All sessions collected')
    sessions_by_user = defaultdict(lambda: [])
    for session in tqdm(sessions):
        sessions_by_user[int(session['user_id'])].append(session)
    for user_id, user_sessions in tqdm(sessions_by_user.keys()):
        path = get_all_user_sessions_path(user_id)
        if not os.path.exists(path):
            save_via_pickle(user_sessions, path)


if __name__ == '__main__':
    save_all_sessions_by_user()
