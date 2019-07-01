import logging
import pickle
import os
import csv

from src.metasessions_module.utils import connect_to_mongo_database
from src.metasessions_module.sessions_utils import load_sessions, load_user_sessions

from collections import defaultdict


def get_users(book_id):
    try:
        sessions = load_sessions(book_id)
    except ValueError:
        logging.error('Unable to load users for book {} (session aren\'t saved)'.format(book_id))
        raise ValueError('Unable to distinct users for book {} (unable to load sessions, save_sessions call required)'
                         .format(book_id))
    return list(set([session['user_id'] for session in sessions]))


def save_users(book_ids):
    for book_id in book_ids:
        users = get_users(book_id)
        users_path = os.path.join('resources', 'users', '{book_id}.pkl'.format(book_id=book_id))
        if not os.path.isfile(users_path):
            logging.info('Processing users of book {}'.format(book_id))
            with open(users_path, 'wb') as file:
                pickle.dump(users, file)
            logging.info('Users of book {}: processing done'.format(book_id))


def load_users(book_id):
    logging.info('Loading users of book {}'.format(book_id))
    users_path = os.path.join('resources', 'users', '{book_id}.pkl'.format(book_id=book_id))
    try:
        with open(users_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error('Loading users of book {} failed (file {} not found)'.format(book_id, users_path))
        raise ValueError('Unable to load users of book {} (file {} not found)'
                         .format(book_id, users_path))


def get_common_users(book_ids):
    logging.info('Loading common users for books {}'.format(str(book_ids)))
    users = set()
    if len(book_ids) != 0:
        users.update(load_users(book_ids[0]))
        for book_id in book_ids:
            next_book_users = set(load_users(book_id))
            users = users.intersection(next_book_users)
    logging.info('Total found {} common users for books {}'.format(len(users), str(book_ids)))
    return list(users)


def save_common_users(book_ids):
    users = get_common_users(book_ids)
    with open(os.path.join('resources', 'users', '{}_common.pkl'.format('_'.join(list(map(str, book_ids))))), 'wb') \
            as file:
        pickle.dump(users, file)
    logging.info('Common users for books {} saved'.format(str(book_ids)))


def get_books_users_sessions(book_ids):
    logging.info('Selecting users for books {}'.format(str(book_ids)))
    users = set()
    for book_id in book_ids:
        book_users = load_users(book_id)
        users.update(book_users)
    logging.info('Total found {} users'.format(len(users)))

    db = connect_to_mongo_database('sessions')
    sessions = list(db['sessions'].find({}))
    return [session for session in sessions if session['user_id'] in users]


def load_common_users(book_ids):
    users_path = os.path.join('resources', 'users', '{}_common.pkl'.format('_'.join(list(map(str, book_ids)))))
    try:
        with open(users_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error('Loading users common users of books {} failed (file {} not found)'
                      .format(str(book_ids), users_path))
        raise ValueError('Unable to load common users of books {} (file {} not found)'
                         .format(str(book_ids), users_path))


def save_books_users_sessions(book_ids):
    logging.info('Selecting sessions for users of books {}'.format(str(book_ids)))
    sessions = get_books_users_sessions(book_ids)
    books_name = '_'.join(book_ids) + '_users.pkl'
    sessions_path = os.path.join('resources', 'sessions', books_name)
    logging.info('Saving sessions for users of books {} to {}'.format(str(book_ids), sessions_path))
    with open(sessions_path, 'wb') as file:
        pickle.dump(sessions, file)
    logging.info('Sessions for users of books {} saved to {}'.format(str(book_ids), sessions_path))


def get_user_document_id(book_id, user_id):
    db_work = connect_to_mongo_database('bookmate_work')
    collection_name = 'sessions_{}'.format(book_id)
    session = db_work[collection_name].find_one({'user_id': user_id})
    return session['document_id']


def get_users_books_amount(user_id):
    logging.info('Counting books amount for user {}'.format(user_id))
    db = connect_to_mongo_database('sessions')
    user_sessions = list(db['sessions'].find({'user_id': user_id}))
    logging.info('Found {} sessions for user {}'.format(len(user_sessions), user_id))
    books = set([session['book_id'] for session in user_sessions])
    logging.info('Found {} books for user {}'.format(len(books), user_id))
    return len(books)


def get_users_extra_information(path=None, id_header='id'):
    path = os.path.join('resources', 'users', 'users.csv') if path is None else path
    info = {}
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        info_headers = [header for header in headers if header != id_header]
        for row in reader:
            info[row[id_header]] = {header: (row[header] if header in row and row[header] != 'NULL'
                                             else '?') for header in info_headers}
    return info


def get_good_users_info(book_id, path=None, user_header='user_id'):
    path = os.path.join('resources', 'users', 'csv', '{}.csv'.format(book_id)) if path is None else path
    users = {}
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        info_headers = [header for header in headers if header != user_header]
        for row in reader:
            users[int(row[user_header])] = {header: row[header] for header in info_headers}
    return users


def upload_good_users(book_id):
    users_path = os.path.join('resources', 'users', '{}_good_users_id_amount.pkl'.format(book_id))
    if os.path.isfile(users_path):
        with open(users_path, 'rb') as file:
            return pickle.load(file)
    else:
        logging.error('Unable to upload good users for book {}'.format(book_id))


def upload_good_users_with_percents(book_id):
    users_path = os.path.join('resources', 'users', '{}_good_users_id_amount_percent.pkl'.format(book_id))
    if os.path.isfile(users_path):
        with open(users_path, 'rb') as file:
            return pickle.load(file)
    else:
        logging.error('Unable to upload good users for book {}'.format(book_id))


def get_user_selection(book_id):
    output_path = os.path.join('resources', 'users', '{}_users_selection.pkl'.format(book_id))
    if os.path.isfile(output_path):
        with open(output_path, 'rb') as file:
            return pickle.load(file)
    logging.error('Unable to load users selection for book {}'.format(book_id))
    return []


def save_user_sessions_by_place_in_book(book_id, document_id, user_id, output_path=None):
    output_path = output_path if output_path is not None \
        else os.path.join('resources', 'sessions_filtered', '{}_{}_{}.pkl'.format(book_id, document_id, user_id))
    logging.info('Saving sessions by place in document {} of book {} for user {}'.format(user_id,
                                                                                         document_id,
                                                                                         book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    sessions_dict = defaultdict(lambda: [])
    for session in user_sessions:
        sessions_dict[(session['book_from'], session['book_to'])].append(session)
    unique_sessions = {}
    for key, value in sessions_dict.items():
        value.sort(key=lambda session_reading: session_reading['read_at'])
        unique_sessions[key] = value[0] if len(value) == 1 else value[1]
    with open(output_path, 'wb') as file:
        pickle.dump(unique_sessions, file)
    logging.info('Sessions by place in document {} of book {} for user {} saved to {}'
                 .format(document_id, book_id, user_id, output_path))


def get_user_sessions_by_place_in_book(book_id, document_id, user_id, rebuild=False):
    output_path = os.path.join('resources', 'sessions_filtered', '{}_{}_{}.pkl'.format(book_id, document_id, user_id))
    if not os.path.isfile(output_path) or rebuild:
        save_user_sessions_by_place_in_book(book_id, document_id, user_id, output_path)
    logging.info('Loading sessions by place in document {} of book {} for user {}'
                 .format(document_id, book_id, user_id))
    with open(output_path, 'rb') as file:
        return pickle.load(file)


def load_hidden_ids_map(book_id):
    ids_path = os.path.join('resources', 'users', 'ids', '{}.csv'.format(book_id))
    if not os.path.isfile(ids_path):
        logging.error('Error loading user ids map for book {}, file {} not found'.format(book_id, ids_path))
        raise ValueError('Incorrect book for loading ids: {}'.format(book_id))
    result = dict()
    with open(ids_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result[int(row['real_id'])] = int(row['id'])
    return result
