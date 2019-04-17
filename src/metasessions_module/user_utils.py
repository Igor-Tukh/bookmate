import logging
import pickle
import os

from metasessions_module.utils import connect_to_mongo_database
from metasessions_module.sessions_utils import load_sessions


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
        logging.info('Processing users of book {}'.format(book_id))
        users = get_users(book_id)
        with open(os.path.join('resources', 'users', '{book_id}.pkl'.format(book_id=book_id)), 'wb') as file:
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
    logging.info('Sessions for users of books{} saved to {}'.format(str(book_ids), sessions_path))


def get_user_document_id(book_id, user_id):
    db_work = connect_to_mongo_database('bookmate_work')
    collection_name = 'sessions_{}'.format(book_id)
    session = db_work[collection_name].find_one({'user_id': user_id})
    return session['document_id']
