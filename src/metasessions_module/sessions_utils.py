import pickle
import os
import logging
import math

from collections import defaultdict
from tqdm import tqdm
from metasessions_module.utils import connect_to_mongo_database, date_from_timestamp
from metasessions_module.item_utils import get_items


INFINITE_SPEED = 10000000
UNKNOWN_SPEED = -1


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
    books_to_save = ['book_id' for book_id in book_ids
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
        book_sessions = list(db_work['sessions_{book_id}'.format(book_id=book_id)].find({}))
        sessions_path = os.path.join('resources', 'sessions', '{book_id}.pkl'.format(book_id=book_id))
        os.makedirs(sessions_path)
        with open(sessions_path, 'wb') as file:
            pickle.dump(book_sessions, file)
        logging.info('book {book_id}: done'.format(book_id=book_id))


def load_user_sessions(book_id, document_id, user_id):
    db_work = connect_to_mongo_database('bookmate_work')
    collection_name = 'sessions_{}'.format(book_id)
    sessions = db_work[collection_name].find({'user_id': user_id, 'document_id': document_id})
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
            logging.error('Item {} for session {} not found, session skipped'.format(session['item_id'], session['_id']))
            continue
        try:
            session_book_from = float(session_item['_from']) + \
                                (float(session_item['_to']) - float(session_item['_from'])) * float(session['_from']) / 100
            session_book_to = float(session_item['_from']) + \
                              (float(session_item['_to']) - float(session_item['_from'])) * float(session['_to']) / 100
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
