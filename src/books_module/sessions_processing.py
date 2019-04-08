from pymongo import MongoClient
import datetime
import pymongo
import bisect
import math
import logging
import timeit
import sys
from tqdm import tqdm
from collections import Counter

sys.setrecursionlimit(2000)
from src.books_module.SegmentTree import SegmentTree

# logs
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("{0}/{1}.log".format('logs', 'bookmate'), 'w')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000

# db
BOOKS_DB = 'bookmate'
USERS_DB = 'bookmate_users'
FULL_SESSIONS_DB = 'sessions'

# variables with varying values
SESSIONS_PER_FRAGMENT = 3
USER_DEVICES = 2
BOOKS_IN_PERIOD = 1
BOOKS_PER_USER = 20
SESSIONS_PAUSE = 5


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp // 1000)


def process_sessions_fields_to_int(collection):
    rootLogger.info('Process sessions fields to int')
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[collection].find()

    counter = 0
    sessions_num = sessions.count()
    for session in sessions:
        if session['user_id'] is not None:
            db[collection].update({'_id': session['_id']},
                                  {'$set':
                                       {'user_id': int(session['user_id'])}
                                   })
        else:
            db[collection].remove({'_id': session['_id']})
        counter += 1
        if counter % log_step == 0:
            rootLogger.info('Process %d/%d sessions' % (counter, sessions_num))

    return


def remove_duplicate_sessions(book_id):
    # Because logs have some duplicate sessions, we need to remove them
    rootLogger.info('Begin to remove duplicates')
    db_sessions = connect_to_mongo_database(BOOKS_DB)
    book_sessions = db_sessions[book_id].find(no_cursor_timeout=True)
    processed_sessions, removed_sessions = 0, 0
    db_sessions[book_id].create_index(
        [('_from', pymongo.ASCENDING), ('_to', pymongo.ASCENDING), ('item_id', pymongo.ASCENDING),
         ('user_id', pymongo.ASCENDING)])

    for session, index in zip(book_sessions, tqdm(range(book_sessions.count()))):
        duplicate_sessions = db_sessions[book_id].find({'_from': session['_from'],
                                                        '_to': session['_to'],
                                                        'item_id': session['item_id'],
                                                        'user_id': session['user_id']})
        for duplicate_session in duplicate_sessions:
            if duplicate_session['_id'] != session['_id']:
                if abs(float((session['read_at'] - duplicate_session['read_at']).total_seconds())) <= 5:
                    db_sessions[book_id].remove({'_id': duplicate_session['_id']})
                    removed_sessions = removed_sessions + 1
        processed_sessions = processed_sessions + 1
    rootLogger.info('Remove {%d} duplicates' % removed_sessions)


def process_sessions_to_book_percent_scale(book_id, update_old=False):
    rootLogger.info('Process sessions to book procent/symbols scales')
    db = connect_to_mongo_database(BOOKS_DB)

    if not update_old:
        sessions = db[book_id].find({'symbol_from': {'$not': {'$exists': True}}}, no_cursor_timeout=True)
    else:
        sessions = db[book_id].find(no_cursor_timeout=True)
    items = db['%s_items' % book_id]
    rootLogger.info('Found {%s} sessions' % sessions.count())

    book_symbols_num = db['books'].find_one({'_id': book_id})['symbols_num']

    for session, index in zip(sessions, tqdm(range(sessions.count()))):
        session_item = items.find_one({'id': session['item_id']})
        if session_item is None:
            rootLogger.info('Find None item with id {%s}' % session['item_id'])
            # Better to remove such sessions because we can't do anything with them
            db['%s' % book_id].remove({'_id': session['_id']})
            continue

        # some magic with commas in database
        session['_from'] = str(session['_from']).replace(',', '.')
        session['_from'] = float(session['_from'])
        session['_to'] = str(session['_to']).replace(',', '.')
        session['_to'] = float(session['_to'])

        try:
            item_percent_in_book = (session_item['_to'] - session_item['_from']) / 100
            book_from = float(session_item['_from']) + float(session['_from']) * item_percent_in_book
            book_to = float(session_item['_from']) + float(session['_to']) * item_percent_in_book

            symbol_from = int(book_symbols_num * book_from / 100)
            symbol_to = int(book_symbols_num * book_to / 100)
            db['%s' % book_id].update({'_id': session['_id']},
                                      {'$set': {'book_from': book_from,
                                                'book_to': book_to,
                                                'symbol_from': symbol_from,
                                                'symbol_to': symbol_to
                                                }})
        except Exception as e:
            db[book_id].remove({'_id': session['_id']})


def remove_by_begin(book_id, user_ids, remove=True):
    rootLogger.info('Begin to remove by begin')
    db = connect_to_mongo_database(BOOKS_DB)
    db[book_id].create_index([('user_id', pymongo.ASCENDING)])
    # users = db[book_id].distinct('user_id')
    remove_counter = 0
    for user_id in user_ids:
        # clear users who begin to read at > 10% of the book
        min_position = 100.0
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            if 'book_from' not in session:
                db[book_id].remove({'_id': session['_id']})
            elif float(session['book_from']) < min_position:
                min_position = session['book_from']
        if remove and min_position > 5.0:
            remove_counter += 1
            user_sessions = db[book_id].find({'user_id': user_id})
            for session in user_sessions:
                db[book_id].remove({'_id': session['_id']})
        db['%s_users' % book_id].update({'_id': user_id},
                                        {'$set': {'min_position': min_position}})
    if remove:
        rootLogger.info('Removed by reading after begin: [%d]' % remove_counter)


def process_session_per_fragment(book_id):
    rootLogger.info('Begin to define the number of sessions/fragment')
    db = connect_to_mongo_database(BOOKS_DB)
    users = db[book_id].distinct('user_id')

    for user_id, index in zip(users, tqdm(range(len(users)))):
        fragments = dict()
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            if '%d_%d' % (session['symbol_from'], session['symbol_to']) not in fragments:
                fragments['%d_%d' % (session['symbol_from'], session['symbol_to'])] = 1
                db[book_id].update({'_id': session['_id']},
                                   {'$set': {'status': 'reading'}})
            else:
                fragments['%d_%d' % (session['symbol_from'], session['symbol_to'])] += 1
                db[book_id].update({'_id': session['_id']},
                                   {'$set': {'status': 'returning'}})
        max_per_fragment = 0
        for fragment in fragments:
            max_per_fragment = max(max_per_fragment, fragments[fragment])
        if db['%s_users' % book_id].find({'_id': user_id}).count() != 0:
            db['%s_users' % book_id].update_one({'_id': user_id},
                                                {'$set': {'max_per_fragment': max_per_fragment}})
        else:
            db['%s_users' % book_id].insert_one({'_id': user_id,
                                                 'max_per_fragment': max_per_fragment})


def remove_by_devices(book_id, user_ids, remove=True):
    rootLogger.info('Begin to remove by devices')
    db = connect_to_mongo_database(BOOKS_DB)
    full_sessions = connect_to_mongo_database(FULL_SESSIONS_DB)
    # users = db[book_id].distinct('user_id')
    remove_counter = 0

    for user_id in user_ids:
        if db['%s_users' % book_id].find({'_id': user_id}).count != 0:
            user = db['%s_users' % book_id].find_one({'_id': user_id})
            if 'user_devices' in user:
                user_devices = user['user_devices']
        else:
            user_devices = len(full_sessions['sessions'].find({'user_id': user_id}).distinct('app_user_agent'))
            db['%s_users' % book_id].update_one({'_id': user_id},
                                                {'$set': {'user_devices': user_devices}})
        if remove and user_devices > USER_DEVICES:
            remove_counter += 1
            user_sessions = db[book_id].find({'user_id': user_id})
            for session in user_sessions:
                db[book_id].remove({'_id': session['_id']})

    if remove:
        rootLogger.info('Remove users with multiple devices: [%d]' % remove_counter)


def remove_by_parallel_reading(book_id, user_ids, remove=True):
    rootLogger.info('Begin to remove by parallel reading')
    db = connect_to_mongo_database(BOOKS_DB)
    full_sessions = connect_to_mongo_database(FULL_SESSIONS_DB)
    # users = db[book_id].distinct('user_id')
    remove_counter = 0

    for user_id in user_ids:
        book_sessions = db[book_id].find({'user_id': user_id}).sort('read_at', pymongo.ASCENDING).distinct('read_at')
        min_date = book_sessions[0]
        max_date = book_sessions[len(book_sessions) - 1]
        books_in_period = len(
            full_sessions['sessions'].find({'read_at': {'$gte': datetime.datetime.timestamp(min_date) * 1000,
                                                        '$lte': datetime.datetime.timestamp(max_date) * 1000},
                                            'user_id': user_id}).distinct('book_id'))
        db['%s_users' % book_id].update_one({'_id': user_id},
                                            {'$set': {'books_in_period': books_in_period}})
        if remove and books_in_period > BOOKS_IN_PERIOD:
            remove_counter += 1
            user_sessions = db[book_id].find({'user_id': user_id})
            for session in user_sessions:
                db[book_id].remove({'_id': session['_id']})

    if remove:
        rootLogger.info('Remove users with parallel books: [%d]' % remove_counter)


def remove_by_absolute_num_of_books(book_id, user_ids, remove=True):
    rootLogger.info('Remove users with big number of books per whole period')
    db = connect_to_mongo_database(BOOKS_DB)
    full_sessions = connect_to_mongo_database(FULL_SESSIONS_DB)
    # users = db[book_id].distinct('user_id')
    remove_counter = 0

    for user_id in user_ids:
        user_books = len(full_sessions['sessions'].find({'user_id': user_id}).distinct('book_id'))
        db['%s_users' % book_id].update_one({'_id': user_id},
                                            {'$set': {'user_books': user_books}})
        if remove and user_books > BOOKS_PER_USER:
            user_sessions = db[book_id].find({'user_id': user_id})
            remove_counter += 1
            for session in user_sessions:
                db[book_id].remove({'_id': session['_id']})

    if remove:
        rootLogger.info('Remove by absolute number of the books: [%d]' % remove_counter)


def clear_book_users(book_id, user_ids):
    rootLogger.info('Begin to clear book users')
    remove_by_begin(book_id, user_ids)
    # remove_by_devices(book_id) # TODO
    # remove_by_parallel_reading(book_id) # TODO
    # remove_by_absolute_num_of_books(book_id, user_ids)


def convert_sessions_time(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find()
    processed = 0
    for session in sessions:
        read_at_timestamp = date_from_timestamp(session['read_at'])
        db[book_id].update({'_id': session['_id']},
                           {'$set': {'read_at': read_at_timestamp}})
        processed += 1
        if processed % log_step == 0:
            rootLogger.info('Processed {%d} sessions.' % processed)


def validate_by_sessions_count(book_id, users_id):
    db = connect_to_mongo_database(BOOKS_DB)
    validated_users = []
    for user_id in users_id:
        sessions = db[book_id].find({'user_id': user_id})
        user_sessions_number = sessions.count()
        if user_sessions_number > 0:
            validated_users.append(user_id)
    return validated_users


def calculate_session_speed(book_id, user_id, max_speed=1200):
    # Calculation of speed for every micro-session
    # rootLogger.info ('Calculate session speed')
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find({'user_id': user_id}).sort('read_at')

    total_symbols = 0
    total_time = 0
    session_break = 5  # in minutes

    user_sessions_number = sessions.count()
    if user_id is not None and user_sessions_number == 0:
        rootLogger.info('There are no session for user [%d]' % user_id)
        return

    unknown_sessions_number = 0
    previous_session = -1
    for session in sessions:
        if previous_session == -1:
            previous_session = session
            db[book_id].update({'_id': previous_session['_id']},
                               {'$set': {'speed': -1,
                                         'category': 'normal'}})
            continue

        stats = dict()
        stats['session_time'] = -1
        if int(session['size']) > 0:
            stats['symbols'] = session['symbol_to'] - session['symbol_from']
            stats['session_time'] = float((session['read_at'] - previous_session['read_at']).total_seconds() / 60)
            if stats['session_time'] >= session_break:
                stats['speed'] = -1
                unknown_sessions_number += 1
            elif stats['session_time'] != 0:
                stats['speed'] = stats['symbols'] / stats['session_time']
                if stats['speed'] <= max_speed:
                    # we don't need big (skip) speeds for average speed processing
                    total_symbols += stats['symbols']
                    total_time += stats['session_time']
                    stats['category'] = 'normal'
                else:
                    stats['category'] = 'skip'
            else:
                db[book_id].remove({'_id': session['_id']})
        previous_session = session
        db[book_id].update({'_id': session['_id']},
                           {'$set': stats})

    if unknown_sessions_number / user_sessions_number * 100 >= 50:
        # rootLogger.info('User [%d] has more then 50 percents of uninterpretable sessions, deleted' % user_id)
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            db[book_id].remove({'_id': session['_id']})
        rootLogger.info('Remove user [%d], too much unknown sessions' % user_id)
        return
    if total_time == 0:
        # rootLogger.info('User [%d] has only sessions with speed >= 1200 symbols/min, deleted' % user_id)
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            db[book_id].remove({'_id': session['_id']})
        rootLogger.info('Remove user [%d], total sessions time is zero')
        return

    avr_speed = total_symbols / total_time
    unknown_sessions = db[book_id].find({'speed': -1,
                                         'user_id': int(user_id)})
    for session in unknown_sessions:
        db[book_id].update({'_id': session['_id']},
                           {'$set': {'speed': avr_speed,
                                     'category': 'normal'}})

    rootLogger.info('{user_id} : {avr_speed}'.format(user_id=user_id, avr_speed=avr_speed))
    db['%s_users' % book_id].update({'_id': user_id},
                                    {'$set': {'avr_speed': avr_speed}})


def calculate_relative_speed(book_id, user_id):
    # Calculate relative speed for user and define sessions categories
    db = connect_to_mongo_database(BOOKS_DB)
    rootLogger.info('{user_id}'.format(user_id=user_id))
    if db['%s_users' % book_id].find_one({'$and': [{'_id': user_id}, {'avr_speed': {'$exists': True}}]}) is None:
        return
    user_avr_speed = db['%s_users' % book_id].find_one({'_id': user_id})['avr_speed']
    if user_avr_speed > 0:
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            if 'speed' in session:
                abs_speed = session['speed'] / user_avr_speed
                db[book_id].update({'_id': session['_id']},
                                   {'$set': {'abs_speed': abs_speed}})
            else:
                db[book_id].remove({'_id': session['_id']})


def define_borders_for_items(book_id):
    # define the percentage for every item inside the book
    db_books = connect_to_mongo_database(BOOKS_DB)
    # move items from common collection to separate
    if ('%s_items' % book_id) not in db_books.collection_names():
        book_items = db_books['items'].find({'book_id': int(book_id)})
        for book_item in book_items:
            db_books['%s_items' % book_id].insert(book_item)
    else:
        rootLogger.info('Book {%s} done' % book_id)
        return

    # define book percents
    rootLogger.info('Define items percents/symbols')
    book = db_books['books'].find_one({'_id': book_id})
    symbols_num = book['symbols_num']
    documents = db_books['%s_items' % book_id].distinct('document_id')
    for document_id, index in zip(documents, range(len(documents))):
        document_items = db_books['%s_items' % book_id].find({'document_id': document_id})
        document_size = 0
        for item in document_items:
            document_size += item['media_file_size']
        _from, _to = 0, 0
        document_items = db_books['%s_items' % book_id].find({'document_id': document_id}).sort('position')
        for item in document_items:
            _from = _to
            _to = _from + item['media_file_size'] / document_size
            symbol_from = math.ceil(_from * symbols_num)
            symbol_to = math.ceil(_to * symbols_num)
            db_books['%s_items' % book_id].update({'_id': item['_id']},
                                                  {'$set': {'_from': _from * 100.0,
                                                            '_to': _to * 100.0,
                                                            'symbol_from': symbol_from,
                                                            'symbol_to': symbol_to}})
    rootLogger.info('Book {%s} done' % book_id)
    rootLogger.info('Inserted %d items' % db_books['%s_items' % book_id].find().count())


def get_target_users_by_borders(book_id, begin=10.0, end=80.0):
    # return list of users, who begin to read the book before begin(10.0) and end after end(80.0)
    rootLogger.info('Get target users...')
    db_sessions = connect_to_mongo_database(BOOKS_DB)
    try:
        db_sessions[book_id].create_index([('user_id', pymongo.ASCENDING), ('book_to', pymongo.ASCENDING)])
        db_sessions[book_id].create_index([('user_id', pymongo.ASCENDING), ('book_from', pymongo.ASCENDING)])
    except:
        rootLogger.info('Indexes are already created')

    target_users = list()
    users_id = db_sessions[book_id].distinct('user_id')
    seen_users = 0
    if ('%s_users' % book_id) not in db_sessions.collection_names():
        db_sessions.create_collection('%s_users' % book_id)
    users_collection = db_sessions['%s_users' % book_id]
    rootLogger.info('Found {%d} user ids' % len(users_id))
    for user_id in users_id:
        if user_id is None:
            continue
        if users_collection.find({'_id': user_id}).count() > 0:
            user = users_collection.find_one({'_id': user_id})
            if user['end_sessions'] > 0 and user['begin_sessions'] > 0:
                target_users.append(user_id)
            continue

        user = dict()
        end_sessions_count = db_sessions[book_id].find({'user_id': user_id,
                                                        'book_to': {'$gte': end}}).count()
        begin_sessions_count = db_sessions[book_id].find({'user_id': user_id,
                                                          'book_from': {'$lte': begin}}).count()
        center_sessions_count = db_sessions[book_id].find({'user_id': user_id,
                                                           'book_to': {'$gte': 10.0}}).count()

        user['_id'] = user_id
        user['end_sessions'] = end_sessions_count
        user['begin_sessions'] = begin_sessions_count
        user['center_sessions'] = center_sessions_count

        users_collection.insert(user)
        if end_sessions_count > 0 and begin_sessions_count > 0:
            target_users.append(user_id)
            # rootLogger.info('Found %d target users' % len(target_users))

        seen_users += 1
        if seen_users % 500 == 0:
            rootLogger.info('Process %d/%d users' % (seen_users, len(users_id)))

    rootLogger.info('Return {%d} target users' % len(target_users))
    return target_users


def process_users_book_percent_coverage(book_id):
    rootLogger.info('Calculate users percent coverage for the book...')
    db = connect_to_mongo_database(BOOKS_DB)
    users_id = db[book_id].distinct('user_id')

    for user_id, index in zip(users_id, tqdm(range(len(users_id)))):
        read_symbols, read_percents = 0, 0.0
        user_sessions = db[book_id].find({'user_id': user_id,
                                          'status': 'reading'})
        for session in user_sessions:
            read_percents += (session['book_to'] - session['book_from'])
            read_symbols += (session['symbol_to'] - session['symbol_from'])

        if db['%s_users' % book_id].find({'_id': user_id}).count() == 0:
            db['%s_users' % book_id].insert({'_id': user_id,
                                             'read_symbols': read_symbols,
                                             'read_percents': read_percents})
        else:
            db['%s_users' % book_id].update({'_id': user_id},
                                            {'$set':
                                                 {'read_symbols': read_symbols,
                                                  'read_percents': read_percents}})


def verify_users_book_procent_coverage(book_id, users_id):
    rootLogger.info('Begin to verify users percent coverage for the book')
    db = connect_to_mongo_database(BOOKS_DB)
    db[book_id].create_index([('read_at', pymongo.ASCENDING)])
    book_symbols = connect_to_mongo_database(BOOKS_DB)['books'].find_one({'_id': book_id})['symbols_num']
    target_users = []

    for user_id, index in zip(users_id, tqdm(range(len(users_id)))):
        user_profile = db['%s_users' % book_id].find_one({'_id': user_id})
        if 'reading' in user_profile and 'returning' in user_profile and 'sessions' in user_profile:
            if user_profile['returning'] / user_profile['sessions'] < 0.2:
                target_users.append(user_id)
            continue

        sessions = db[book_id].find({'user_id': user_id}).sort('read_at')
        sessions_num = sessions.count()
        if sessions_num == 0:
            rootLogger.info('User [%d] has zero sessions, skip' % user_id)
            continue
        segment_tree = SegmentTree(0, book_symbols)
        returning_percent = 0
        for session in sessions:
            if session['symbol_from'] >= session['symbol_to']:
                db[book_id].remove({'_id': session['_id']})
                continue
            sum = segment_tree.query_sum(session['symbol_from'] + 1, session['symbol_to'])
            if sum != 0 and sum <= session['symbol_to'] - session['symbol_from']:
                db[book_id].update({'_id': session['_id']},
                                   {'$set': {'status': 'returning'}})
                returning_percent += session['book_to'] - session['book_from']
            else:
                db[book_id].update({'_id': session['_id']},
                                   {'$set': {'status': 'reading'}})
                # for symbol in (session['symbol_from'], session['symbol_to']):
                #     segment_tree.update(1, 0, book_symbols - 1, symbol, 1)
                segment_tree.add(session['symbol_from'] + 1, session['symbol_to'], 1)

        reading_num = db[book_id].find({'user_id': user_id, 'status': 'reading'}).count()
        returning_num = db[book_id].find({'user_id': user_id, 'status': 'returning'}).count()
        db['%s_users' % book_id].update({'_id': user_id},
                                        {'$set': {'reading': reading_num,
                                                  'sessions': sessions_num,
                                                  'returning': returning_num}})

        if returning_num / sessions_num < 0.2:
            target_users.append(user_id)

    rootLogger.info('Target users number after verification of coverage: [%d]', len(target_users))
    return target_users


def get_target_users_by_percent_coverage(book_id, min_percent_coverage, max_percent_coverage):
    db = connect_to_mongo_database(BOOKS_DB)
    users = db['%s_users' % book_id].find({'read_percents':
                                               {'$gte': min_percent_coverage,
                                                '$lte': max_percent_coverage}}).distinct('_id')
    rootLogger.info('Return {%d} target users' % len(users))
    return users


def get_book_users(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    users = db[book_id].find().distinct('user_id')
    return users


def get_book_items(book_id):
    db = connect_to_mongo_database('readings')
    items = db[book_id].find().distinct('item_id')
    return items


def check_null_sessions(book_id):
    db_books = connect_to_mongo_database(BOOKS_DB)
    pages = db_books['%s_pages' % book_id].find()

    for page in pages:
        if 'sessions' in page:
            page_copy = page
            session_users = list(page['sessions'].keys())
            for user in session_users:
                if len(page['sessions'][user]) == 0:
                    del page_copy['sessions'][user]
            db_books['%s_pages' % book_id].update({'_id': page['_id']},
                                                  {'$set': {'sessions': page_copy['sessions']}})
    return


def get_all_items_borders(book_id):
    #  Process each item for getting all possible borders inside them
    rootLogger.info('Begin to get all items borders')
    db = connect_to_mongo_database(BOOKS_DB)
    db[book_id].create_index([('item_id', pymongo.ASCENDING)])
    db['%s_items' % book_id].create_index([('id', pymongo.ASCENDING)])
    items = db[book_id].find().distinct('item_id')

    counter = 1
    for item_id, index in zip(items, tqdm(range(len(items)))):
        if counter % 1000 == 0:
            rootLogger.info('Processing %d/%d items' % (counter, len(items)))
        item_borders = list()
        item_sessions = db[book_id].find({'item_id': item_id})
        sessions_count = item_sessions.count()
        for session in item_sessions:
            if 'symbol_to' not in session:
                db['%s_sesions' % book_id].remove({'_id': session['_id']})
            elif session['symbol_to'] not in item_borders:
                item_borders.append(session['symbol_to'])
            elif session['symbol_from'] not in item_borders:
                item_borders.append(session['symbol_from'])
        db['%s_items' % book_id].update({'id': item_id},
                                        {'$set':
                                             {'item_borders': item_borders,
                                              'sessions_count': sessions_count}
                                         })
        counter += 1


def set_sessions_borders(book_id, session_collection, drop_old=False, define_sessions_borders=False,
                         create_borders_collection=False):
    rootLogger.info('Begin to set sessions borders')
    db = connect_to_mongo_database(BOOKS_DB)
    db['%s_borders' % book_id].create_index([('symbol_from', pymongo.ASCENDING),
                                             ('symbol_to', pymongo.ASCENDING)]
                                            )

    all_borders = list()
    if not drop_old:
        all_borders = db['books'].find_one({'_id': book_id})['all_borders']
    else:
        rootLogger.info('Get all items borders')
        items = db['%s_items' % book_id].find()
        for item in items:
            if 'item_borders' not in item:
                continue
            all_borders.extend(item['item_borders'])
        rootLogger.info('Get all section borders')
        sections = db['%s_sections' % book_id].find({})
        for section in sections:
            if section['symbol_to'] not in all_borders:
                all_borders.append(section['symbol_to'])
        all_borders = list(set(all_borders))
        all_borders.sort()
        db['books'].update({'_id': book_id},
                           {'$set':
                                {'all_borders': all_borders}
                            })

    all_borders.sort()
    section_borders = db['%s_sections' % book_id].find().distinct('symbol_to')
    if create_borders_collection:
        rootLogger.info('Create borders collection')
        db['%s_borders' % book_id].drop()
        for i in range(0, len(all_borders) - 1):
            if all_borders[i + 1] not in section_borders:
                section_flag = False
            else:
                section_flag = True
            db['%s_borders' % book_id].insert_one({'_id': i,
                                                   'symbol_from': all_borders[i],
                                                   'symbol_to': all_borders[i + 1],
                                                   'section': section_flag})

    if define_sessions_borders:
        rootLogger.info('begin to define sessions borders')
        sessions = db[session_collection].find()
        for session, index in zip(sessions, tqdm(range(sessions.count()))):
            begin_border = bisect.bisect_left(all_borders, session['symbol_from'])
            end_border = bisect.bisect_left(all_borders, session['symbol_to']) - 1
            db[session_collection].update({'_id': session['_id']},
                                          {'$set': {
                                              'begin_border': begin_border,
                                              'end_border': end_border
                                          }})


def define_target_sessions(book_id, target_users):
    rootLogger.info('Select target users sessions...')
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find()
    db['%s_target' % book_id].drop()
    for session in sessions:
        if 'user_id' in session and session['user_id'] in target_users:
            db['%s_target' % book_id].insert(session)
    rootLogger.info('Insert %d target users sessions' % db['%s_target' % book_id].find().count())


def get_absolute_speeds_for_borders(book_id, sessions_collection):
    rootLogger.info('Begin to calculate absolute speed for sessions with normal category and reading status')
    db = connect_to_mongo_database(BOOKS_DB)
    borders_num = db['%s_borders' % book_id].find().count()

    borders_abs_speeds = [0 for i in range(borders_num)]
    borders_sessions_num = [0 for i in range(borders_num)]
    db[sessions_collection].create_index([('category', pymongo.ASCENDING)])

    sessions = db[sessions_collection].find({'category': 'normal'})
    for session in sessions:
        for border_id in range(session['begin_border'], session['end_border'] + 1):
            borders_abs_speeds[border_id] += session['abs_speed']
            borders_sessions_num[border_id] += 1

    rootLogger.info('Begin to update borders with absolute speed')
    abs_speeds = list()
    for border_id in range(0, len(borders_abs_speeds)):
        if borders_sessions_num[border_id] != 0:
            abs_speed = borders_abs_speeds[border_id] / borders_sessions_num[border_id]
        else:
            abs_speed = 0
        abs_speeds.append(abs_speed)
        db['%s_borders' % book_id].update({'_id': border_id},
                                          {'$set': {
                                              'abs_speed': abs_speed
                                          }})


def get_unusual_sessions_for_borders_by_length(book_id, sessions_collection):
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[sessions_collection].find()
    borders_num = db['%s_borders' % book_id].find().count()
    borders = [0 for i in range(borders_num)]

    for session in sessions:
        if 'long' in session and session['long']:
            for border_id in range(session['begin_border'], session['end_border'] + 1):
                borders[border_id] += 1

    for border_id in range(0, len(borders)):
        db['%s_borders' % book_id].update_one({'_id': border_id},
                                              {'$set': {
                                                  'unusual_sessions': borders[border_id]
                                              }})


def aggregate_borders(book_id, symbols_num=1000):
    # Aggregate borders to the size of 1000 symbols. Update those borders, where every ~1000 symbols achieved
    rootLogger.info('Begin to aggregate borders')
    db = connect_to_mongo_database(BOOKS_DB)
    borders = db['%s_borders' % book_id].find().sort('_id')

    # get strict sections borders
    sections = db['%s_sections' % book_id].find().sort('_id')
    sections_borders = list()
    for section in sections:
        sections_borders.append(section['symbol_to'])

    page_symbols = 0
    # page_time = 0
    begin_page_id, end_page_id = 0, 0
    page = 1
    break_flag = False
    for border in borders:
        if border['abs_speed'] == 0:
            rootLogger.info('problem with border (id == %d), absolute speed is zero, skip' % border['_id'])
            continue
        page_symbols += border['symbol_to'] - border['symbol_from']
        if border['section'] or page_symbols >= symbols_num:
            break_flag = True

        if break_flag:
            end_page_id = border['_id']

            page_borders = db['%s_borders' % book_id].find({'_id': {'$gte': begin_page_id},
                                                            '$and': [{'_id': {'$lte': end_page_id}}]})
            page_speed = 0
            page_unusual_sessions = 0
            page_sessions = 0
            page_skip_sessions = 0
            page_return_sessions = 0
            for page_border in page_borders:
                symbols_part = ((page_border['symbol_to'] - page_border['symbol_from']) / page_symbols)
                page_speed += symbols_part * page_border['abs_speed']
                page_unusual_sessions += symbols_part * page_border['unusual_sessions']
                page_skip_sessions += symbols_part * page_border['skip_sessions']
                page_return_sessions += symbols_part * page_border['return_sessions']
                page_sessions += symbols_part * page_border['sessions']

            db['%s_borders' % book_id].update({'_id': border['_id']},
                                              {'$set': {'page_speed': page_speed,
                                                        'page_unusual_sessions': page_unusual_sessions,
                                                        'page_sessions': page_sessions,
                                                        'page_skip_sessions': page_skip_sessions,
                                                        'page_return_sessions': page_return_sessions,
                                                        'page_unusual_percent': page_unusual_sessions / page_sessions,
                                                        'page_skip_percent': page_skip_sessions / page_sessions,
                                                        'page_return_percent': page_return_sessions / page_sessions,
                                                        'page': page}})
            page_symbols = 0
            # page_time = 0
            page += 1
            begin_page_id = end_page_id + 1
            break_flag = False


def count_number_of_users(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find()
    users = list()

    for session in sessions:
        if 'user_id' in session:
            if session['user_id'] not in users:
                users.append(session['user_id'])

    rootLogger.info('Found %d users' % len(users))


def count_sessions_category_per_border(book_id, sessions_collection):
    rootLogger.info('Begin to count sessions categories per border')
    db = connect_to_mongo_database(BOOKS_DB)

    borders = db['%s_borders' % book_id].find()

    db[sessions_collection].create_index([('symbol_from', pymongo.ASCENDING),
                                          ('symbol_to', pymongo.ASCENDING),
                                          ('category', pymongo.ASCENDING)])
    db[sessions_collection].create_index([('symbol_from', pymongo.ASCENDING),
                                          ('symbol_to', pymongo.ASCENDING)])

    skips_per_border = [0 for i in range(borders.count())]
    returns_per_border = [0 for i in range(borders.count())]
    sessions_per_border = [0 for i in range(borders.count())]
    users = db[sessions_collection].find().distinct('user_id')

    for user_id in users:
        sessions = db[sessions_collection].find({'user_id': user_id}).sort('read_at', pymongo.ASCENDING)
        begin_border = sorted(sessions.distinct('begin_border'))[0]
        end_border = sorted(sessions.distinct('begin_border'))[len(sessions.distinct('begin_border')) - 1]
        seen_borders = list()
        # user sessions
        for session in sessions:
            # make fragments as seen
            for border_id in range(session['begin_border'], session['end_border'] + 1):
                seen_borders.append(border_id)
                sessions_per_border[border_id] += 1
            # skips
            if 'category' in session and session['category'] == 'skip':
                for border_id in range(session['begin_border'], session['end_border'] + 1):
                    skips_per_border[border_id] += 1
            # returns
            elif session['status'] == 'returning' and session['category'] == 'normal':
                for border_id in range(session['begin_border'], session['end_border'] + 1):
                    returns_per_border[border_id] += 1

        # find fragments that user didn't see and mark them as skip
        # unseen_borders = list(set(range(begin_border, end_border)) - set(seen_borders))
        # for border_id in unseen_borders:
        #     # мы добавляем скип-сессию и просто сессию, иначе количество скипов будет >100%
        #     sessions_per_border[border_id] += 1
        #     skips_per_border[border_id] += 1

    rootLogger.info('Begin to update borders collection')
    for border_id in range(0, len(skips_per_border)):
        db['%s_borders' % book_id].update({'_id': border_id},
                                          {'$set': {'skip_sessions': skips_per_border[border_id],
                                                    'return_sessions': returns_per_border[border_id],
                                                    'sessions': sessions_per_border[border_id]}})


def select_top_document_ids(book_id, top_n=3):
    """Select top N most popular documents and delete other sessions"""
    rootLogger.info('Select top %d documents and delete other sessions' % top_n)
    db = connect_to_mongo_database(BOOKS_DB)
    db[book_id].create_index([('document_id', pymongo.ASCENDING)])
    document_ids = db[book_id].find().distinct('document_id')

    docs_num = {}
    for id in document_ids:
        docs_num[id] = db[book_id].find({'document_id': id}).count()

    sorted_docs = [(k, docs_num[k]) for k in sorted(docs_num, key=docs_num.get, reverse=True)][0:top_n]
    popular_ids = []
    for doc_id in sorted_docs:
        popular_ids.append(doc_id[0])

    sessions = db[book_id].find()
    for session, index in zip(sessions, tqdm(range(sessions.count()))):
        if session['document_id'] not in popular_ids:
            db[book_id].remove({'_id': session['_id']})


def import_all_book_sessions(book_id):
    rootLogger.info('Begin to import sessions')
    start_time = timeit.default_timer()

    full_sessions_db = connect_to_mongo_database(FULL_SESSIONS_DB)
    books_db = connect_to_mongo_database(BOOKS_DB)
    books_db[book_id].drop()

    if '%s_save' % book_id in books_db.collection_names():
        book_sessions = books_db['%s_save' % book_id].find()
        for session, index in zip(book_sessions, tqdm(range(book_sessions.count()))):
            books_db[book_id].insert(session)
    else:
        book_sessions = full_sessions_db['sessions'].find({'book_id': int(book_id)})
        for session, index in zip(book_sessions, tqdm(range(book_sessions.count()))):
            if session['user_id'] is None:
                continue
            session['book_id'] = str(session['book_id'])
            session['read_at'] = datetime.datetime.fromtimestamp(session['read_at'] / 1000)
            books_db[book_id].insert(session)
            books_db['{book_id}_copy'.format(book_id=book_id)].insert(session)

    elapsed = timeit.default_timer() - start_time
    rootLogger.info('Inserted [%d] book sessions in %s seconds' % (book_sessions.count(), str(elapsed)))


def save_sessions(book_id):
    rootLogger.info('Save book sessions without duplicates and with top-3 popular documents')
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find()

    db['%s_save' % book_id].drop()
    for session in sessions:
        db['%s_save' % book_id].insert(session)


def collect_long_sessions(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    users = db[book_id].find().distinct('user_id')

    rootLogger.info('Aggregate sessions to long sessions')
    db['%s_long_sessions' % book_id].drop()
    for user_id, index in zip(users, tqdm(range(len(users)))):
        sessions = db[book_id].find({'user_id': user_id}).sort('read_at', pymongo.ASCENDING)
        prev_session = None
        long_session = {}
        for session in sessions:
            if prev_session is not None:
                long_session['end_border'] = session['end_border']
                long_session['symbol_to'] = session['symbol_to']
                long_session['book_to'] = session['book_to']
                long_session['small'].append(session['_id'])
                if abs(float((session['read_at'] - prev_session['read_at']).total_seconds())) <= SESSIONS_PAUSE * 60:
                    long_session['time'] += abs(float((session['read_at'] - prev_session['read_at']).total_seconds()))
                    prev_session = session
                else:
                    long_session['size'] = long_session['symbol_to'] - long_session['symbol_from']

                    db['%s_long_sessions' % book_id].insert(long_session)
                    long_session = session
                    long_session['small'] = []
                    long_session['time'] = 0
                    prev_session = session
            else:
                prev_session = session
                long_session = session
                long_session['time'] = 0.0
                long_session['small'] = []

    rootLogger.info('Detect session type short/long')
    users = db[book_id].find().distinct('user_id')
    for user_id, index in zip(users, tqdm(range(len(users)))):
        long_sessions = db['%s_long_sessions' % book_id].find({'user_id': user_id})
        long_sizes = []
        for long_session in long_sessions:
            long_sizes.append(long_session['time'])
        long_sizes = Counter(long_sizes)
        count = list(reversed(sorted(long_sizes.values())))
        size = list(reversed(sorted(long_sizes, key=long_sizes.get)))

        counter = 0
        usual_sizes = []
        for i in range(0, len(count)):
            if counter + count[i] <= 0.6 * sum(count):
                usual_sizes.append(size[i])
                counter += count[i]

        long_sessions = db['%s_long_sessions' % book_id].find({'user_id': user_id})
        for long_session in long_sessions:
            if long_session['time'] in usual_sizes:
                is_long = False
            else:
                is_long = True
            for session_id in long_session['small']:
                db[book_id].update({'_id': session_id},
                                   {'$set': {'long': is_long}})


def calculate_categories_stats(book_id, user_id):
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find({'user_id': user_id})
    normal, skip = 0, 0
    for session in sessions:
        if session['category'] == 'normal':
            normal += 1
        else:
            skip += 1
    db['%s_users' % book_id].update({'_id': user_id},
                                    {'$set': {'normal': normal,
                                              'skip': skip}})


def verify_users_by_skips(book_id, target_users, skip_percent=0.5):
    rootLogger.info('Verify users on skip percent [%.3f] in all sessions' % skip_percent)
    db = connect_to_mongo_database(BOOKS_DB)
    approved_users = []

    for user_id, index in zip(target_users, tqdm(range(len(target_users)))):
        user = db['%s_users' % book_id].find_one({'_id': user_id,
                                                  'status': 'reading'})
        if user is None:
            continue
        if user['skip'] / (user['skip'] + user['normal']) <= skip_percent:
            approved_users.append(user_id)
    rootLogger.info('Approved [%d] users with normal number of skips' % len(approved_users))
    return approved_users


def sessions_processing(book_id):
    rootLogger.info('Sessions for book [%s] process begin' % str(book_id))
    # sessions processing
    # import_all_book_sessions(book_id)
    # remove_duplicate_sessions(book_id)
    # select_top_document_ids(book_id, 3)
    # define_borders_for_items(book_id)
    process_sessions_to_book_percent_scale(book_id,
                                           update_old=True)
    get_all_items_borders(book_id)
    set_sessions_borders(book_id,
                         session_collection=book_id,
                         drop_old=True,
                         define_sessions_borders=True,
                         create_borders_collection=True)
    process_session_per_fragment(book_id)
    collect_long_sessions(book_id)
    save_sessions(book_id)


def users_processing(book_id):
    rootLogger.info('Users for book [%s] process begin' % str(book_id))
    process_users_book_percent_coverage(book_id)
    target_users = get_target_users_by_percent_coverage(book_id,
                                                        min_percent_coverage=80.0,
                                                        max_percent_coverage=120.0)

    clear_book_users(book_id, target_users)
    target_users = verify_users_book_procent_coverage(book_id, target_users)
    rootLogger.info('Begin to calculate users sessions speed')
    for user_id, index in zip(target_users, tqdm(range(len(target_users)))):
        calculate_session_speed(book_id, user_id, max_speed=1200)
        calculate_relative_speed(book_id, user_id)
        calculate_categories_stats(book_id, user_id)

    target_users = verify_users_by_skips(book_id, target_users, skip_percent=0.2)
    define_target_sessions(book_id, target_users)
    target_sessions_collection = str(book_id) + '_target'

    # sessions stats
    count_sessions_category_per_border(book_id, target_sessions_collection)
    get_absolute_speeds_for_borders(book_id, target_sessions_collection)
    get_unusual_sessions_for_borders_by_length(book_id, target_sessions_collection)


def full_book_process(book_id):
    sessions_processing(book_id)
    users_processing(book_id)


def custom_book_process(book_id):
    rootLogger.info('Import for book [%s] process begin' % str(book_id))
    # sessions processing
    import_all_book_sessions(book_id)
    rootLogger.info('Remove duplicate for book [%s] process begin' % str(book_id))
    remove_duplicate_sessions(book_id)
    rootLogger.info('Define borders for book [%s] process begin' % str(book_id))
    # select_top_document_ids(book_id, 3)
    define_borders_for_items(book_id)
    rootLogger.info('Process sessions to percent scale for book [%s] process begin' % str(book_id))
    process_sessions_to_book_percent_scale(book_id,
                                           update_old=True)
    rootLogger.info('Get all items for book [%s] process begin' % str(book_id))
    get_all_items_borders(book_id)
    rootLogger.info('Set session borders for book [%s] process begin' % str(book_id))
    set_sessions_borders(book_id,
                         session_collection=book_id,
                         drop_old=True,
                         define_sessions_borders=True,
                         create_borders_collection=True)
    process_session_per_fragment(book_id)
    collect_long_sessions(book_id)
    rootLogger.info('Saving sessions for book [%s] process begin' % str(book_id))
    save_sessions(book_id)
    rootLogger.info("Sessions for book [%s] process end" % str(book_id))

    rootLogger.info('Users for book [%s] process begin' % str(book_id))
    process_users_book_percent_coverage(book_id)
    target_users = get_target_users_by_percent_coverage(book_id,
                                                        min_percent_coverage=50.0,
                                                        max_percent_coverage=150.0)

    clear_book_users(book_id, target_users)
    target_users = verify_users_book_procent_coverage(book_id, target_users)
    rootLogger.info('Begin to calculate users sessions speed')
    for user_id, index in zip(target_users, tqdm(range(len(target_users)))):
        calculate_session_speed(book_id, user_id, max_speed=1200)
        calculate_relative_speed(book_id, user_id)
        calculate_categories_stats(book_id, user_id)


def load_users(book_ids):
    for book_id in book_ids:
        full_sessions_db = connect_to_mongo_database(FULL_SESSIONS_DB)
        book_sessions = full_sessions_db['several_books_{book_id}'.format(book_id=book_id)].find()
        users = set()
        for session in book_sessions:
            users.add(session["user_id"])
        for user in users:
            full_sessions_db['users_{book_id}'.format(book_id=book_id)].insert({'user_id': str(user)})


def load_items(book_ids):
    for book_id in book_ids:
        full_sessions_db = connect_to_mongo_database(FULL_SESSIONS_DB)
        book_sessions = full_sessions_db['items'].find({'book_id': int(book_id)})

        for session in book_sessions:
            full_sessions_db['items_{book_id}'.format(book_id=book_id)].insert(session)


def load_sessions():
    full_sessions_db = connect_to_mongo_database(FULL_SESSIONS_DB)
    book_sessions = full_sessions_db['several_books'].find()

    for session in book_sessions:
        full_sessions_db['several_books_{book_id}'.format(book_id=str(session["book_id"]))].insert(session)


def clever_process_users_book_percent_coverage(book_id):
    rootLogger.info('Calculate users percent coverage for the book...')
    db = connect_to_mongo_database(BOOKS_DB)
    users_id = db[book_id].distinct('user_id')

    for user_id, index in zip(users_id, tqdm(range(len(users_id)))):
        read_symbols, read_percents = 0, 0.0
        user_sessions = db[book_id].find({'user_id': user_id,
                                          'status': 'reading'})
        for session in user_sessions:
            read_percents += (session['book_to'] - session['book_from'])
            read_symbols += (session['symbol_to'] - session['symbol_from'])

        if db['%s_users' % book_id].find({'_id': user_id}).count() == 0:
            db['%s_users' % book_id].insert({'_id': user_id,
                                             'read_symbols': read_symbols,
                                             'read_percents': read_percents})
        else:
            db['%s_users' % book_id].update({'_id': user_id},
                                            {'$set':
                                                 {'read_symbols': read_symbols,
                                                  'read_percents': read_percents}})


def documents_processiong(books):
    for book_id, document_id in books:
        rootLogger.info('Sessions for book [%s] process begin' % str(book_id))
        # sessions processing
        import_all_book_sessions(book_id)
        rootLogger.info('All sessions for book [%s] loaded' % str(book_id))
        remove_duplicate_sessions(book_id)
        rootLogger.info('All duplicate sessions for book [%s] removed' % str(book_id))
        define_borders_for_items(book_id)
        rootLogger.info('Borders for book [%s] book_id defined' % str(book_id))
        process_sessions_to_book_percent_scale(book_id,
                                               update_old=True)
        get_all_items_borders(book_id)
        set_sessions_borders(book_id,
                             session_collection=book_id,
                             drop_old=True,
                             define_sessions_borders=True,
                             create_borders_collection=True)
        process_session_per_fragment(book_id)
        collect_long_sessions(book_id)
        save_sessions(book_id)
        clever_process_users_book_percent_coverage(book_id)

        """target_users = get_target_users_by_percent_coverage(book_id,
                                                            min_percent_coverage=60.0,
                                                            max_percent_coverage=100.0)

        clear_book_users(book_id, target_users)
        target_users = verify_users_book_procent_coverage(book_id, target_users)
        target_users = validate_by_sessions_count(book_id, target_users)
        rootLogger.info('Begin to calculate users sessions speed')
        for user_id, index in zip(target_users, tqdm(range(len(target_users)))):
            calculate_session_speed(book_id, user_id, max_speed=1200)
            calculate_relative_speed(book_id, user_id)
            calculate_categories_stats(book_id, user_id)

        target_users = verify_users_by_skips(book_id, target_users, skip_percent=0.2)
        define_target_sessions(book_id, target_users)"""


def import_all_cleared_sessions(book_id, collection_name):
    rootLogger.info('Begin to import sessions')
    start_time = timeit.default_timer()

    full_sessions_db = connect_to_mongo_database(FULL_SESSIONS_DB)
    books_db = connect_to_mongo_database(BOOKS_DB)
    books_db[book_id].drop()

    if '%s_save' % book_id in books_db.collection_names():
        book_sessions = books_db['%s_save' % book_id].find()
        for session, index in zip(book_sessions, tqdm(range(book_sessions.count()))):
            books_db[book_id].insert(session)
    else:
        book_sessions = full_sessions_db[collection_name].find({'book_id': int(book_id)})
        for session, index in zip(book_sessions, tqdm(range(book_sessions.count()))):
            if session['user_id'] is None:
                continue
            session['book_id'] = str(session['book_id'])
            session['read_at'] = datetime.datetime.fromtimestamp(session['read_at'] / 1000)
            books_db[book_id].insert(session)

    elapsed = timeit.default_timer() - start_time
    rootLogger.info('Inserted [%d] book sessions in %s seconds' % (book_sessions.count(), str(elapsed)))


def process_cleared_sessions(book_id='135089'):
    # rootLogger.info('Cleared sessions process started')
    # import_all_cleared_sessions('135089', 'clear_sessions')
    # remove_duplicate_sessions(book_id)
    # define_borders_for_items(book_id)
    # process_sessions_to_book_percent_scale(book_id,
    #                                        update_old=True)
    # get_all_items_borders(book_id)
    # set_sessions_borders(book_id,
    #                      session_collection=book_id,
    #                      drop_old=True,
    #                      define_sessions_borders=True,
    #                      create_borders_collection=True)
    # process_session_per_fragment(book_id)
    # collect_long_sessions(book_id)
    # save_sessions(book_id)

    # process_users_book_percent_coverage(book_id)
    target_users = get_target_users_by_percent_coverage(book_id,
                                                        min_percent_coverage=0.0,
                                                        max_percent_coverage=4000.0)
    # clear_book_users(book_id, target_users)
    # target_users = verify_users_book_procent_coverage(book_id, target_users)
    for user_id, index in zip(target_users, tqdm(range(len(target_users)))):
        if user_id == '':
            continue
        calculate_session_speed(book_id, user_id, max_speed=1200)
        calculate_relative_speed(book_id, user_id)
        calculate_categories_stats(book_id, user_id)
    """
    target_users = verify_users_by_skips(book_id, target_users, skip_percent=0.2)
    define_target_sessions(book_id, target_users)
    target_sessions_collection = str(book_id) + '_target'

    target_users = get_target_users_by_percent_coverage(book_id,
                                                        min_percent_coverage=0.0,
                                                        max_percent_coverage=100.0)

    clear_book_users(book_id, target_users)
    target_users = verify_users_book_procent_coverage(book_id, target_users)
    target_users = validate_by_sessions_count(book_id, target_users)
    rootLogger.info('Begin to calculate users sessions speed')
    for user_id, index in zip(target_users, tqdm(range(len(target_users)))):
        calculate_session_speed(book_id, user_id, max_speed=1200)
        calculate_relative_speed(book_id, user_id)
        calculate_categories_stats(book_id, user_id)

    target_users = verify_users_by_skips(book_id, target_users, skip_percent=0.2)
    define_target_sessions(book_id, target_users)
    """


def main():
    """
    book_ids = ['2206', '259222', '266700', '9297', '2543']
    for book_id in book_ids:
        rootLogger.info('Process for DB [%s]' % BOOKS_DB)
        full_book_process(book_id)
        aggregate_borders(book_id, symbols_num=1000)

    book_ids = ['135089']

    for book_id in book_ids:
        rootLogger.info('Process for DB [%s]' % BOOKS_DB)
        custom_book_process(book_id)

    load_items(['2207', '11833', '24304', '24305', '135089', '210901', '329267'])
    clever_processiong(['135089'])
     """
    pass


if __name__ == "__main__":
    process_cleared_sessions()
