from pymongo import MongoClient
import datetime


READINGS_DB = 'readings'
BOOKS_DB = 'bookmate'
log_step = 100000


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp//1000)


def remove_duplicate_sessions(book_id):
    # Because logs have some duplicate sessions, we need to remove them
    db_sessions = connect_to_mongo_database(READINGS_DB)
    book_sessions = db_sessions[book_id].find()
    processed_sessions, removed_sessions, log_step = 0, 0, 1000
    for session in book_sessions:
        duplicate_sessions = db_sessions[book_id].find({'_from': session['_from'],
                                                        '_to': session['_to'],
                                                        'item_id': session['item_id'],
                                                        'user_id': session['user_id']})
        for duplicate_session in duplicate_sessions:
            if duplicate_session['_id'] != session['_id']:
                db_sessions[book_id].remove({'_id': duplicate_session['_id']})
                removed_sessions = removed_sessions + 1
                if removed_sessions % log_step == 0:
                    print ('Remove {%d} duplicates' % removed_sessions)
        processed_sessions = processed_sessions + 1
        if processed_sessions % log_step == 0:
            print ('Process {%d} sessions' % processed_sessions)


def convert_sessions_time(book_id):
    db = connect_to_mongo_database(READINGS_DB)
    sessions = db[book_id].find()
    processed = 0
    for session in sessions:
        read_at_timestamp = date_from_timestamp(session['read_at'])
        db[book_id].update({'_id': session['_id']},
                            {'$set': {'read_at': read_at_timestamp}})
        processed += 1
        if processed % log_step == 0:
            print('Processed {%d} sessions.' % processed)


def calculate_session_speed(book_id, user_id):
    # Calculation of speed for every micro-session
    db = connect_to_mongo_database(READINGS_DB)
    sessions_collection = db['%s' % book_id]
    sessions = sessions_collection.find({'user_id': user_id})

    total_symbols = 0
    total_time = 0

    for session in sessions:
        previous_session = sessions_collection.find_one({'book_to': session['book_from'],
                                                         'user_id': user_id,
                                                         'read_at': {'$lte': session['read_at']}})
        stats = dict()
        stats['session_time'] = -1
        if previous_session is not None:
            if session['size'] > 0:
                stats['symbols'] = session['size']
                stats['session_time'] = float((session['read_at'] - previous_session['read_at']).total_seconds() / 60)
                stats['speed'] = stats['symbols'] / stats['session_time']

                total_symbols += stats['symbols']
                total_time += stats['session_time']

        sessions_collection.update({'_id': session['_id']},
                                   {'$set': stats})

    avr_speed = total_symbols / total_time
    unknown_sessions = sessions_collection.find({'session_time': -1})
    for session in unknown_sessions:
        sessions_collection.update({'_id': session['_id']},
                                   {'$set': {'speed': avr_speed}})


def define_percents_for_items(book_id):
    # define the percentage for every item inside the book
    db_books = connect_to_mongo_database(BOOKS_DB)
    documents = db_books['%s_items' % book_id].distinct('document_id')
    for document_id in documents:
        document_items = db_books['%s_items' % book_id].find({'document_id': document_id})
        document_size = 0
        for item in document_items:
            document_size += item['media_file_size']
        _from, _to = 0, 0
        document_items = db_books['%s_items' % book_id].find({'document_id': document_id}).sort('position')
        for item in document_items:
            _from = _to
            _to = _from + item['media_file_size'] / document_size
            db_books['%s_items' % book_id].update({'_id': item['_id']},
                                     {'$set': {'_from': _from * 100.0,
                                               '_to': _to * 100.0}})
    print ('Book {%s} done' % book_id)


def process_sessions_to_book_percent_scale(book_id):
    # process percents of sessions in item into percent of sessions in book
    db_sessions = connect_to_mongo_database(READINGS_DB)
    db_books = connect_to_mongo_database(BOOKS_DB)

    sessions = db_sessions['%s' % book_id].find({'book_from': {'$not': {'$exists': True}}})
    items = db_books['%s_items' % book_id]
    print ('Found {%s} sessions' % sessions.count())
    num_sessions, log_step = 0, 1000

    for session in sessions:
        session_item = items.find_one({'id': session['item_id']})
        if session_item is None:
            print ('Find None item with id {%s}' % session['item_id'])
            # Better to remove such sessions because we can't do anything with them
            db_sessions['%s' % book_id].remove({'_id': session['_id']})
            continue
        item_percent_in_book = (session_item['_to'] - session_item['_from']) / 100
        book_from = float(session_item['_from']) + float(session['_from']) * item_percent_in_book
        book_to = float(session_item['_from']) + float(session['_to']) * item_percent_in_book
        db_sessions['%s' % book_id].update({'_id': session['_id']},
                                          {'$set': {'book_from': book_from,
                                                    'book_to': book_to}})
        num_sessions = num_sessions + 1
        if num_sessions % log_step == 0:
            print ('{%d} sessions processed' % num_sessions)


def get_target_users(book_id, begin=10.0, end=80.0):
    # return list of users, who begin to read the book not after begin(10.0) and end not before end(80.0)
    db_sessions = connect_to_mongo_database(READINGS_DB)
    target_users = list()
    users_id = db_sessions[book_id].distinct('user_id')
    print ('Found {%d} user ids' % len(users_id))
    for user_id in users_id:
        end_sessions_count = db_sessions[book_id].find({'user_id': user_id,
                                                       'book_to': {'$gte': end}}).count()
        begin_sessions_count = db_sessions[book_id].find({'user_id': user_id,
                                                          'book_from': {'$lte': begin}}).count()
        if end_sessions_count > 0 and begin_sessions_count > 0:
            target_users.append(user_id)

    return users_id


def process_sessions_to_pages(book_id, user_id):
    if user_id == 'None':
        print('User ID cannot be None!')
        return

    db_sessions = connect_to_mongo_database(READINGS_DB)
    db_books = connect_to_mongo_database(BOOKS_DB)

    # FIXME remove this after successful work
    # copy collection for easy work
    book_sessions_collection = book_id + '_sessions_speed'
    book_pages = db_books[book_id + '_pages'].find()

    for page in book_pages:
        # find session which begins at this page and can be full in page or can end in another next pages
        pages_sessions = db_sessions['%s_sessions' % book_id].find({'user_id': user_id,
                                                    'book_from': {'$gte': page['_from']},
                                                    'book_from': {'$lt': page['_to']}})
        if 'sessions' not in page:
            page['sessions'] = {}
        if user_id not in page['sessions']:
            page['sessions'][user_id] = list()

        for initial_session in pages_sessions:
            session = dict()
            session['book_from'] = initial_session['book_from']
            session['speed'] = initial_session['speed']
            if initial_session['book_to'] < page['_to']:
                session['book_to'] = initial_session['book_to']
            else:
                session['book_to'] = page['_to']
            session['size'] = session['book_to'] - session['book_from']

            page['sessions'][user_id].append(session)

        # find sessions that end at this page and begins at anothers
        pages_sessions = db_sessions['%s_sessions' % book_id].find({'user_id': user_id,
                                                    'book_to': {'$gte': page['_from']},
                                                    'book_to': {'$lt': page['_to']}})
        for initial_session in pages_sessions:
            session = dict()
            if initial_session['book_from'] >= page['_from']:
                # than no need to see at this session, because it begins at this page and we calculated it before
                pass
            if initial_session['book_from'] < page['_from']:
                session['book_from'] = page['_from']
                session['book_to'] = initial_session['book_to']
                session['size'] = session['book_from'] - session['book_to']
                session['speed'] = initial_session['speed']
            page['sessions'][user_id].append(session)

        # find sessions that contains full page in them
        pages_sessions = db_sessions['%s_sessions' % book_id].find({'user_id': user_id,
                                                    'book_from': {'$lt': page['_from']},
                                                    'book_to': {'$gt': page['_to']}})
        for initial_session in pages_sessions:
            session = dict()
            session['book_from'] = page['_from']
            session['book_to'] = page['_to']
            session['size'] = session['book_to'] - session['book_from']
            session['speed'] = initial_session['speed']
            page['sessions'][user_id].append(session)

        db_books[book_sessions_collection].remove({'_id': page['_id']})
        db_books[book_sessions_collection].insert(page)
        print ('Page {%s} updated with {%d} sessions.' % (page['_id'], len(page['sessions'])))


def get_book_users(book_id):
    db = connect_to_mongo_database('readings')
    users = db[book_id].find().distinct('user_id')
    return users


def get_book_items(book_id):
    db = connect_to_mongo_database('readings')
    items = db[book_id].find().distinct('item_id')
    return items


def filter_book_core_users():
    # delete those users, who begin to read the book from more then 10% of the size
    db = connect_to_mongo_database(READINGS_DB)
    users = get_book_users(book_id)
    print ('Users in the {%s book} collection: {%d}' % (book_id, len(users)))
    for user_id in users:
        sessions = db[book_id].find({'user_id': user_id})
        min_position = 100.0
        for session in sessions:
            if float(session['_from']) < min_position:
                min_position = session['_from']
        if min_position > 10.0:
            db[book_id].remove({'user_id': user_id})
            print ('User {%s} was removed' % user_id)

    users = get_book_users(book_id)
    print('Users in the {%s book} collection: {%d}' % (book_id, len(users)))


book_id = '210901'
user_id = '1746639'

process_sessions_to_book_percent_scale(book_id)
# calculate_session_speed(book_id=book_id, user_id=user_id)



# remove_duplicate_sessions(book_id)
# control_users = get_book_users(book_id)[0:20]
# print (control_users)
# for user_id in control_users:
#     if user_id != None:
#         calculate_session_speed(book_id, user_id)
#         print ('Session speed calculated for user {%s}' % user_id)
# process_sessions_to_pages(book_id, control_users[1])
# define_percents_for_items(book_id=book_id)
# process_sessions_to_book_percent_scale(book_id=book_id)
# calculate_session_speed(book_id=book_id, user_id=user_id)
# process_sessions_to_pages(book_id=book_id, user_id=user_id)
