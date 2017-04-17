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
    db = connect_to_mongo_database(READINGS_DB)
    sessions_collection = db[book_id]
    sessions = db[book_id].find({'user_id': user_id})

    total_symbols = 0
    total_time = 0

    for session in sessions:
        previous_session = sessions_collection.find({'_to': session['_from'],
                                                     'user_id': user_id,
                                                     'read_at': { '$lte': session['read_at']}})
        stats = dict()
        stats['session_time'] = -1
        if previous_session.count() > 0:
            previous_session = sessions_collection.find_one({'_to': session['_from'],
                                                             'user_id': user_id})
            if session['size'] > 0:
                stats['symbols'] = session['size']
                stats['session_time'] = (float)((session['read_at'] - previous_session['read_at']).total_seconds() / 60)
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


def process_sessions_to_pages(book_id, user_id):
    if user_id == 'None':
        print ('User ID cannot be None!')
        return

    db_sessions = connect_to_mongo_database(READINGS_DB)
    db_books = connect_to_mongo_database(BOOKS_DB)

    # FIXME remove this after successful work
    # copy collection for easy work
    book_sessions_collection = book_id + '_sessions'
    book_pages = db_books[book_id + '_pages'].find()

    for page in book_pages:
        # find session which begins at this page and can be full in page or can end in another next pages
        pages_sessions = db_sessions[book_id].find({'user_id': user_id,
                                                    '_from': { '$gte': page['from_percent']},
                                                    '_from': {'$lt': page['to_percent']}})
        if 'sessions' not in page:
            page['sessions'] = {}
        if user_id not in page['sessions']:
            page['sessions'][user_id] = list()

        for initial_session in pages_sessions:
            session = dict()
            session['_from'] = initial_session['_from']
            session['speed'] = initial_session['speed']
            if initial_session['_to'] < page['to_percent']:
                session['_to'] = initial_session['_to']
            else:
                session['_to'] = page['to_percent']
            session['size'] = session['_to'] - session['_from']

            page['sessions'][user_id].append(session)

        # find sessions that end at this page and begins at anothers
        pages_sessions = db_sessions[book_id].find({'user_id': user_id,
                                                    '_to': {'$gte': page['from_percent']},
                                                    '_to': {'$lt': page['to_percent']}})
        for initial_session in pages_sessions:
            session = dict()
            if initial_session['_from'] >= page['from_percent']:
                # than no need to see at this session, because it begins at this page and we calculated it before
                pass
            if initial_session['_from'] < page['from_percent']:
                session['_from'] = page['from_percent']
                session['_to'] = initial_session['_to']
                session['size'] = session['_from'] - session['_to']
                session['speed'] = initial_session['speed']
            page['sessions'][user_id].append(session)

        # find sessions that contains full page in them
        pages_sessions = db_sessions[book_id].find({'user_id': user_id,
                                                    '_from': {'$lt': page['from_percent']},
                                                    '_to': {'$gt': page['to_percent']}})
        for initial_session in pages_sessions:
            session = dict()
            session['_from'] = page['from_percent']
            session['_to'] = page['to_percent']
            session['size'] = session['_to'] - session['_from']
            session['speed'] = initial_session['speed']
            page['sessions'][user_id].append(session)

        db_books[book_sessions_collection].remove({'_id': page['_id']})
        db_books[book_sessions_collection].insert(page)
        print ('Page {%s} updated with {%d} sessions.' % (page['_id'], pages_sessions.count()))


def get_book_users(book_id):
    db = connect_to_mongo_database('readings')
    users = db[book_id].find().distinct('user_id')
    return users


def get_book_items(book_id):
    db = connect_to_mongo_database('readings')
    items = db[book_id].find().distinct('item_id')
    return items


def filter_book_core_users(book_id):
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
control_users = get_book_users(book_id)[0:20]
print (control_users)
for user_id in control_users:
    if user_id != None:
        calculate_session_speed(book_id, user_id)
        print ('Session speed calculated for user {%s}' % user_id)
process_sessions_to_pages(book_id, control_users[1])


