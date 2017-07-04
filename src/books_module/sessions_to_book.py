from pymongo import MongoClient
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pymongo

READINGS_DB = 'readings'
BOOKS_DB = 'bookmate'
log_step = 100000


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp // 1000)


def remove_duplicate_sessions(book_id):
    # Because logs have some duplicate sessions, we need to remove them
    db_sessions = connect_to_mongo_database(BOOKS_DB)
    book_sessions = db_sessions[book_id].find()
    processed_sessions, removed_sessions, log_step = 0, 0, 1000
    try:
        db_sessions[book_id].create_index([('_from', pymongo.ASCENDING), ('_to', pymongo.ASCENDING), ('item_id', pymongo.ASCENDING), ('user_id', pymongo.ASCENDING)])
    except Exception as e:
        print ('Exception in index creating for duplicated sessions')
        print(e)

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
                    print('Remove {%d} duplicates' % removed_sessions)
        processed_sessions = processed_sessions + 1
        if processed_sessions % log_step == 0:
            print('Process {%d} sessions' % processed_sessions)


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
    db = connect_to_mongo_database(BOOKS_DB)
    sessions_collection = db[book_id]
    sessions = sessions_collection.find({'user_id': user_id}).sort('read_at')
    previous_session = sessions.next()
    sessions_collection.update({'_id': previous_session['_id']},
                               {'$set': {'speed': -1}})

    total_symbols = 0
    total_time = 0
    session_break = 5  # in minutes

    for session in sessions:
        stats = dict()
        stats['session_time'] = -1
        if previous_session is not None:
            if int(session['size']) > 0:
                stats['symbols'] = session['size']
                stats['session_time'] = float((session['read_at'] - previous_session['read_at']).total_seconds() / 60)
                if stats['session_time'] > session_break:
                    stats['speed'] = -1
                else:
                    try:
                        stats['speed'] = stats['symbols'] / stats['session_time']
                        total_symbols += stats['symbols']
                        total_time += stats['session_time']
                    except ZeroDivisionError:
                        pass
        previous_session = session

        sessions_collection.update({'_id': session['_id']},
                                   {'$set': stats})

    avr_speed = total_symbols / total_time
    unknown_sessions = sessions_collection.find({'speed': -1,
                                                 'user_id': user_id})
    for session in unknown_sessions:
        sessions_collection.update({'_id': session['_id']},
                                   {'$set': {'speed': avr_speed}})


def define_percents_for_items(book_id):
    # define the percentage for every item inside the book
    db_books = connect_to_mongo_database(BOOKS_DB)
    # move items from common collection to separate
    if ('%s_items' % book_id) not in db_books.collection_names():
        book_items = db_books['items'].find({'book_id': int(book_id)})
        for book_item in book_items:
            db_books['%s_items' % book_id].insert(book_item)
    else:
        print('Book {%s} done' % book_id)
        return

    # define book percents
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
    print('Book {%s} done' % book_id)


def process_sessions_to_book_percent_scale(book_id):
    # process percents of sessions in item into percent of sessions in book
    db_sessions = connect_to_mongo_database(BOOKS_DB)
    db_books = connect_to_mongo_database(BOOKS_DB)

    sessions = db_sessions['%s' % book_id].find({'book_from': {'$not': {'$exists': True}}}, no_cursor_timeout=True)
    items = db_books['%s_items' % book_id]
    print('Found {%s} sessions' % sessions.count())
    num_sessions, log_step = 0, 1000

    for session in sessions:
        session_item = items.find_one({'id': session['item_id']})
        if session_item is None:
            print('Find None item with id {%s}' % session['item_id'])
            # Better to remove such sessions because we can't do anything with them
            db_sessions['%s' % book_id].remove({'_id': session['_id']})
            continue

        if 'book_from' in session and 'book_to' in session:
            num_sessions = num_sessions + 1
            if num_sessions % log_step == 0:
                print('{%d} sessions processed' % num_sessions)
            continue

        # some magic with commas in database
        session['_from'] = str(session['_from']).replace(',', '.')
        session['_from'] = float(session['_from'])
        session['_to'] = str(session['_to']).replace(',', '.')
        session['_to'] = float(session['_to'])

        item_percent_in_book = (session_item['_to'] - session_item['_from']) / 100
        book_from = float(session_item['_from']) + float(session['_from']) * item_percent_in_book
        book_to = float(session_item['_from']) + float(session['_to']) * item_percent_in_book
        db_sessions['%s' % book_id].update({'_id': session['_id']},
                                           {'$set': {'book_from': book_from,
                                                     'book_to': book_to}})
        num_sessions = num_sessions + 1
        if num_sessions % log_step == 0:
            print('{%d} sessions processed' % num_sessions)


def get_target_users(book_id, begin=10.0, end=80.0):
    # return list of users, who begin to read the book not after begin(10.0) and end not before end(80.0)
    db_sessions = connect_to_mongo_database(BOOKS_DB)
    try:
        db_sessions[book_id].create_index([('user_id', pymongo.ASCENDING), ('book_to', pymongo.ASCENDING)])
        db_sessions[book_id].create_index([('user_id', pymongo.ASCENDING), ('book_from', pymongo.ASCENDING)])
    except:
        print('Indexes are already created')

    target_users = list()
    users_id = db_sessions[book_id].distinct('user_id')
    seen_users = 0
    if ('%s_users' % book_id) not in db_sessions.collection_names():
        db_sessions.create_collection('%s_users' % book_id)
    users_collection = db_sessions['%s_users' % book_id]
    print('Found {%d} user ids' % len(users_id))
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

        user['_id'] = user_id
        user['end_sessions'] = end_sessions_count
        user['begin_sessions'] = begin_sessions_count

        users_collection.insert(user)
        if end_sessions_count > 0 and begin_sessions_count > 0:
            target_users.append(user_id)
            print('Found %d target users' % len(target_users))

        seen_users += 1
        if seen_users % 100 == 0:
            print('Process %d/%d users' % (seen_users, len(users_id)))

    print('Return {%d} target users' % len(target_users))
    return target_users


def process_sessions_to_pages(book_id, user_id):
    db_sessions = connect_to_mongo_database(READINGS_DB)
    db_books = connect_to_mongo_database(BOOKS_DB)

    # TODO insert indexes
    # copy collection for easy work
    book_sessions_collection = book_id
    book_pages = db_books[book_sessions_collection].find()
    book_stats = db_books['books'].find_one({'_id': book_id})

    total_sessions = 0

    for page in book_pages:
        # find session which begins at this page and ends in another next pages
        pages_sessions = db_sessions[book_id].find({'user_id': user_id,
                                                    'book_from': {'$gte': page['_from'],
                                                                  '$lt': page['_to']},
                                                    'book_to': {'$gt': page['_to']},
                                                    'speed': {'$exists': True}})
        if 'sessions' not in page:
            page['sessions'] = {}
        if user_id not in page['sessions']:
            page['sessions'][user_id] = list()

        for initial_session in pages_sessions:
            session = dict()
            # FXME uncomment later
            # session['book_from'] = initial_session['book_from']
            session['speed'] = initial_session['speed']
            session['book_from'] = page['from']
            session['book_to'] = page['_to']
            # FIXME uncomment later
            # if initial_session['book_to'] < page['_to']:
            #     session['book_to'] = initial_session['book_to']
            # else:
            #     session['book_to'] = page['_to']
            session['size'] = session['book_to'] - session['book_from']
            session['symbols_num'] = session['size'] * book_stats['symbols_num'] / 100

            page['sessions'][user_id].append(session)
            total_sessions += 1

        # find sessions that end at this page and begins at anothers
        pages_sessions = db_sessions[book_id].find({'user_id': user_id,
                                                    'book_to': {'$gt': page['_from'],
                                                                '$lte': page['_to']},
                                                    'book_from': {'lt': page['_from']},
                                                    'speed': {'$exists': True}})
        for initial_session in pages_sessions:
            session = dict()
            if initial_session['book_from'] >= page['_from']:
                # than no need to see at this session, because it begins at this page and we calculated it before
                pass
            if initial_session['book_to'] < page['_from']:
                # FIXME uncomment later
                # session['book_from'] = page['_from']
                # session['book_to'] = initial_session['book_to']
                session['book_from'] = page['_from']
                session['book_to'] = page['_to']

                session['size'] = session['book_from'] - session['book_to']
                session['speed'] = initial_session['speed']
                session['symbols_num'] = session['size'] * book_stats['symbols_num'] / 100
                page['sessions'][user_id].append(session)
            total_sessions += 1

        # find sessions that contains full page in them
        pages_sessions = db_sessions[book_id].find({'user_id': user_id,
                                                    'book_from': {'$lte': page['_from']},
                                                    'book_to': {'$gte': page['_to']},
                                                    'speed': {'$exists': True}})
        for initial_session in pages_sessions:
            session = dict()
            session['book_from'] = page['_from']
            session['book_to'] = page['_to']
            session['size'] = session['book_to'] - session['book_from']
            session['speed'] = initial_session['speed']
            page['sessions'][user_id].append(session)
            session['symbols_num'] = session['size'] * book_stats['symbols_num'] / 100
            page['sessions'][user_id].append(session)
            total_sessions += 1

        db_books[book_sessions_collection].update({'_id': page['_id']},
                                                  {'$set': {'sessions': page['sessions']}})
        # print ('Page {%s} updated with {%d} sessions.' % (page['_id'], len(page['sessions'][user_id])))

    print('User {%s} updated book with {%d} sessions' % (user_id, total_sessions))


def count_average_page_speed(book_id):
    db_books = connect_to_mongo_database(BOOKS_DB)
    pages = db_books['%s' % book_id].find()

    average_user_speeds = list()
    for page in pages:
        if 'sessions' in page and len(page['sessions']) == 0:
            continue
        page['avr_speed'] = 0
        for user in page['sessions']:
            average_user_score = 0
            for user_session in page['sessions'][user]:
                if len(page['sessions'][user]) > 0:
                    average_user_score += \
                        (user_session['symbols_num'] / page['symbols_num']) * user_session['speed']
                    # FIXME fix the speed for 800 or 1000 symbols?
                    if average_user_score > 800:
                        continue
                page['avr_speed'] += average_user_score
                if average_user_score > 0:
                    average_user_speeds.append(average_user_score)

        page['avr_speed'] /= len(page['sessions'])
        db_books['%s_sessions_speed' % book_id].update({'_id': page['_id']},
                                                       {'$set': {'avr_speed': page['avr_speed']}})

    return average_user_speeds


def count_absolute_page_speed(book_id):
    db_books = connect_to_mongo_database(BOOKS_DB)
    pages = db_books['%s_sessions_speed' % book_id].find()
    absolute_user_speeds = list()

    avr_book_speed = 0
    for page in pages:
        avr_book_speed += page['avr_speed']

    pages = db_books['%s_sessions_speed' % book_id].find()
    avr_book_speed = avr_book_speed / pages.count()

    for page in pages:
        page['absolute_speed'] = page['avr_speed'] / avr_book_speed
        db_books['%s_sessions_speed' % book_id].update({'_id': page['_id']},
                                                       {'$set': {'absolute_speed': page['absolute_speed']}})
        absolute_user_speeds.append(page['absolute_speed'])

    return absolute_user_speeds


def count_who_see_the_page_plot(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    pages = db['%s_pages' % book_id].find().sort('_id', pymongo.ASCENDING)

    try:
        db[book_id].create_index([('book_from', pymongo.ASCENDING), ('book_to', pymongo.ASCENDING)])
    except:
        print('Some problem with index building')

    x, y = [list()] * 2
    for page in pages:
        x.append(page['_id'])
        users = db['%s' % book_id].find({'book_from': {'$gte': page['_from']},
                                         'book_to': {'$lte': page['_to']}
                                         }).distinct('user_id')
        users_count = len(users)
        db['%s_pages' % book_id].update({'_id': page['_id']},
                                        {'$set': {'seen_num': users_count}})
        y.append(users_count)

    plt.clf()
    plt.hist(x)
    plt.savefig('who_see_%s.png' % book_id, bbox_inches='tight')
    return


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
    print('Users in the {%s book} collection: {%d}' % (book_id, len(users)))
    for user_id in users:
        sessions = db[book_id].find({'user_id': user_id})
        min_position = 100.0
        for session in sessions:
            if float(session['_from']) < min_position:
                min_position = session['_from']
        if min_position > 10.0:
            db[book_id].remove({'user_id': user_id})
            print('User {%s} was removed' % user_id)

    users = get_book_users(book_id)
    print('Users in the {%s book} collection: {%d}' % (book_id, len(users)))


def check_null_sessions(book_id):
    db_books = connect_to_mongo_database(BOOKS_DB)
    pages = db_books['%s_sessions_speed' % book_id].find()

    for page in pages:
        if 'sessions' in page:
            page_copy = page
            session_users = list(page['sessions'].keys())
            for user in session_users:
                if len(page['sessions'][user]) == 0:
                    del page_copy['sessions'][user]
            db_books['%s_sessions_speed' % book_id].update({'_id': page['_id']},
                                                           {'$set': {'sessions': page_copy['sessions']}})
    return


def smooth_points(Y, N=10):
    new_Y = []
    for i in range(0, len(Y)):
        smooth_N = N
        if i - N < 0:
            smooth_N = i
            # new_Y.append(Y[i])
            # continue
        elif i + N >= len(Y):
            smooth_N = len(Y) - i - 1

        sum = 0
        for j in range(-smooth_N, smooth_N):
            sum += Y[i + j]
        sum /= ((2 * smooth_N) + 1)
        new_Y.append(sum)

    return new_Y


def average_speed_plot(book_id, smooth=False):
    db_books = connect_to_mongo_database(BOOKS_DB)
    pages = db_books['%s_sessions_speed' % book_id].find().sort('_id')

    plt.clf()
    x, avr_speed, dialogs_num, person_pronouns, sentiment_words, new_words_count = \
        list(), list(), list(), list(), list(), list()
    for page in pages:
        x.append(page['_id'])
        avr_speed.append(page['avr_speed'])
        dialogs_num.append(page['dialogs_num'] * 20)
        person_pronouns.append(page['person_pronouns_num'] * 5)
        sentiment_words.append(page['sum_sentiment'] * 5)
        new_words_count.append(page['new_words_count'])

    if smooth:
        avr_speed = smooth_points(avr_speed, N=5)
        dialogs_num = smooth_points(dialogs_num, N=5)
        person_pronouns = smooth_points(person_pronouns, N=5)
        sentiment_words = smooth_points(sentiment_words, N=5)
        new_words_count = smooth_points(new_words_count, N=5)

    plt.plot(x, avr_speed, label='avr_speed')
    plt.plot(x, dialogs_num, label='dialogs_num')
    plt.plot(x, person_pronouns, label='person_pronouns')
    plt.plot(x, sentiment_words, label='sentiment_words')
    plt.plot(x, new_words_count, label='new_words_count')

    plt.legend()

    plt.tight_layout()
    filename = '%s_avr_speed' % book_id
    if smooth:
        filename += '_smooth'
    plt.savefig(filename + '.png', bbox_inches='tight')


def absolute_speed_plot(book_id, smooth=False):
    db_books = connect_to_mongo_database(BOOKS_DB)
    pages = db_books['%s' % book_id].find().sort('_id')

    x, y = list(), list()

    plt.clf()
    for page in pages:
        x.append(page['_id'])
        y.append(page['absolute_speed'])

    y = smooth_points(y, N=5)
    plt.plot(x, y, label='absolute_speed')

    plt.legend()

    plt.tight_layout()
    filename = '%s_absolute_speed' % book_id
    if smooth:
        filename += '_smooth'
    plt.savefig(filename + '.png', bbox_inches='tight')


def full_book_process(book_id):
    # print ('Remove duplicate sessions')
    # remove_duplicate_sessions(book_id)

    print ('Define procents for items')  # TODO add automatic copying of items to seperate collection
    define_percents_for_items(book_id=book_id)

    print ('Process sessiond to book scale')
    process_sessions_to_book_percent_scale(book_id)

    # print ('Get target users')
    # target_users = get_target_users(book_id)
    #
    # print ('Begin to process users sessions...')
    #
    # processed_users = 0
    # for user_id in target_users:
    #     try:
    #         calculate_session_speed(book_id=book_id, user_id=user_id)
    #         process_sessions_to_pages(book_id, user_id)
    #     except Exception as e:
    #         print('Exception with user %s happened' % user_id)
    #         print (e)
    #     processed_users += 1
    #     print('Processed %d/%d users' % (processed_users, len(target_users)))
    # check_null_sessions(book_id)
    #
    # print ('Begin to count average/absolute speed for pages, preparing plots')
    # # Average speed frequency hist
    # average_user_speeds = count_average_page_speed(book_id)
    # plt.hist(average_user_speeds, bins=np.arange(0, 5000, 20))
    # plt.savefig('speed_frequency' + '.png', bbox_inches='tight')
    #
    # # Average speed plot TODO do smth with > 1000 speed
    # average_speed_plot(book_id, smooth=True)
    #
    # absolute_speed_plot(book_id, smooth=True)
    count_who_see_the_page_plot(book_id)


book_id = '11833'
full_book_process(book_id)
