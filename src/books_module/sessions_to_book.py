from pymongo import MongoClient
import datetime
import pymongo
import bisect
import math

BOOKS_DB = 'bookmate'
USERS_DB = 'bookmate_users'
log_step = 100000


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp // 1000)


def process_sessions_fields_to_int(collection):
    print('Process sessions fields to int')
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
            print('Process %d/%d sessions' % (counter, sessions_num))

    return


def remove_duplicate_sessions(book_id):
    # Because logs have some duplicate sessions, we need to remove them
    print ('Begin to remove duplicates')
    db_sessions = connect_to_mongo_database(BOOKS_DB)
    book_sessions = db_sessions[book_id].find()
    dump_collection = db_sessions['%s_clear' % book_id]
    processed_sessions, removed_sessions = 0, 0
    try:
        db_sessions[book_id].create_index(
            [('_from', pymongo.ASCENDING), ('_to', pymongo.ASCENDING), ('item_id', pymongo.ASCENDING),
             ('user_id', pymongo.ASCENDING)])
    except Exception as e:
        print('Exception in index creating for duplicated sessions')
        print(e)

    for session in book_sessions:
        duplicate_sessions = db_sessions[book_id].find({'_from': session['_from'],
                                                        '_to': session['_to'],
                                                        'item_id': session['item_id'],
                                                        'user_id': session['user_id']})
        for duplicate_session in duplicate_sessions:
            if duplicate_session['_id'] != session['_id']:
                if abs(float((session['read_at'] - duplicate_session['read_at']).total_seconds())) <= 10:
                    db_sessions[book_id].remove({'_id': duplicate_session['_id']})
                    removed_sessions = removed_sessions + 1
                    if removed_sessions % log_step == 0:
                        print('Remove {%d} duplicates' % removed_sessions)
        processed_sessions = processed_sessions + 1
        if processed_sessions % log_step == 0:
            print('Process {%d} sessions' % processed_sessions)

    sessions = db_sessions[book_id].find()
    for session in sessions:
        dump_collection.insert(session)


def process_sessions_to_book_percent_scale(book_id, update_old=False):
    # process percents of sessions in item into percent of sessions in book
    db = connect_to_mongo_database(BOOKS_DB)

    if not update_old:
        sessions = db[book_id].find({'symbol_from': {'$not': {'$exists': True}}}, no_cursor_timeout=True)
    else:
        sessions = db[book_id].find(no_cursor_timeout=True)
    items = db['%s_items' % book_id]
    print('Found {%s} sessions' % sessions.count())
    num_sessions, deleted_sessions = 0, 0

    book_symbols_num = db['books'].find_one({'_id': book_id})['symbols_num']

    for session in sessions:
        session_item = items.find_one({'id': session['item_id']})
        if session_item is None:
            print('Find None item with id {%s}' % session['item_id'])
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
            # print(e)
            # print('skip and remove this session...')
            db[book_id].remove({'_id': session['_id']})
            deleted_sessions += 1
            if deleted_sessions % 1000 == 0:
                print ("{%d} sessions removed" % deleted_sessions)

        num_sessions = num_sessions + 1
        if num_sessions % log_step == 0:
            print('{%d} sessions processed' % num_sessions)

    book_sessions = db[book_id].find()
    db['%s_clear' % book_id].drop()

    for session in book_sessions:
        db['%s_clear' % book_id].insert(session)


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
            print('Processed {%d} sessions.' % processed)


def calculate_session_speed(book_id, user_id):
    # Calculation of speed for every micro-session
    # print ('Calculate session speed')
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find({'user_id': user_id}).sort('read_at')

    total_symbols = 0
    total_time = 0
    session_break = 5  # in minutes

    user_sessions_number = sessions.count()
    unknown_sessions_number = 0
    previous_session = -1
    for session in sessions:
        if previous_session == -1:
            previous_session = session
            db[book_id].update({'_id': previous_session['_id']},
                               {'$set': {'speed': -1}})
            continue

        stats = dict()
        stats['session_time'] = -1
        if int(session['size']) > 0:
            stats['symbols'] = session['symbol_to'] - session['symbol_from']
            stats['session_time'] = float((session['read_at'] - previous_session['read_at']).total_seconds() / 60)
            if stats['session_time'] > session_break:
                stats['speed'] = -1
                unknown_sessions_number += 1
            else:
                try:
                    stats['speed'] = stats['symbols'] / stats['session_time']
                    if stats['speed'] <= 1500:
                        # we don't need big speeds for average speed processing
                        total_symbols += stats['symbols']
                        total_time += stats['session_time']
                except ZeroDivisionError:
                    pass
        previous_session = session
        db[book_id].update({'_id': session['_id']},
                           {'$set': stats})

    if unknown_sessions_number / user_sessions_number * 100 >= 50:
        # print('User [%d] has more then 50 percents of uninterpretable sessions, deleted' % user_id)
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            db[book_id].remove({'_id': session['_id']})
        return
    if total_time == 0:
        # print('User [%d] has only sessions with speed >= 1500 symbols/min, deleted' % user_id)
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            db[book_id].remove({'_id': session['_id']})
        return

    try:
        avr_speed = total_symbols / total_time
    except:
        print('User [%d] has unexpected exception, deleted' % user_id)
        sessions = db[book_id].find({'user_id': user_id})
        for session in sessions:
            db[book_id].remove({'_id': session['_id']})
        return
    unknown_sessions = db[book_id].find({'speed': -1,
                                         'user_id': int(user_id)})
    for session in unknown_sessions:
        db[book_id].update({'_id': session['_id']},
                           {'$set': {'speed': avr_speed}})

    unknown_sessions = db[book_id].find({'session_time': -1,
                                         'user_id': int(user_id)})
    for session in unknown_sessions:
        db[book_id].update({'_id': session['_id']},
                           {'$set': {'speed': avr_speed}})


def define_borders_for_items(book_id):
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
    print('Define items percents/symbols')
    book = db_books['books'].find_one({'_id': book_id})
    symbols_num = book['symbols_num']
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
            symbol_from = math.ceil(_from * symbols_num)
            symbol_to = math.ceil(_to * symbols_num)
            db_books['%s_items' % book_id].update({'_id': item['_id']},
                                                  {'$set': {'_from': _from * 100.0,
                                                            '_to': _to * 100.0,
                                                            'symbol_from': symbol_from,
                                                            'symbol_to': symbol_to}})
    print('Book {%s} done' % book_id)
    print('Inserted %d items' % db_books['%s_items' % book_id].find().count())


def get_target_users(book_id, begin=10.0, end=80.0):
    # return list of users, who begin to read the book before begin(10.0) and end after end(80.0)
    print('Get target users...')
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
        center_sessions_count = db_sessions[book_id].find({'user_id': user_id,
                                                           'book_to': {'$gte': 10.0}}).count()

        user['_id'] = user_id
        user['end_sessions'] = end_sessions_count
        user['begin_sessions'] = begin_sessions_count
        user['center_sessions'] = center_sessions_count

        users_collection.insert(user)
        if end_sessions_count > 0 and begin_sessions_count > 0:
            target_users.append(user_id)
            # print('Found %d target users' % len(target_users))

        seen_users += 1
        if seen_users % 500 == 0:
            print('Process %d/%d users' % (seen_users, len(users_id)))

    print('Return {%d} target users' % len(target_users))
    return target_users


def get_book_users(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    users = db[book_id].find().distinct('user_id')
    return users


def get_book_items(book_id):
    db = connect_to_mongo_database('readings')
    items = db[book_id].find().distinct('item_id')
    return items


def filter_book_core_users(book_id):
    # delete those users, who begin to read the book from more then 10% of the size
    print('Begin to remove users with begin position > 10% of the book')
    db = connect_to_mongo_database(BOOKS_DB)
    db[book_id].create_index([('user_id', pymongo.ASCENDING)])

    users = db[book_id].find().distinct('user_id')
    print('Users in the {%s book} collection: {%d}' % (book_id, len(users)))
    remove_counter, process_counter = 0, 0
    for user_id in users:
        sessions = db[book_id].find({'user_id': user_id})
        min_position = 100.0
        for session in sessions:
            if 'book_from' not in session:
                db[book_id].remove({'_id': session['_id']})
            elif float(session['book_from']) < min_position:
                min_position = session['book_from']
        if min_position > 10.0:
            sessions = db[book_id].find({'user_id': user_id})
            for session in sessions:
                db[book_id].remove({'_id': session['_id']})
            remove_counter += 1
            if remove_counter % 500 == 0:
                print('%d/%d users removed' % (remove_counter, len(users)))
        process_counter += 1
        if process_counter % 1000 == 0:
            print('%d/%d users processed' % (process_counter, len(users)))

    users = get_book_users(book_id)
    print('Users in the {%s book} collection: {%d}' % (book_id, len(users)))


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


def smooth_points(Y, N=10):
    new_Y = []
    for i in range(0, len(Y)):
        smooth_N = N
        if i - N < 0:
            smooth_N = i
            new_Y.append(Y[i])
            continue
        elif i + N >= len(Y):
            smooth_N = len(Y) - i - 1
            new_Y.append(Y[i])
            continue

        sum = 0
        for j in range(-smooth_N, smooth_N):
            sum += Y[i + j]
        sum /= ((2 * smooth_N) + 1)
        new_Y.append(sum)

    return new_Y


def get_all_items_borders(book_id):
    #  Process each item for getting all possible borders inside them
    print('Begin to get all items borders')
    db = connect_to_mongo_database(BOOKS_DB)
    db[book_id].create_index([('item_id', pymongo.ASCENDING)])
    db['%s_items' % book_id].create_index([('id', pymongo.ASCENDING)])
    items = db[book_id].find().distinct('item_id')

    counter = 1
    for item_id in items:
        if counter % 1000 == 0:
            print('Processing %d/%d items' % (counter, len(items)))
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


def set_sessions_borders(book_id, session_collection, drop_old=False, define_sessions_borders=False):
    print('Begin to set sessions borders')
    db = connect_to_mongo_database(BOOKS_DB)
    db['%s_borders' % book_id].create_index([('symbol_from', pymongo.ASCENDING),
                                             ('symbol_to', pymongo.ASCENDING)]
                                            )

    all_borders = list()
    if 'all_borders' in db['books'].find_one({'_id': book_id}):
        all_borders = db['books'].find_one({'_id': book_id})['all_borders']
    else:
        items = db['%s_items' % book_id].find()
        for item in items:
            if 'item_borders' not in item:
                continue
            all_borders.extend(item['item_borders'])
        all_borders = list(set(all_borders))
        all_borders.sort()
        db['books'].update({'_id': book_id},
                           {'$set':
                                {'all_borders': all_borders}
                            })

    all_borders.sort()
    if drop_old:
        db['%s_borders' % book_id].drop()
        for i in range(0, len(all_borders) - 1):
            db['%s_borders' % book_id].insert_one({'_id': i,
                                                   'symbol_from': all_borders[i],
                                                   'symbol_to': all_borders[i + 1]})

    if define_sessions_borders:
        print('begin to define sessions borders')
        sessions = db[session_collection].find()
        counter = 0
        sessions_num = sessions.count()
        for session in sessions:
            begin_border = bisect.bisect_left(all_borders, session['symbol_from'])
            end_border = bisect.bisect_left(all_borders, session['symbol_to']) - 1
            db[session_collection].update({'_id': session['_id']},
                                      {'$set': {
                                          'begin_border': begin_border,
                                          'end_border': end_border
                                      }})
            counter += 1
            if counter % log_step == 0:
                print('Process %d/%d sessions' % (counter, sessions_num))
        db['%s_clear' % book_id].drop()
        sessions = db[session_collection].find()
        for session in sessions:
            db['%s_clear' % book_id].insert(session)

    return all_borders


def define_target_sessions(book_id, target_users):
    print('Select target users sessions...')
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[book_id].find()

    db['%s_target' % book_id].drop()
    for session in sessions:
        if 'user_id' in session and session['user_id'] in target_users:
            db['%s_target' % book_id].insert(session)

    print('Found %d target users sessions' % db['%s_target' % book_id].find().count())


def get_absolute_speeds_for_borders(target_users, sessions_collection, borders_num):
    print('Begin to calculate absolute speed for sessions')
    db = connect_to_mongo_database(BOOKS_DB)
    user_count = 0

    borders_abs_speeds = [0 for i in range(borders_num)]
    borders_sessions_num = [0 for i in range(borders_num)]
    db[sessions_collection].create_index([('user_id', pymongo.ASCENDING), ('category', pymongo.ASCENDING)])

    for user_id in target_users:
        sessions = db[sessions_collection].find({'user_id': int(user_id),
                                                 'category': 'normal'})
        symbols = 0
        total_time = 0
        for session in sessions:
            symbols += session['size']
            total_time += session['size'] / session['speed']

        if symbols > 0:
            avr_book_speed = symbols / total_time
            sessions = db[sessions_collection].find({'user_id': int(user_id),
                                                     'category': 'normal'})
            for session in sessions:
                abs_speed = session['speed'] / avr_book_speed
                db[sessions_collection].update({'_id': session['_id']},
                                               {'$set': {'abs_speed': abs_speed}})

                for border_id in range(session['begin_border'], session['end_border'] + 1):
                    borders_abs_speeds[border_id] += abs_speed
                    borders_sessions_num[border_id] += 1

        user_count += 1
        if user_count % 500 == 0:
            print('%d/%d users processed' % (user_count, len(target_users)))

    print('Begin to update borders with absolute speed')
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


def get_unususual_sessions_for_borders(book_id, sessions_collection, borders_num):
    borders = [0 for i in range(borders_num)]
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[sessions_collection].find()

    for session in sessions:
        if session['type'] == 'unusual':
            for border_id in range(session['begin_border'], session['end_border'] + 1):
                borders[border_id] += 1

    for border_id in range(0, len(borders)):
        db['%s_borders' % book_id].update_one({'_id': border_id},
                                              {'$set': {
                                                  'unusual_sessions': borders[border_id]
                                              }})


def define_normal_speed(book_id, sessions_collection, skip_percent=0.6):
    # define normal/skip speed for book
    # delete sessions without speed field
    print('Begin to define book speeds (normal/skip)')
    db = connect_to_mongo_database(BOOKS_DB)
    sessions = db[sessions_collection].find()

    total_symbols, total_time, sessions_number = 0, 0, 0
    for session in sessions:
        if 'speed' in session:
            if 5000 >= session['speed'] > 0:
                total_symbols += session['size']
                total_time += session['size'] / session['speed']
                sessions_number += 1
        else:
            db[sessions_collection].remove({'_id': session['_id']})

        if sessions_number % log_step == 0:
            print('%d sessions processed' % sessions_number)

    avr_book_speed = total_symbols / total_time
    skip_speed = avr_book_speed + avr_book_speed * skip_percent

    db['books'].update({'_id': str(book_id)},
                       {'$set': {
                           'normal_speed': avr_book_speed,
                           'skip_speed': skip_speed
                       }})

    print ('Begin to set categories for sessions')
    sessions = db[sessions_collection].find()
    counter = 0
    for session in sessions:
        if 'speed' not in session:
            db[sessions_collection].remove({'_id': session['_id']})
        elif session['speed'] >= skip_speed:
            category = 'skip'
        else:
            category = 'normal'
        db[sessions_collection].update({'_id': session['_id']},
                                       {'$set': {
                                           'category': category
                                       }})
        counter += 1
        if counter % log_step == 0:
            print('Process %d/%d sessions' % (counter, sessions_number))


def aggregate_borders(book_id, symbols_num=1000):
    # Aggregate borders to the size of 1000 symbols. Update those borders, where every ~1000 symbols achieved
    print('Begin to aggregate borders')
    db = connect_to_mongo_database(BOOKS_DB)
    borders = db['%s_borders' % book_id].find().sort('_id')

    # get strict sections borders
    sections = db['%s_sections' % book_id].find().sort('_id')
    sections_borders = list()
    for section in sections:
        sections_borders.append(section['symbol_to'])

    page_symbols = 0
    begin_page_id, end_page_id = 0, 0
    page = 1
    break_flag = False
    for border in borders:
        if border['abs_speed'] == 0:
            print('problem with border (id == %d), absolute speed is zero, skip' % border['_id'])
            continue
        for section_border in sections_borders:
            if border['symbol_from'] <= section_border <= border['symbol_to']:
                break_flag = True

        if not break_flag:
            page_symbols += border['symbol_to'] - border['symbol_from']
        if page_symbols >= symbols_num:
            break_flag = True

        if break_flag:
            end_page_id = border['_id']

            page_borders = db['%s_borders' % book_id].find({ '_id': {'$gte': begin_page_id },
                                                             '$and': [{'_id': {'$lte': end_page_id }}]})
            page_speed = 0
            page_unusual_sessions = 0
            page_sessions = 0
            page_skip_sessions = 0
            for page_border in page_borders:
                page_speed += ((page_border['symbol_to'] - page_border['symbol_from']) / page_symbols) * page_border['abs_speed']
                page_unusual_sessions += ((page_border['symbol_to'] - page_border['symbol_from']) / page_symbols) * page_border['unusual_sessions']
                page_skip_sessions += ((page_border['symbol_to'] - page_border['symbol_from']) / page_symbols) * page_border['skip_sessions']
                page_sessions += ((page_border['symbol_to'] - page_border['symbol_from']) / page_symbols) * page_border['sessions']

            db['%s_borders' % book_id].update({'_id': border['_id']},
                                              {'$set': {'page_speed': page_speed,
                                                        'page_unusual_sessions': page_unusual_sessions,
                                                        'page_sessions': page_sessions,
                                                        'page_skip_sessions': page_skip_sessions,
                                                        'page_unusual_percent': page_unusual_sessions / page_sessions,
                                                        'page_skip_percent': page_skip_sessions / page_sessions,
                                                        'page': page}})
            page_symbols = 0
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

    print('Found %d users' % len(users))


def count_sessions_category_per_border(book_id, sessions_collection):
    db = connect_to_mongo_database(BOOKS_DB)

    borders = db['%s_borders' % book_id].find()

    db['%s_target' % book_id].create_index([('symbol_from', pymongo.ASCENDING),
                                            ('symbol_to', pymongo.ASCENDING),
                                            ('category', pymongo.ASCENDING)])
    db['%s_target' % book_id].create_index([('symbol_from', pymongo.ASCENDING),
                                            ('symbol_to', pymongo.ASCENDING)])

    sessions = db[sessions_collection].find()
    sessions_per_border = [0 for i in range(borders.count())]
    skips_per_border = [0 for i in range(borders.count())]

    for session in sessions:
        if session['category'] == 'skip':
            for border_id in range(session['begin_border'], session['end_border'] + 1):
                skips_per_border[border_id] += 1
        for border_id in range(session['begin_border'], session['end_border'] + 1):
            sessions_per_border[border_id] += 1

    for border_id in range(0, len(sessions_per_border)):
            db['%s_borders' % book_id].update({'_id': border_id},
                                    {'$set': {'sessions': sessions_per_border[border_id],
                                              'skip_sessions': skips_per_border[border_id]}})


def count_unusual_sessions(sessions_collection):
    print ('Begin to find unusual sessions')
    db = connect_to_mongo_database(BOOKS_DB)
    users_db = connect_to_mongo_database(USERS_DB)

    users = db[sessions_collection].find().distinct('user_id')
    db[sessions_collection].create_index([('user_id', pymongo.ASCENDING)])
    processed_users, all_users = 0, len(users)

    for user_id in users:
        sessions = users_db[str(user_id)].find()
        sessions_count = sessions.count()
        daytimes = list()
        for i in range(0, 24):
            daytimes.append(0)

        for session in sessions:
            daytimes[session['read_at'].hour] += 1

        sorted_daytime_indexes = sorted(range(len(daytimes)), key=lambda k: daytimes[k], reverse=True)[:16]
        daytimes = sorted(daytimes, reverse=True)[:16]
        usual_sessions_times = []
        counter = 0
        for (hour, session_num) in zip(sorted_daytime_indexes, daytimes):
            # if counter + session_num <= int(0.9 * sessions_count):
            usual_sessions_times.append(hour)
                # counter += session_num
            # else:
            #     break

        sessions = db[sessions_collection].find({'user_id': user_id})
        for session in sessions:
            if session['read_at'].hour in usual_sessions_times:
                db[sessions_collection].update({'_id': session['_id']},
                                   {'$set': {
                                       'type': 'usual'
                                   }})
            else:
                db[sessions_collection].update({'_id': session['_id']},
                                   {'$set': {
                                       'type': 'unusual'
                                   }})
        processed_users += 1
        if processed_users % 100 == 0:
            print ('Process %d/%d users' % (processed_users, all_users))


def select_top_document_ids(book_id, top_n = 3):
    """Select top N most popular documents and delete other sessions"""
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
    for session in sessions:
        if session['document_id'] not in popular_ids:
            db[book_id].remove({'_id': session['_id']})


def full_book_process(book_id):
    # print('Book [%s] process begin' % str(book_id))
    # remove_duplicate_sessions(book_id)
    # select_top_document_ids(book_id, 3)
    # define_borders_for_items(book_id=book_id)
    # process_sessions_to_book_percent_scale(book_id, update_old=True)
    #
    # filter_book_core_users(book_id)
    #
    # get_all_items_borders(book_id)

    all_borders = set_sessions_borders(book_id, session_collection=book_id,
                                       drop_old=False, define_sessions_borders=False)

    target_users = get_target_users(book_id)
    processed_users = 0

    # print('Begin to calculate users sessions speed')
    # for user_id in target_users:
    #     calculate_session_speed(book_id=book_id, user_id=user_id)
    #     processed_users += 1
    #     if processed_users % 500 == 0:
    #         print('Processed %d/%d users' % (processed_users, len(target_users)))
    #
    # define_target_sessions(book_id, target_users)
    target_sessions_collection = str(book_id) + '_target'
    # define_normal_speed(book_id, target_sessions_collection)
    count_unusual_sessions(target_sessions_collection)
    count_sessions_category_per_border(book_id, target_sessions_collection)
    # get_absolute_speeds_for_borders(target_users, target_sessions_collection, len(all_borders))
    get_unususual_sessions_for_borders(book_id, target_sessions_collection, len(all_borders))


book_id = '2289'
# full_book_process(book_id)
aggregate_borders(book_id, symbols_num=1000)
