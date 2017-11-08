import pickle
import datetime
import math
import meta_indexes
import mining_help
from record import Record
import calculate_stats


books_users_dict = dict()
# the structure will be the following:
# [book_id] -> {[user_id] -> (min_begin_percent, max_end_percent, sum_percent)]}
users_books_dict = dict()
# the structure will be the following:
# [user_id] -> {book_id1, book_id2, ..., book_idN}


def write_record_to_mongodb(record, db, collection):
    if record.empty:
        return

    db = mining_help.connect_to_mongo_database(db)
    find_session = dict()
    find_session['_id'] = record.read_at
    if db[collection].find(find_session).count() != 0:
        return

    record_tosave = dict()
    record_tosave['empty'] = record.empty
    record_tosave['library_card_id'] = record.library_card_id
    record_tosave['read_at'] = record.read_at
    record_tosave['is_uploaded'] = record.is_uploaded
    record_tosave['app_user_agent'] = record.app_user_agent
    record_tosave['user_id'] = record.user_id
    record_tosave['ip'] = record.ip
    record_tosave['created_at'] = record.created_at
    record_tosave['phantom_id'] = record.phantom_id
    record_tosave['updated_at'] = record.updated_at
    record_tosave['author_page'] = record.author_page
    record_tosave['country_code3'] = record.country_code3
    record_tosave['to_percent'] = record.to_percent
    record_tosave['user_agent'] = record.user_agent
    record_tosave['book_id'] = record.book_id

    record_tosave['item_id'] = record.item_id
    record_tosave['from_percent'] = record.from_percent
    record_tosave['record_id'] = record.record_id
    record_tosave['is_phantom'] = record.is_phantom
    record_tosave['city'] = record.city
    record_tosave['document_id'] = record.document_id
    record_tosave['size'] = record.size
    record_tosave['_id'] = record.read_at

    db[collection].insert(record_tosave)


def update_books_and_users_stats(record):
    global books_users_dict
    global users_books_dict

    # get book stats for reading record
    if record.book_id not in books_users_dict:
            books_users_dict[record.book_id] = list()
    else:
        if record.user_id not in books_users_dict[record.book_id]:
            books_users_dict[record.book_id].append(record.user_id)

    # get user stats for selecting records
    if record.user_id in users_books_dict:
            if record.book_id not in users_books_dict[record.user_id]:
                users_books_dict[record.user_id].append(record.book_id)
    else:
        users_books_dict[record.user_id] = list()
        users_books_dict[record.user_id].append(record.book_id)


def collect_book_sessions(book_id):
    table_name = 'book_%s' % str(book_id)
    query = '''select * from''' % (str(table_name))
    print ('Get book %s,' % (str(book_id)))
    mining_help.collect_data(collect_stats, sql=query, save_collection_name=book_id)


def collect_core_users_books():
    offset = 0
    limit = 1000000
    query = '''select * from readings limit %s offset %s''' % (str(limit), str(offset))
    while (mining_help.collect_data(collect_stats, sql = query) != 0):
        offset += limit
        query = '''select * from readings limit %s offset %s''' % (str(limit), str(offset))
        print ('%s records processed' % offset)
        #get_core_books_users(10, 100)

        global books_users_dict
        global users_books_dict
        mining_help.serialize_object(books_users_dict, 'book-users-dict')
        mining_help.serialize_object(users_books_dict, 'user_books-dict')


def collect_stats(record):
    if record.user_id is None:
        return
    # update_books_and_users_stats(record)
    write_record_to_mongodb(record, str(record.book_id), str(record.user_id))


collect_book_sessions(210901)

