import pickle
# import cPickle
import sqlite3
import datetime
from pymongo import MongoClient
import meta_indexes
from record import Record

DB_PATH = '/Volumes/KseniyaB/bookmate.db'


def connect_to_mongo_database(book_id):
    client = MongoClient('localhost', 27017)
    db = client[str(book_id)]
    return db


def __push_keys(dictionary, keys):
    for key in keys:
        if key not in dictionary:
            dictionary[key] = {}
        dictionary = dictionary[key]
    return dictionary


def add_number(dictionary, keys, number):
    dictionary = __push_keys(dictionary, keys[:-1])
    if keys[-1] not in dictionary:
        dictionary[keys[-1]] = 0.0
    dictionary[keys[-1]] += number


def set_object(dictionary, keys, obj):
    dictionary = __push_keys(dictionary, keys[:-1])
    dictionary[keys[-1]] = obj


def get_object(dictionary, keys):
    obj = dictionary
    for key in keys:
        if key not in obj:
            return None
        obj = obj[key]
    return obj


def add_object(dictionary, keys, obj):
    dictionary = __push_keys(dictionary, keys[:-1])
    if keys[-1] not in dictionary:
        dictionary[keys[-1]] = set()
    dictionary[keys[-1]].add(obj)


def add_objects(dictionary, keys, objects):
    dictionary = __push_keys(dictionary, keys[:-1])
    if keys[-1] not in dictionary:
        dictionary[keys[-1]] = set()
    dictionary[keys[-1]] |= objects


def serialize_object(obj, name):
    folder = 'dumps'
    with open(folder + '/' + name + '.pk', 'wb') as f:
        pickler = pickle.Pickler(f)
        pickler.fast = True
        pickler.dump(obj)
    print('{0} is serialized'.format(name))


def collect_data(collect_stats, sql, save_collection_name=None, days_to_pass=None):
    lcid_bid = meta_indexes.load_lcid_bid()
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(sql)
    print('Query executed successfully')
    index = 0
    log_step = 1000
    start = datetime.datetime.now()
    # if save_collection_name != None:
    #     db = connect_to_mongo_database()
    # record = write_record_to_mongodb(db, save_collection_name, cursor.fetchone())
    record = Record(cursor.fetchone(), lcid_bid)

    # record = Record(cursor.fetchone(), lcid_bid)
    # first_log_moment = record.read_at

    while not record.empty:
        collect_stats(record)
        index += 1
        if index%log_step == 0:
            print('records processed: {0}'.format(index))
        record = Record(cursor.fetchone(), lcid_bid)
        # record = write_record_to_mongodb(db, save_collection_name, cursor.fetchone())
    finish = datetime.datetime.now()
    delta = finish - start
    print('Time passed: {0}'.format(delta))
    connection.close()
    return index


def calc_passed_time(datetime1, datetime2):
    if datetime1 is None or datetime2 is None:
        return None
    seconds = (datetime2 - datetime1).total_seconds()
    return seconds/60.0


def calc_read_symbols(book_id, percents, books_index):
    if book_id in books_index and \
            'letter_count' in books_index[book_id] and \
            books_index[book_id]['letter_count'].isdigit():
        return percents/100*float(books_index[book_id]['letter_count'])
    else:
        return None
