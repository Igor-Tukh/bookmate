import os

import pymongo
import datetime
import logging
import pickle
import numpy as np


def connect_to_mongo_database(db):
    client = pymongo.MongoClient('localhost', 27017)
    db = client[db]
    return db


def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp // 1000)


def date_to_percent_of_day(date):
    return (date.hour * 3600 + date.minute * 60 + date.second) / (24 * 60 * 60)


def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as pickle_file:
        logging.info('Loading dumped value from {}'.format(pickle_path))
        return pickle.load(pickle_file)


def save_via_pickle(value, output_pickle_path):
    with open(output_pickle_path, 'wb') as output_pickle_path:
        logging.info('Saving value to {}'.format(output_pickle_path))
        pickle.dump(value, output_pickle_path)
        return value


def get_batch_by_percent(batches_number, percent):
    batch_percent = 100.0 / batches_number
    return min(round(percent / batch_percent), batches_number - 1)


def is_int(value):
    try:
        int(value)
        return True
    except:
        return False


def array_is_trivial(a):
    return len(np.unique(a)) <= 1


def get_stats_path(stats_filename):
    return os.path.join('resources', 'stats', stats_filename)
