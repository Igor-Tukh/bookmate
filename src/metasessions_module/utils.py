import os

import pymongo
import datetime
import logging
import pickle
import csv
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


def save_result_to_csv(results, output_file):
    if len(results) == 0:
        logging.info('Unable to save results to csv: empty list of results provided')
        return
    with open(output_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def min_max_scale(x):
    min_x = np.min(x)
    max_x = np.max(x)
    if np.isclose(min_x, max_x):
        logging.info('Can\'t apply min max scaling to x: x is a constant')
    return (x - min_x) / (max_x - min_x)
