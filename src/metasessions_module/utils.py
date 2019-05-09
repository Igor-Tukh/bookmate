import pymongo
import datetime
import logging
import pickle


def connect_to_mongo_database(db):
    client = pymongo.MongoClient('localhost', 27017)
    db = client[db]
    return db


def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp // 1000)


def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as pickle_file:
        logging.info('Loading dumped value from {}'.format(pickle_path))
        return pickle.load(pickle_file)


def save_via_pickle(value, output_pickle_path):
    with open(output_pickle_path, 'wb') as output_pickle_path:
        logging.info('Saving value to {}'.format(output_pickle_path))
        pickle.dump(value, output_pickle_path)
