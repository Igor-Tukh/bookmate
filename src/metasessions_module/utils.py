import pymongo
import datetime


def connect_to_mongo_database(db):
    client = pymongo.MongoClient('localhost', 27017)
    db = client[db]
    return db


def date_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp // 1000)
