from pymongo import MongoClient

BOOKS_DB = 'bookmate'


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


