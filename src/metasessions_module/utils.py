import pymongo


def connect_to_mongo_database(db):
    client = pymongo.MongoClient('localhost', 27017)
    db = client[db]
    return db