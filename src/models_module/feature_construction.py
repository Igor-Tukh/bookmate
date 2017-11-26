from pymongo import MongoClient
import csv
BOOKS_DB = 'bookmate'


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def collect_features_to_csv(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    cursor = db['resources/%s_pages' % book_id].find({}, {'_id': 1, 'person_verbs_part': 1, 'person_pronouns_part': 1,
                                                'avr_word_len': 1, 'new_words_count': 1, 'sentiment': 1,
                                                'sentiment_words_portion': 1})
    with open('%s_features.csv' % book_id, 'w') as outfile:
        fields = ['_id', 'person_verbs_part', 'person_pronouns_part',
                  'avr_word_len', 'new_words_count', 'sentiment',
                  'sentiment_words_portion']
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for x in cursor:
            writer.writerow(x)


def collect_groundtruth_to_csv(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    cursor = db['%s_pages'].find({}, {'_id': 1, 'page_speed': 1, 'page_skip_percent': 1,
                                      'page_return_percent': 1, 'page_unusual_percent': 1})
    with open('resources/%s_groundtruth.csv' % book_id, 'w') as outfile:
        fields = ['_id', 'page_speed', 'page_skip_percent', 'page_return_percent', 'page_unusual_percent']
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for x in cursor:
            writer.writerow(x)


def main():
    book_id = '210901'
    collect_features_to_csv(book_id)
    collect_groundtruth_to_csv(book_id)


if __name__ == "__main__":
    main()

