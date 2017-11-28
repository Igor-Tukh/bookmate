from pymongo import MongoClient
import csv
import pandas as pd
import pickle
BOOKS_DB = 'bookmate'


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def collect_features_to_csv(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    cursor = db['%s_pages' % book_id].find({}, {'_id': 0, 'person_verbs_part': 1, 'person_pronouns_part': 1,
                                                'avr_word_len': 1, 'new_words_count': 1, 'sentiment': 1,
                                                'sentiment_words_portion': 1, 'person_verbs_num': 1,
                                                'person_pronouns_num': 1, 'words_num': 1, 'sentences_num': 1,
                                                'dialogs_num': 1})
    with open('resources/%s_features.csv' % book_id, 'w') as outfile:
        fields = ['person_verbs_part', 'person_pronouns_part',
                  'avr_word_len', 'new_words_count', 'sentiment',
                  'sentiment_words_portion', 'person_verbs_num', 'person_pronouns_num',
                  'words_num', 'sentences_num', 'dialogs_num']
        writer = csv.DictWriter(outfile, fieldnames=fields)
        # writer.writeheader()
        for x in cursor:
            writer.writerow(x)


def collect_alphabet():
    all_text = ''
    book_ids = ['210901', '2289']
    db = connect_to_mongo_database(BOOKS_DB)
    for book_id in book_ids:
        pages = db['%s_pages' % book_id].find()
        for page in pages:
            all_text += page['text']
    return list(set(list(all_text)))


def collect_one_hot_encoding_to_csv(book_id):
    alphabet = collect_alphabet()
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    pages = connect_to_mongo_database(BOOKS_DB)['%s_pages' % book_id].find()
    pages_vectors = list()
    for page in pages:
        integer_encoded = [char_to_int[char] for char in list(page['text'])]
        if len(integer_encoded) > 1000:
            integer_encoded = integer_encoded[0:1000]
        elif len(integer_encoded) < 1000:
            continue
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        pages_vectors.append(onehot_encoded)
    pickle.dump(pages_vectors, open('resources/%s_one_hot.pkl' % book_id, 'wb'))


def collect_groundtruth_to_csv(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    cursor = db['%s_pages' % book_id].find({}, {'_id': 0, 'page_speed': 1, 'page_skip_percent': 1,
                                                'page_return_percent': 1, 'page_unusual_percent': 1, 'text': 1})
    with open('resources/%s_groundtruth.csv' % book_id, 'w') as outfile:
        fields = ['page_speed', 'page_skip_percent', 'page_return_percent', 'page_unusual_percent']
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for x in cursor:
            if len(x['text']) >= 1000:
                del x['text']
                writer.writerow(x)


def main():
    book_id = '210901'
    # collect_features_to_csv(book_id)
    collect_groundtruth_to_csv(book_id)
    # collect_one_hot_encoding_to_csv(book_id)

if __name__ == "__main__":
    main()

