from xml.etree import cElementTree as ET
from pymongo import MongoClient
import argparse
import re
import nltk
import string
import os
import pymorphy2
import gensim
import timeit
from stop_words import get_stop_words
from tqdm import tqdm
import traceback
from bson.int64 import Int64
import json
import numpy as np
from pymystem3 import Mystem


# GLOBAL VARIABLES SECTION
db = None
punctuation = string.punctuation
morph = pymorphy2.MorphAnalyzer()
person_pronouns_list = ['я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они']
word2vec_model = gensim.models.Word2Vec.load_word2vec_format('../../ruscorpora_russe.model.bin', binary = True,
                                                    unicode_errors='ignore')

ru_stopwords = get_stop_words('russian')
mystem = Mystem()
# GLOBAL VARIABLES SECTION END


def config(item):
    # Some magic with xml tags
    return re.sub('{[^>]+}', '', item)


def connect_to_database_books_collection():
    client = MongoClient('localhost', 27017)
    global db
    db = client.bookmate


def get_athor_information(description):
    first_name = 'None'
    second_name = 'None'
    middle_name = 'None'
    for node in description:
        if re.sub('{[^>]+}', '', node.tag) == "title-info":
            for info in node:
                if re.sub('{[^>]+}', '', info.tag) == "author":
                    for name in info:
                        if re.sub('{[^>]+}', '', name.tag) == "first-name":
                            first_name = name.text
                        elif re.sub('{[^>]+}', '', name.tag) == "last-name":
                            second_name = name.text
                        elif re.sub('{[^>]+}', '', name.tag) == "middle-name":
                            middle_name = name.text
    author = dict()
    author['first_name'] = first_name
    author['second_name'] = second_name
    author['middle_name'] = middle_name
    return author


def number_of_words(text):
    '''
    :param text: Text from one paragraph
    :return: Number of words in text exclude the punctuation
    '''
    try:
        text = nltk.word_tokenize(text)
        text = [word for word in text if word not in string.punctuation]
        return len(text)
    except:
        return 0


def number_of_sentences(text):
    try:
        sentences = nltk.sent_tokenize(text)
        return len(sentences)
    except:
        return 0


def add_book_to_main_table(book_id, book_stats):
    '''
    :param book: xml structure of the book
    :return: creates book's tables in database: paragraph table, window table;
    insert book info into main db table "books", return True in case of no exceptions and errors
    '''
    global db
    item = dict()
    book_table = db[book_id]
    # Book whole stats
    item['num_of_words'] = book_stats['num_of_words']
    item['num_of_dialogues'] = book_stats['num_of_dialogues']
    item['num_of_sentences'] = book_stats['num_of_sentences']
    item['num_of_symbols'] = book_stats['num_of_symbols']
    item['num_of_paragraphs'] = book_table.find().count()

    # Average book stats
    item['avr_word_len'] = book_stats['num_of_symbols'] / book_stats['num_of_words']
    item['avr_sentence_len'] = book_stats['num_of_words'] / book_stats['num_of_sentences']
    item['avr_dialogues'] = book_stats['num_of_dialogues'] / item['num_of_paragraphs']

    item['_id'] = book_id

    db.books.insert_one(item)
    print ('Book %s added to database' % (book_id))
    return


def process_book_simple_features_by_paragraph(book, _id):
    book_table = db[str(_id)]
    book_stats = dict()
    book_stats['num_of_words'] = 0
    book_stats['num_of_dialogues'] = 0
    book_stats['num_of_sentences'] = 0
    book_stats['num_of_symbols'] = 0
    global db
    tree = ET.ElementTree(book)
    root = tree.getroot()
    id = 0
    position = Int64(0)
    print ('\nProcessing book paragraphs')
    for item in root.iter():
        if config(item.tag) == 'p':
            stats = count_simple_text_features(item.text)
            if stats is None:
                continue
            stats.update(count_morphological_stats(item.text))
            stats['_id'] = id
            stats['begin_position'] = position
            stats['_id'] = id
            stats['end_position'] = position + stats['num_of_symbols']

            try:
                book_table.save(stats)
            except:
                traceback.print_exc()
                continue

            id += 1
            position = stats['end_position'] + 1

            if stats['is_dialogue']:
                book_stats['num_of_dialogues'] += 1
            book_stats['num_of_words'] += stats['num_of_words']
            book_stats['num_of_sentences'] += stats['num_of_sentences']
            book_stats['num_of_symbols'] += stats['num_of_symbols']

    return book_stats


def build_pages_table(book_id, window_size = 2000):
    # divide book into pages of ~1000 words till the end of the sentence
    collection_name = book_id + '_pages'
    book_table = db[book_id]
    print ('Building windows for book %s' % (book_id))
    for _id in tqdm(range(0, book_table.find().count())):
        window = count_window_features(book_table, _id, window_size)
        db[collection_name].insert_one(window)
    return


def count_simple_text_features(text):
    if text is None:
        return None
    if len(text) == 0:
        return None

    stats = dict()
    # Simple stats
    stats['text'] = text
    if text[0] == '—' or text[0] == '–':
        stats['is_dialogue'] = True
    else:
        stats['is_dialogue'] = False
    stats['num_of_words'] = number_of_words(text)
    stats['num_of_sentences'] = number_of_sentences(text)
    stats['num_of_symbols'] = len(text)

    if stats['num_of_words'] == 0:
        return None
    stats['avr_word_length'] = stats['num_of_symbols'] / stats['num_of_words']
    stats['avr_sentence_length']= stats['num_of_symbols'] / stats['num_of_sentences']

    return stats


def count_window_features(book_table, beginId, window_size):
    window = dict()
    text = ''
    paragraphs = 0
    dialogues = 0
    sentences = 0
    words = 0
    symbols = 0
    window['_id'] = Int64(beginId)
    person_verbs = 0
    person_pronouns = 0

    while symbols < window_size:
        paragraph = book_table.find_one({"_id": beginId})
        if paragraph is None:
            break
        paragraphs += 1
        if paragraph['is_dialogue']:
            dialogues += 1
        sentences += paragraph['num_of_sentences']
        words += paragraph['num_of_words']
        symbols += paragraph['num_of_symbols']
        beginId += 1
        text += paragraph['text']
        text += ' '
        person_verbs += paragraph['person_verbs_num']
        person_pronouns += paragraph['person_pronouns_num']

    # Basic features
    if words != 0:
        window['text'] = text
        window['dialoques_number'] = dialogues
        window['sentences_number'] = sentences
        window['num_of_words'] = words
        window['symbols_number'] = symbols
        window['end_id'] = beginId - 1
        window['person_verbs_num'] = person_verbs
        window['person_pronouns_num'] = person_pronouns

        # Average features per window
        window['avr_word_length'] = symbols / words
        window['avr_sentence_length'] = words / sentences
        window['avr_dialogues_part'] = dialogues / paragraphs
        window['avr_person_verbs_part'] = person_verbs / words
        window['avr_person_pronouns_part'] = person_pronouns / words

    return window


def count_morphological_stats(text):
    words = nltk.word_tokenize(text)
    morphological_stats = dict()
    morphological_stats['person_verbs_num'] = 0
    morphological_stats['person_pronouns_num'] = 0
    for word in words:
        if word in punctuation:
            continue
        token = morph.parse(word)

        # Morphological stats

        for p in token:
            if p.tag.POS == 'NPRO':
                morphological_stats['person_verbs_num'] += 1
            if p.normal_form in person_pronouns_list:
                morphological_stats['person_pronouns_num'] += 1
            # use break here to process only most possible word form
            break

    morphological_stats['person_verbs_portion'] = morphological_stats['person_verbs_num'] / len(words)
    morphological_stats['person_pronouns_portion'] = morphological_stats['person_pronouns_num'] / len(words)
    return morphological_stats


def count_new_vocabulary(book_id):
    print("Counting new vocabluary")
    new_vocabulary = dict()
    items = db[book_id].find()
    for item in items:
        new_words_count = 0
        words = nltk.word_tokenize(item['text'])
        for word in words:
            if word in punctuation:
                continue
            token = morph.parse(word)
            try:
                new_vocabulary[token[0].normal_form] += 1
            except:
                new_vocabulary[token[0].normal_form] = 1
                new_words_count += 1
        db[book_id].update({'_id': item['_id']},
                           {'$set' : {'new_words_count' : new_words_count}})
    return


def count_new_vocabulary_for_windows(book_id):
    windows = db[book_id + '_pages'].find()
    for window in windows:
        window_vocabulary = 0
        for i in range(window['_id'], window['end_id'] + 1):
            window_vocabulary += db[book_id].find_one({'_id': i})['new_words_count']
        db[book_id + '_pages'].update({'_id': window['_id']},
                           {'$set': {'new_words_count': window_vocabulary}})


def count_sentiment(book_id):
    print("Begin sentiment processing")
    items = db[book_id].find()
    with open('../resources/sentiment_dictionary.json', 'r') as f:
        sentiment_dict = json.load(f)

    for item in items:
        avr_sentiment = 0
        sentiment_words_proportion = 0
        try:
            words = nltk.word_tokenize(item['text'])
        except:
            continue
        for word in words:
            tokens = morph.parse(word)
            for token in tokens:
                try:
                    avr_sentiment += float(sentiment_dict[token.normal_form])
                    sentiment_words_proportion += 1
                    break
                except:
                    continue
        sum_sentiment = avr_sentiment
        avr_sentiment /= len(words)
        sentiment_words_proportion /= len(words)
        db[book_id].update({'_id': item['_id']},
                           {'$set': {'avr_sentiment': avr_sentiment,
                                     'sum_sentiment': sum_sentiment,
                                     'sentiment_words_portion': sentiment_words_proportion}})


def count_sentiment_for_windows(book_id):
    windows = db[book_id + '_pages'].find()
    for window in windows:
        window_sentiment = 0
        sentiment_words_portion = 0
        for i in range(window['_id'], window['end_id'] + 1):
            window_sentiment += db[book_id].find_one({'_id': i})['sum_sentiment']
            sentiment_words_portion += db[book_id].find_one({'_id': i})['sentiment_words_portion']
        sentiment_words_portion /= (window['end_id'] - window['_id'] + 1)
        db[book_id + '_pages'].update({'_id': window['_id']},
                           {'$set': {'sum_sentiment': window_sentiment,
                                     'sentiment_words_portion': sentiment_words_portion}})


def count_labels_portion(book_id):
    # count part of words with labels in each window/paragraph
    print ('Begin labels processing')
    with open('../resources/word_to_labels.json', 'r') as f:
        words_to_labels = json.load(f)

    items = db[book_id].find()
    for item in items:
        text = item['text']
        labels = 0
        words = nltk.word_tokenize(text)
        for word in words:
            normal_form = morph.normal_forms(word)[0]
            if normal_form in words_to_labels:
                labels += 1
        words_with_labels = labels
        labels /= item['num_of_words']
        db[book_id].update({'_id': item['_id']},
                           {'$set': {'labels_portion': labels,
                                     'words_with_labels': words_with_labels}})


def count_labels_for_windows(book_id):
    windows = db[book_id + '_pages'].find()
    for window in windows:
        words_with_labels = 0
        for i in range(window['_id'], window['end_id'] + 1):
            words_with_labels += db[book_id].find_one({'_id': i})['words_with_labels']
        labels = words_with_labels / window['num_of_words']
        db[book_id + '_pages'].update({'_id': window['_id']},
                           {'$set': {'labels_portion': labels,
                                     'words_with_labels': words_with_labels}})


def get_disjoint_windows_ids(book_id):
    table = book_id + '_pages'
    X, Y = [], []
    i = 0
    collection_size = db[table].find().count()
    while i < collection_size:
        X.append(db[table].find_one({'_id': i})['_id'])
        i = db[table].find_one({'_id': i})['end_id'] + 1

    db.books.update({'_id': book_id},
                    {'$set': {'disjoint_windows_id': list(X)}})

    windows = db[table].find()
    for window in windows:
        if window['_id'] not in X:
            db[table].remove({'_id': window['_id']})


def count_word2vecs_for_windows(book_id, model):
    print ('Building word2vec vectors...')
    windows = db[book_id + '_pages'].find()
    window_vector = np.zeros(model.vector_size)

    for window in windows:
        words_count = 0
        text = window['text']
        lemmas = mystem.lemmatize(text)
        text = ''.join(lemmas)
        words = nltk.word_tokenize(text)
        for word in words:
            if word in punctuation:
                continue
            if word not in ru_stopwords:
                try:
                    window_vector += model[word]
                    words_count += 1
                except:
                    print ('There is no word %s in word2vec model' % word)
                    pass

        window_vector /= words_count
        window_vector = window_vector.tolist()
        db[book_id + '_pages'].update({'_id': window['_id']},
                                       {'$set': {'vector': window_vector}})
    return


def count_begin_end_window_percentage(book_id):
    # [begin, end)
    print ('Begin to count percentages for windows')
    windows = db[book_id + '_pages'].find()
    book = db.books.find_one({'_id': book_id})
    total_words = book['num_of_words']

    cur = 0.0
    for window in windows:
        begin = cur
        end = begin + float(window['num_of_words']) / float(total_words)
        db[book_id + '_pages'].update({'_id': window['_id']},
                                       {'$set': {'from_percent': begin * 100,
                                                 'to_percent': end * 100}})
        cur = end


def main():

    parser = argparse.ArgumentParser(description='Book(s) processing script')
    parser.add_argument("-file", type=str, help='Path to file with fb2 book source')
    parser.add_argument("-folder", type=str, help="Path to folder with fb2 books sources")
    args = parser.parse_args()
    connect_to_database_books_collection()
    files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]
    for file in tqdm(files):
        book_id = file[:-4]
        try:
            book = open(args.folder + '/' + file).read()
        except:
            continue
        start_time = timeit.default_timer()
        book_xml = ET.XML(book)
        book_stats = process_book_simple_features_by_paragraph(book_xml, book_id)
        add_book_to_main_table(book_id, book_stats)
        build_pages_table(book_id, 1000)
        count_new_vocabulary(book_id)
        count_new_vocabulary_for_windows(book_id)
        count_sentiment(book_id)
        count_sentiment_for_windows(book_id)
        count_labels_portion(book_id)
        count_labels_for_windows(book_id)
        count_word2vecs_for_windows(book_id, word2vec_model)
        get_disjoint_windows_ids(book_id)
        count_begin_end_window_percentage(book_id)
        elapsed = timeit.default_timer() - start_time
        print('Book with id %s was processed in %s seconds \n' % (book_id, str(elapsed)))

if __name__ == "__main__":
    main()

