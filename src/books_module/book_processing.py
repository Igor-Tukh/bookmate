from xml.etree import cElementTree as ET
from pymongo import MongoClient
import argparse
import re
import nltk
import string
import os
import pymorphy2
import timeit
from stop_words import get_stop_words
from tqdm import tqdm
import traceback
from bson.int64 import Int64
import json
import numpy as np
from pymystem3 import Mystem
from PageStats import PageStats
from BookStats import BookStats


# GLOBAL VARIABLES SECTION
punctuation = string.punctuation
punctuation += '—'
punctuation += '…'
morph = pymorphy2.MorphAnalyzer()
person_pronouns_list = ['я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они']
ru_stopwords = get_stop_words('russian')
mystem = Mystem()
# GLOBAL VARIABLES SECTION END


def config(item):
    # Some magic with xml tags
    return re.sub('{[^>]+}', '', item)


def connect_to_database_books_collection():
    client = MongoClient('localhost', 27017)
    return client.bookmate


def number_of_words(text):
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


def process_book(book, _id, page_size):
    print ("\nProcess book text now.")
    db = connect_to_database_books_collection()
    book_table = db['%s_pages' % str(_id)]
    book_stats = BookStats()
    book_stats._id = str(_id)
    root = ET.ElementTree(book).getroot()
    id = 0
    position = Int64(0)
    page_stats = PageStats()
    page_stats.begin_symbol_pos = position
    section_flag = False  # if we are inside section or not
    section_num = 1
    for item in root.iter():
        if config(item.tag) == 'section':
            if not section_flag:
                section_flag = True
                page_stats.begin_of_section = True
            else:
                section_flag = False
                if (page_stats.words_num == 0):
                    continue
                page_stats.end_of_section = True
                page_stats._id = id
                page_stats.section_num = section_num
                page_stats.clear_text += '\n'
                book_table.insert(page_stats.to_dict())
                position += page_stats.symbols_num + 1
                page_stats = PageStats()
                page_stats.begin_symbol_pos = position
                id += 1
                section_num += 1
                continue

        if config(item.tag) == 'p':
            page_stats.p_num += 1
            page_stats.section_num = section_num
            update_page_stats(page_stats, item.text)

        if page_stats.symbols_num >= page_size:
            page_stats._id = id
            book_stats.pages_num += 1
            position += page_stats.symbols_num + 1
            page_stats.section_num += 1
            page_stats.clear_text += '\n'
            book_table.insert(page_stats.to_dict())
            update_book_stats(book_stats, page_stats)
            page_stats = PageStats()
            page_stats.begin_symbol_pos = position
            id += 1
    get_full_book_stats(book_stats)
    db.books.insert(book_stats.to_dict())


def update_page_stats(page_stats, text):
    count_simple_text_features(page_stats, text)
    count_morphological_features(page_stats, text)


def update_book_stats(book_stats, page_stats):
    book_stats.symbols_num += page_stats.symbols_num
    book_stats.dialogs_num += page_stats.dialogs_num
    book_stats.words_num += page_stats.words_num
    book_stats.sentences_num += page_stats.sentences_num
    book_stats.p_num += page_stats.p_num


def get_full_book_stats(book_stats):
    book_stats.avr_sentence_len = book_stats.symbols_num / book_stats.sentences_num
    book_stats.avr_dialogs_part = book_stats.dialogs_num / book_stats.p_num
    book_stats.avr_word_len = book_stats.symbols_num / book_stats.words_num


def get_full_page_stats(page_stats):
    page_stats.person_verbs_part = page_stats.person_verbs_num / page_stats.words_num
    page_stats.dialogs_part = page_stats.dialogs_num / page_stats.p_num
    page_stats.person_pronouns_part = page_stats.person_pronouns_num / page_stats.words_num


def count_simple_text_features(page_stats, text):
    if text is None or len(text) == 0:
        return
    page_stats.text += text
    if text[0] == '—' or text[0] == '–':
        page_stats.dialogs_num += 1
    page_stats.words_num += number_of_words(text)
    page_stats.sentences_num += number_of_sentences(text)
    page_stats.symbols_num += len(text)


def count_morphological_features(page_stats, text):
    try:
        words = nltk.word_tokenize(text)
    except:
        return
    for word in words:
        if word in punctuation:
            continue
        token = morph.parse(word)
        for p in token:
            if p.tag.POS == 'NPRO':
                page_stats.person_verbs_num += 1
            if p.normal_form in person_pronouns_list:
                page_stats.person_pronouns_num += 1
            page_stats.clear_text += p.normal_form + ' '
            # use break here to process only most possible word form
            break


def count_new_vocabulary(book_id):
    print("Process new vocabulary now.")
    new_vocabulary = dict()
    db = connect_to_database_books_collection()
    items = db['%s_pages' % book_id].find()
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
                           {'$set': {'new_words_count': new_words_count}})


def count_sentiment(book_id):
    print("Process sentiment now.")
    db = connect_to_database_books_collection()
    items = db['%s_pages' % book_id].find()
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
        if len(words) > 0:
            avr_sentiment /= len(words)
            sentiment_words_proportion /= len(words)
        else:
            avr_sentiment = 0
            sentiment_words_proportion = 0
            sum_sentiment = 0

        db[book_id].update({'_id': item['_id']},
                           {'$set': {'avr_sentiment': avr_sentiment,
                                     'sum_sentiment': sum_sentiment,
                                     'sentiment_words_portion': sentiment_words_proportion}})


def count_labels_portion(book_id):
    print ('Process labeled words now')
    with open('../resources/word_to_labels.json', 'r') as f:
        words_to_labels = json.load(f)
    db = connect_to_database_books_collection()
    items = db['%s_pages' % book_id].find()
    for item in items:
        text = item['text']
        labels = 0
        words = nltk.word_tokenize(text)
        for word in words:
            normal_form = morph.normal_forms(word)[0]
            if normal_form in words_to_labels:
                labels += 1
        words_with_labels = labels
        if item['words_num'] > 0:
            labels /= item['words_num']
        else:
            labels = 0
        db[book_id].update({'_id': item['_id']},
                           {'$set': {'labels_part': labels,
                                     'labeled_words_num': words_with_labels}})


def count_percents_for_pages(book_id):
    print ('Percentage calculation begins...')
    db = connect_to_database_books_collection()
    pages = db['%s_pages' % book_id].find()
    book = db['books'].find_one({'_id': book_id})
    _from, _to = 0.0, 0.0
    for page in pages:
        _from = _to
        _to = _from + page['symbols_num'] / book['symbols_num']
        db['%s_pages' % book_id].update({'_id': page['_id']},
                                        {'$set': {'_from': _from * 100.0,
                                                  '_to': _to * 100.0}})


def main():

    parser = argparse.ArgumentParser(description='Book(s) processing script')
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
        process_book(book_xml, book_id, 1000)
        count_new_vocabulary(book_id)
        count_sentiment(book_id)
        count_percents_for_pages(book_id)
        elapsed = timeit.default_timer() - start_time
        print('Book with id %s was processed in %s seconds \n' % (book_id, str(elapsed)))

if __name__ == "__main__":
    main()

