from xml.etree import cElementTree as ET
from pymongo import MongoClient
import os
import re
import nltk
import string
import pymorphy2
import timeit
from stop_words import get_stop_words
from bson.int64 import Int64
import json
from pymystem3 import Mystem
from PageStats import PageStats

print ('>book_processing.py')


# GLOBAL VARIABLES SECTION
punctuation = string.punctuation
punctuation += '—'
punctuation += '…'
morph = pymorphy2.MorphAnalyzer()
person_pronouns_list = ['я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они']
ru_stopwords = get_stop_words('russian')
mystem = Mystem()
DB = 'bookmate'
# GLOBAL VARIABLES SECTION END


def smooth_points(Y, N=10):
    new_Y = []
    for i in range(0, len(Y)):
        smooth_N = N
        if i - N < 0:
            smooth_N = i
            new_Y.append(Y[i])
            continue
        elif i + N >= len(Y):
            smooth_N = len(Y) - i - 1
            new_Y.append(Y[i])
            continue

        sum = 0
        for j in range(-smooth_N, smooth_N):
            sum += Y[i + j]
        sum /= ((2 * smooth_N) + 1)
        new_Y.append(sum)

    return new_Y


def get_tag(item):
    # Some magic with xml tags
    return re.sub('{[^>]+}', '', item)


def connect_to_database_books_collection():
    client = MongoClient('localhost', 27017)
    return client[DB]


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


def get_book_text_from_sections(book_id):
    db = connect_to_database_books_collection()
    text = ''
    sections = db['%s_sections' % book_id].find()

    for section in sections:
        text += section['text']
        text = text[:-1] + ' '
    text = text[:-1]
    return text


def process_book(book_id):
    print ('Process book text now')
    db = connect_to_database_books_collection()
    book_table = db['%s_pages' % str(book_id)]
    borders = db['%s_borders' % book_id].find({'page_speed': {'$exists': True}}).sort('symbol_from')

    position = Int64(0)
    page_stats = PageStats()
    page_stats.begin_symbol_pos = position
    current_border = borders.next()
    current_text_pull = ''
    full_text = ''

    text = nltk.word_tokenize(get_book_text_from_sections(book_id))

    for word in text:
        if len(full_text) <= current_border['symbol_to']:
            if word in punctuation:
                page_stats.text = page_stats.text[:-1]
                full_text = full_text[:-1]
            page_stats.text += word + ' '
            full_text += word + ' '
        else:
            current_text_pull += word + ' '
        if len(full_text) >= current_border['symbol_to']:
            page_stats._id = current_border['page']
            update_page_stats(page_stats, page_stats.text)
            page_stats.symbol_from = position
            page_stats.symbol_to = page_stats.symbol_from + page_stats.symbols_num
            page_stats.clear_text += '\n'
            get_full_page_stats(page_stats)

            page_stats.page_speed = current_border['page_speed']
            page_stats.page_sessions = current_border['page_sessions']
            page_stats.page_skip_percent = current_border['page_skip_percent']
            page_stats.page_unusual_percent = current_border['page_unusual_percent']
            page_stats.page_return_percent = current_border['page_return_percent']

            book_table.insert(page_stats.to_dict())
            position = page_stats.symbol_to + 1

            page_stats = PageStats()
            page_stats.text += current_text_pull
            full_text += current_text_pull
            current_text_pull = ''
            try:
                current_border = borders.next()
            except Exception:
                print('Last page calculated')
                return


def update_page_stats(page_stats, text):
    count_simple_text_features(page_stats, text)
    count_morphological_features(page_stats, text)


def update_book_stats(book_stats, page_stats):
    book_stats.dialogs_num += page_stats.dialogs_num
    book_stats.words_num += page_stats.words_num
    book_stats.sentences_num += page_stats.sentences_num
    book_stats.p_num += page_stats.p_num


def get_full_book_stats(book_stats):
    book_stats.avr_sentence_len = book_stats.symbols_num / book_stats.sentences_num
    # book_stats.avr_dialogs_part = book_stats.dialogs_num / book_stats.p_num
    book_stats.avr_word_len = book_stats.symbols_num / book_stats.words_num


def get_full_page_stats(page_stats):
    try:
        page_stats.person_verbs_part = page_stats.person_verbs_num / page_stats.words_num
        # page_stats.dialogs_part = page_stats.dialogs_num / page_stats.p_num
        page_stats.person_pronouns_part = page_stats.person_pronouns_num / page_stats.words_num
        page_stats.avr_word_len = page_stats.symbols_num / page_stats.words_num
    except Exception as e:
        print(e)


def count_simple_text_features(page_stats, text):
    if text is None or len(text) == 0:
        return
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
    print('Begin to process new vocabulary')
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
        db['%s_pages' % book_id].update({'_id': item['_id']},
                           {'$set': {'new_words_count': new_words_count}})


def count_sentiment(book_id):
    print("Process sentiment now.")
    db = connect_to_database_books_collection()
    items = db['%s_pages' % book_id].find()
    with open('../../resources/sentiment_dictionary.json', 'r') as f:
        sentiment_dict = json.load(f)

    for item in items:
        sentiment = 0
        sentiment_words_proportion = 0
        try:
            words = nltk.word_tokenize(item['text'])
        except:
            continue
        for word in words:
            tokens = morph.parse(word)
            for token in tokens:
                try:
                    sentiment += float(sentiment_dict[token.normal_form])
                    sentiment_words_proportion += 1
                    break
                except:
                    continue
        if len(words) > 0:
            sentiment_words_proportion /= len(words)
        else:
            sentiment = 0
            sentiment_words_proportion = 0

        db['%s_pages' % book_id].update({'_id': item['_id']},
                           {'$set': {'sentiment': sentiment,
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


def export_book_pages(book_id):
    db = connect_to_database_books_collection()
    pages = db['%s_pages' % book_id].find()

    pages_dir = '../../resources/pages/%s/%s' % (DB, book_id)
    if not os.path.exists(pages_dir):
        os.makedirs(pages_dir)
    for page in pages:
        with open('%s/%d.txt' % (pages_dir, page['_id']), 'w', encoding='utf-8') as page_file:
            page_file.write(page['text'])


def main(book_id):
    connect_to_database_books_collection()
    start_time = timeit.default_timer()

    print('Process book [%s]' % book_id)
    process_book(book_id)
    count_new_vocabulary(book_id)
    count_sentiment(book_id)
    elapsed = timeit.default_timer() - start_time
    print('Book with id %s was processed in %s seconds \n' % (book_id, str(elapsed)))


if __name__ == "__main__":
    book_ids = ['210901']
    for book_id in book_ids:
        main(book_id)
        export_book_pages(book_id)

