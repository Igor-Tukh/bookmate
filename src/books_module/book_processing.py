from xml.etree import cElementTree as ET
from pymongo import MongoClient
import argparse
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
from BookStats import BookStats
import codecs
import matplotlib.pyplot as plt
print ('>book_processing.py')


# GLOBAL VARIABLES SECTION
punctuation = string.punctuation
punctuation += '—'
punctuation += '…'
morph = pymorphy2.MorphAnalyzer()
person_pronouns_list = ['я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они']
ru_stopwords = get_stop_words('russian')
mystem = Mystem()
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


def get_book_size_in_symbols(book, book_id):
    print ('\nCalculate book size now')
    db = connect_to_database_books_collection()
    root = ET.ElementTree(book).getroot()
    book_stats = BookStats()
    book_stats._id = str(book_id)

    full_book_text = ''
    for item in root.iter():
        if config(item.tag) == 'p':
            if item.text is None:
                continue
            full_book_text += item.text + ' '
    book_stats.symbols_num = len(full_book_text)
    book_stats.text = full_book_text
    try:
        db['books'].insert({'_id': book_id,
                            'symbols_num': book_stats.symbols_num})
    except Exception as e:
        print (e)
        db.books.update({'_id': book_id},
                        {'$set': {'symbols_num': book_stats.symbols_num}})


def process_book(book, book_id):
    print ('Process book text now')
    db = connect_to_database_books_collection()
    book_table = db['%s_pages' % str(book_id)]
    root = ET.ElementTree(book).getroot()
    borders = db['%s_borders' % book_id].find({'avr_abs_speed': {'$exists': True}})

    position = Int64(0)
    page_stats = PageStats()
    page_stats.begin_symbol_pos = position
    current_border = borders.next()
    current_text_pull = ''
    full_text = ''

    for item in root.iter():
        # if len(current_text) <= current_border['symbol_to']:
        if config(item.tag) == 'p':
            if item.text is None:
                continue
            else:
                page_stats.p_num += 1
                item_text = nltk.word_tokenize(item.text)
                for word in item_text:
                    if len(full_text) <= current_border['symbol_to']:
                        page_stats.text += word + ' '
                        full_text += word + ' '
                    else:
                        current_text_pull += word + ' '
                if len(full_text) >= current_border['symbol_to']:
                    page_stats._id = current_border['page']
                    update_page_stats(page_stats, page_stats.text)
                    get_full_page_stats(page_stats)
                    page_stats._from = position
                    page_stats._to = page_stats._from + page_stats.symbols_num
                    page_stats.clear_text += '\n'
                    book_table.insert(page_stats.to_dict())
                    position = page_stats._to + 1

                    page_stats = PageStats()
                    page_stats.text += current_text_pull
                    full_text += current_text_pull
                    current_text_pull = ''
                    try:
                        current_border = borders.next()
                    except Exception:
                        return

                else:
                    if item.text[0] == '-':
                        page_stats.dialogs_num += 1


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
    page_stats.person_verbs_part = page_stats.person_verbs_num / page_stats.words_num
    page_stats.dialogs_part = page_stats.dialogs_num / page_stats.p_num
    page_stats.person_pronouns_part = page_stats.person_pronouns_num / page_stats.words_num
    page_stats.avr_word_len = page_stats.symbols_num / page_stats.words_num


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
    with open('../resources/sentiment_dictionary.json', 'r') as f:
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
                                        {'$set': {'percent_from': _from * 100.0,
                                                  'percent_to': _to * 100.0}})


def plot_book_stats(book_id):
    db = connect_to_database_books_collection()
    pages = db['%s_pages' % book_id].find().sort('_id')
    symbols, avr_word_len, person_pronouns, person_verbs, sentiment, sentiment_words, new_words = list(), list(), list(), \
                                                                                       list(), list(), list(), list()

    page_begin = 0
    for page in pages:
        symbols_point = int((page_begin * 2 + page['symbols_num']) / 2)
        symbols.append(symbols_point)
        avr_word_len.append(page['avr_word_len'])
        person_pronouns.append(page['person_pronouns_part'])
        person_verbs.append(page['person_verbs_part'])
        sentiment.append(page['sentiment'])
        sentiment_words.append(page['sentiment_words_portion'])
        new_words.append(page['new_words_count'] / page['words_num'])
        page_begin += page['symbols_num'] + 1

    plt.clf()
    # plt.plot(symbols, avr_word_len, label='avr_word_len')
    plt.plot(symbols, smooth_points(person_pronouns, 10), label='Person Pronouns')
    plt.plot(symbols, smooth_points(person_verbs, 10), label='Person Verbs')
    # plt.plot(symbols, sentiment, label='sentiment')
    plt.plot(symbols, smooth_points(sentiment_words, 10), label='Sentiment Words')
    plt.plot(symbols, smooth_points(new_words, 10), label='New Words')

    plt.legend(prop={'size': 16})
    plt.title('Textual Features for Fifty Shadows of Gray')
    plt.savefig('%s_stats.png' % book_id)


def main(is_calculate_size, is_process_book):
    parser = argparse.ArgumentParser(description='Book(s) processing script')
    parser.add_argument("-folder", type=str, help="Path to folder with fb2 books sources")
    args = parser.parse_args()
    connect_to_database_books_collection()
    start_time = timeit.default_timer()
    # book_ids = ['2207', '2289', '2543', '11833', '210901', '259222', '266700', '275066']
    book_ids = ['2289']
    if is_calculate_size:
        for book_id in book_ids:
            try:
                book = open(args.folder + '/' + book_id + '.fb2', encoding='utf-8').read()
            except Exception as e:
                print(e)
                continue
            book_xml = ET.XML(book)
            get_book_size_in_symbols(book_xml, book_id)

    if is_process_book:
        for book_id in book_ids:
            try:
                book = codecs.open(args.folder + '/' + book_id + '.fb2', 'r', encoding='utf-8').read()
            except Exception as e:
                print(e)
                continue
            book_xml = ET.XML(book)
            process_book(book_xml, book_id)
            count_new_vocabulary(book_id)
            count_sentiment(book_id)
            count_percents_for_pages(book_id)
            elapsed = timeit.default_timer() - start_time
            print('Book with id %s was processed in %s seconds \n' % (book_id, str(elapsed)))
            plot_book_stats(book_id)

if __name__ == "__main__":
    is_calculate_size = False
    is_process_book = True
    main(is_calculate_size, is_process_book)

