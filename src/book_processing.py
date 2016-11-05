from xml.etree import cElementTree as ET
from pymongo import MongoClient
import argparse
import re
import nltk
import string
import os
import logging
import pymorphy2
import gensim
import numpy as np
from stop_words import get_stop_words
from tqdm import tqdm
import traceback
from bson.int64 import Int64


# GLOBAL VARIABLES SECTION
db = None
punctuation = string.punctuation
morph = pymorphy2.MorphAnalyzer()
person_pronouns_list = ['я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они']
model = gensim.models.Word2Vec.load_word2vec_format('../../ruscorpora_russe.model.bin', binary = True,
                                                    unicode_errors='ignore')

ru_stopwords = get_stop_words('russian')
# GLOBAL VARIABLES SECTION END


def config(item):
    # Some magic with xml tags
    return re.sub('{[^>]+}', '', item)


def connect_to_database_books_collection() -> None:
    client = MongoClient('localhost', 27017)
    global db
    db = client.bookmate


def get_athor_information(description) -> dict():
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


def get_section_sub_tree(item):
    for tag in item:
        if config(tag.tag) == "section":
            return get_section_sub_tree(tag)
    return item


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


def add_book_to_main_table(book, bookTable, bookStats, bookId):
    '''
    :param book: xml structure of the book
    :return: creates book's tables in database: paragraph table, window table;
    insert book info into main db table "books", return True in case of no exceptions and errors
    '''
    global db
    tree = ET.ElementTree(book)
    root = tree.getroot()
    bookItem = dict()
    for child in root:
        if config(child.tag) == "description":
            for title_info in child:
                if config(title_info.tag) == "title-info":
                    for book_title in title_info:
                        # find in database if book title already exists
                        if config(book_title.tag) == "book-title":
                            if db.books.find({"title": book_title.text}).count() != 0:
                                print("Found, skipping")
                                return
                            else:
                                title = book_title.text
                                break
            bookAuthor = get_athor_information(child)
            bookItem["title"] = title
            bookItem["_id"] = bookId
            bookItem["author"] = bookAuthor["first_name"] + ' ' + bookAuthor["second_name"] + ' ' + bookAuthor["middle_name"]

            # Book whole stats
            bookItem['num_of_words'] = bookStats['num_of_words']
            bookItem['num_of_dialogues'] = bookStats['num_of_dialogues']
            bookItem['num_of_sentences'] = bookStats['num_of_sentences']
            bookItem['num_of_symbols'] = bookStats['num_of_symbols']
            bookItem['num_of_paragraphs'] = bookTable.find().count()

            # Average book stats
            bookItem['avr_word_len'] = bookStats['num_of_symbols'] / bookStats['num_of_words']
            bookItem['avr_sentence_len'] = bookStats['num_of_words'] / bookStats['num_of_sentences']
            bookItem['avr_dialogues'] = bookStats['num_of_dialogues'] / bookItem['num_of_paragraphs']


            db.books.insert_one(bookItem)
            break
    return


def process_book_paragraphs(book, _id):
    bookTable = db[str(_id)]
    bookStats = dict()
    bookStats['num_of_words'] = 0
    bookStats['num_of_dialogues'] = 0
    bookStats['num_of_sentences'] = 0
    bookStats['num_of_symbols'] = 0
    global db
    tree = ET.ElementTree(book)
    root = tree.getroot()
    id = 1
    position = Int64(0)
    for item in tqdm(root.iter()):
        if config(item.tag) == 'p':
            pItem = process_paragraph_text(item.text)
            if pItem is None:
                continue

            pItem['begin_position'] = position
            pItem['_id'] = id
            pItem['end_position'] = position + pItem['num_of_symbols']

            try:
                bookTable.insert_one(pItem)
            except:
                traceback.print_exc()
                continue
            id += 1
            position = pItem['end_position'] + 1

            if pItem['is_dialogue']:
                bookStats['num_of_dialogues'] += 1
            bookStats['num_of_words'] += pItem['num_of_words']
            bookStats['num_of_sentences'] += pItem['num_of_sentences']
            bookStats['num_of_symbols'] += pItem['num_of_symbols']

    return bookTable, bookStats


def build_window_tables(windowSize = 2000):
    global db
    books = db.books.find()
    previous_word_all = None
    previous_word_main = None
    for book in books:
        collectionName = str(book['_id']) + '_window'
        bookTable = db[str(book['_id'])]
        if db[collectionName].find().count() > 0:
            logging.info("Window table for book id %s was found in database, skipping" % str(book['_id']))
            continue
        for _id in tqdm(range(1, bookTable.find().count())):
            window = build_window_part(bookTable, _id, windowSize)
            if _id == 1:
                previous_word_all = window['all_words_main_word']
                previous_word_main = window['main_words_main_word']

            window['similarity_with_previous_window_by_all_words'] = model.similarity(previous_word_all,
                                                                                      window['all_words_main_word'])
            window['similarity_with_previous_window_by_main_words'] = model.similarity(previous_word_main,
                                                                                       window['main_words_main_word'])

            db[collectionName].insert_one(window)
            previous_word_all = window['all_words_main_word']
            previous_word_main = window['main_words_main_word']


def process_paragraph_text(text: str) -> dict():
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

    words = nltk.word_tokenize(text)
    stats['person_verbs_num'] = 0
    stats['person_pronouns_num'] = 0

    stats['all_words_vector'] = np.zeros(model.vector_size)
    stats['main_words_vector'] = np.zeros(model.vector_size)
    for word in words:
        if word in punctuation:
            continue
        token = morph.parse(word)

        # Morphological stats
        for p in token:
            if p.tag.POS == 'NPRO':
                stats['person_verbs_num'] += 1
            if p.normal_form in person_pronouns_list:
                stats['person_pronouns_num'] += 1

        # Word2vec stats
        # All words include stopwords
        try:
            stats['all_words_vector'] += model[word]
        except:
            pass

        # All words exclude stopwords
        if word not in ru_stopwords:
            try:
                stats['main_words_vector'] += model[word]
            except:
                pass

    stats['all_words_main_word'] = model.similar_by_vector(stats['all_words_vector'])[0][0]
    stats['main_words_main_word'] = model.similar_by_vector(stats['main_words_vector'])[0][0]
    stats['all_words_vector'] = stats['all_words_vector'].tolist()
    stats['main_words_vector'] = stats['main_words_vector'].tolist()


    # TODO maybe add the list of most similar words for text
    return stats




def build_window_part(bookTable, beginId, windowSize):
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

    all_vector = np.array(np.zeros(model.vector_size).tolist())
    main_vector = np.array(np.zeros(model.vector_size).tolist())


    while(symbols < windowSize):
        paragraph = bookTable.find_one({"_id": beginId})
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
        all_vector += np.array(paragraph['all_words_vector'])
        main_vector += np.array(paragraph['main_words_vector'])
        person_verbs += paragraph['person_verbs_num']
        person_pronouns += paragraph['person_pronouns_num']

    # Basic features
    window['text'] = text
    window['dialoques_number'] = dialogues
    window['sentences_number'] = sentences
    window['window_length_in_words'] = words
    window['symbols_number'] = symbols
    window['end_id'] = beginId - 1
    window['person_verbs_num'] = person_verbs
    window['person_pronouns_num'] = person_pronouns

    # word2vec features
    window['all_words_main_word'] = model.similar_by_vector(all_vector)[0][0]
    window['main_words_main_word'] = model.similar_by_vector(main_vector)[0][0]
    window['all_words_vector'] = all_vector.tolist()
    window['main_words_vector'] = main_vector.tolist()

    # Average features per window
    window['average_word_length'] = symbols / words
    window['avr_sentence_length'] = words / sentences
    window['avr_dialogues_part'] = dialogues / paragraphs
    window['avr_person_verbs_part'] = person_verbs / words
    window['avr_person_pronouns_part'] = person_pronouns / words

    return window


def main():
    logging.basicConfig(filename='books.log', filemode='w', level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description='Book(s) processing script')
    parser.add_argument("-file", type=str, help='Path to file with fb2 book source')
    #FIXME leave folder for a future
    parser.add_argument("-folder", type=str, help="Path to folder with fb2 books sources")
    args = parser.parse_args()

    connect_to_database_books_collection()
    global db

    files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]
    for file in files:
        try:
            book = ET.XML(open(args.folder + '/' + file).read())
        except:
            continue

        try:
            if db[file[:-4]].find().count() > 0:
                logging.info("Book with id %s found in database, skipping" % str(file[:-4]))
                continue

            bookTable, bookStats = process_book_paragraphs(book, file[:-4])
            add_book_to_main_table(book, bookTable, bookStats, file[:-4])
            logging.info("Book with id %s processed and added to database" % (str(file[:-4])))
        except Exception as ex:
            traceback.print_exc()
            logging.exception("Something happened!")
            continue

    build_window_tables()
    return


if __name__ == "__main__":
    main()