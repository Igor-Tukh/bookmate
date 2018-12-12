from ebooklib import epub, ITEM_DOCUMENT, ITEM_UNKNOWN, ITEM_COVER
from bs4 import BeautifulSoup
from io import StringIO
from pymongo import MongoClient
from epub_conversion.utils import open_book, convert_epub_to_lines
from xml_cleaner import to_raw_text

import nltk
import string
import os

punctuation = string.punctuation
NUMBER_OF_BYTES_PER_CHARACTER = 2


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def get_epub_book_text(book_id):
    book_path = '../epub/{book_id}.epub'.format(book_id=book_id)
    text, book = StringIO(), epub.read_epub(book_path)

    result = ''
    for item in book.get_items():
        type = item.get_type()
        if type == ITEM_DOCUMENT or type == ITEM_UNKNOWN or type == ITEM_COVER:
            soup = BeautifulSoup(item.content, 'lxml')
            result = result + soup.text
    text.write(result)
    return text.getvalue()


def get_epub_book_len(book_id):
    text = get_epub_book_text(book_id)
    return len(text)


def save_epub_book_text(book_id):
    text = get_epub_book_text(book_id)
    with open('../extracted/{book_id}.txt'.format(book_id=book_id), 'w') as file:
        file.write(text)


def get_items_summary_len(document_id):
    items_db = connect_to_mongo_database('bookmate')
    items = items_db['items'].find({'document_id': int(document_id)})
    length = 0
    for item in items:
        length += item['media_file_size']
    return length


def process_items(book_id, document_id):
    document_summary_size = get_items_summary_len(document_id)
    items_db = connect_to_mongo_database('bookmate')
    items = items_db['items'].find({'document_id': int(document_id)}).sort('position')
    document_json = {'_id': int(document_id)}
    summary_size = 0
    for item in items:
        document_json[str(item['position'])] = item
        document_json[str(item['position'])]['percent_from'] = summary_size / document_summary_size * 100
        summary_size += item['media_file_size']
        document_json[str(item['position'])]['percent_to'] = summary_size / document_summary_size * 100

    items_db['%s_documents' % book_id].insert(document_json)


def save_text_for_items(book_id, document_id):
    db = connect_to_mongo_database('bookmate')
    document_json = db['{book_id}_documents'.format(book_id=book_id)].find_one({'_id': int(document_id)})
    items = [(int(index), item) for index, item in document_json.items() if index != '_id']
    items.sort()

    text = get_epub_book_text(book_id)
    words = nltk.word_tokenize(text)

    book_len = get_epub_book_len(book_id)
    current_word_index = 0
    total_len = 0
    for index, item in items:
        item_text = ''
        while current_word_index < len(words) and len(item_text) < book_len * item['percent_to']:
            word = words[current_word_index]
            if word in punctuation:
                item_text = item_text[:-1]
            item_text += word + ' '
            current_word_index += 1
        total_len += len(item_text)
        db['{document_id}_items_text'.format(document_id=document_id)].insert_one({'document_id': int(document_id),
                                                                                   'position': item['position'],
                                                                                   'text': item_text})
    print(total_len)


def get_item(session, book_id, document_id):
    db = connect_to_mongo_database('bookmate')
    if '%s_documents' % book_id not in db.collection_names():
        process_items(book_id, document_id)
    document_json = db['{book_id}_documents'.format(book_id=book_id)].find_one({'_id': int(document_id)})
    for pos, item in document_json.items():
        if pos == '_id':
            continue
        if item['id'] == session['item_id']:
            return int(pos), item
    return -1, None


def get_text_by_session(session, book_id, document_id):
    db = connect_to_mongo_database('bookmate')
    pos, item = get_item(session, book_id, document_id)
    assert pos != -1
    if '%s_items_text' % document_id not in db.collection_names():
        save_text_for_items(book_id, document_id)
    text = db['%s_items_text' % document_id].find_one({'$and': [{'document_id': int(document_id)},
                                                                {'position': pos}]})['text']
    words = nltk.word_tokenize(text)
    from_len = session['_from'] * len(text)
    to_len = session['_to'] * len(text)

    session_text = ''
    skip = 0
    current_word_index = 0

    while current_word_index < len(words) and skip < from_len:
        skip += len(words[current_word_index]) + 1
        current_word_index += 1

    while current_word_index < len(words) and skip + len(session_text) < to_len:
        word = words[current_word_index]
        if word in punctuation:
            session_text = session_text[:-1]
        session_text += word + ' '
        current_word_index += 1

    return session_text


def get_text_by_session_using_percents(session, book_id, book_text=None):
    text = get_epub_book_text_with_ebook_convert(book_id) if book_text is None else book_text
    first = int(len(text) * session['book_from'] / 100)
    last = int(len(text) * session['book_to'] / 100)

    if not is_not_letter(text[first]):
        while first > 0 and not is_not_letter(text[first - 1]):
            first -= 1

    if not is_not_letter(text[last]):
        while last < len(text) - 1 and not is_not_letter(text[last + 1]):
            last += 1

    return text[first:last+1]


def get_epub_book_text_with_ebook_convert(book_id):
    path = os.path.join('..', 'epub', book_id + '.epub')
    output_path = os.path.join('..', 'txt_from_epub', book_id + '.txt')

    if not os.path.isfile(output_path):
        os.system("ebook-convert {path} {output_path}".format(path=path,
                                                              output_path=output_path))

    text = ''
    with open(output_path, 'r') as file:
        for line in file.readlines():
            text += line

    return text


def get_epub_book_text_experimental(book_id):
    folder = os.path.join('..', 'epub')

    book = open_book(os.path.join(folder, '{book_id}.epub'.format(book_id=book_id)))
    text = ''
    for sentence in convert_lines_to_text(convert_epub_to_lines(book)):
        for item in sentence:
            text += item

    return text


def convert_lines_to_text(lines):
    results = []
    for line in lines[:10]:
        sentence = to_raw_text(line)
        print(sentence)
        for item in sentence:
            results.append(item)
    return results


def print_document_stats(book_id, document_id):
    book_len = len(get_epub_book_text_with_ebook_convert(book_id))
    items_len = get_items_summary_len(document_id)
    print('Text len: {len}, items summary len: {items_len}, ratio: {ratio}'.format(len = book_len,
                                                                                   items_len=items_len,
                                                                                   ratio=items_len / book_len))


def is_not_letter(character):
    return character in punctuation or character == '\n' or character == ' '


if __name__ == '__main__':
    print_document_stats('135089', '1222472')
    # book_ids = ['135089']
    # document_ids = ['1222472']
    # for book_id, document_id in zip(book_ids, document_ids):
    # print('Processing book {book_id} started'.format(book_id=book_id))
    # book_len = get_epub_book_len(book_id)
    # items_len = get_items_summary_len(document_id)
    # print('Text len: {len}, items summary len: {items_len}, ratio: {ratio}'.format(len = book_len,
    #                                                                                items_len=items_len,
    #                                                                                ratio=items_len / book_len))
    # process_items(book_id, document_id)
    # save_text_for_items(book_id, document_id)
