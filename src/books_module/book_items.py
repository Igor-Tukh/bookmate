from pymongo import MongoClient
import xml.etree.ElementTree as ET
import re


BOOKMATE_DB = 'bookmate_1'


def connect_to_database_books_collection(db):
    client = MongoClient('localhost', 27017)
    return client[db]


def get_tag(item):
    # Some magic with xml tags
    return re.sub('{[^>]+}', '', item)


def define_sections(book_id, book_file):
    db = connect_to_database_books_collection(BOOKMATE_DB)
    root = ET.ElementTree(book_file).getroot()
    _id = 1
    symbol_from = 0
    for item in root.iter():
        if get_tag(item.tag) == 'section':
            section = dict()
            section['_id'] = _id
            section['text'] = ''
            section['symbol_from'] = symbol_from
            for section_item in item.iter():
                if type(section_item.text) is str and len(section_item.text.strip(' \t\n\r')) > 0 :
                    section['text'] += section_item.text.strip(' \t\n\r') + '\n'
                    # section['text'] += section_item.text + '\n'
            section['symbol_to'] = symbol_from + len(section['text'])
            section['size'] = len(section['text'])
            db['%s_sections' % book_id].insert(section)
            _id += 1
            symbol_from = symbol_from + section['size'] + 1


def get_book_size(book_id):
    db = connect_to_database_books_collection(BOOKMATE_DB)
    sections = db['%s_sections' % book_id].find()
    size = 0
    for section in sections:
        size += section['size']
    print('[%s] size is %d symbols' % (book_id, size))


def process_documents(book_id):
    db = connect_to_database_books_collection(BOOKMATE_DB)
    documents = db['%s_items' % book_id].find().distinct('document_id')



def process_book(book_id):
    books_folder = '/Users/kseniya/Documents/WORK/bookmate/code/resources/in'
    book_file = '%s/%s.fb2' % (books_folder, book_id)
    book_xml = ET.XML(open(book_file, 'r').read())

    define_sections(book_id, book_xml)

book_id = '2289'
process_book(book_id)
get_book_size(book_id)



