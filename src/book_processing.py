from xml.etree import cElementTree as ET
from pymongo import MongoClient
import argparse
import re
import nltk
import string
from lxml import etree

# GLOBAL VARIABLES SECTION
db = None
punctuation = string.punctuation
# GLOBAL VARIABLES SECTION END



def config(item):
    # Some magic with xml tags
    return re.sub('{[^>]+}', '', item)


def connect_to_database_books_collection():
    client = MongoClient('localhost', 27017)
    db = client.bookmate
    return db


def getAuthorInformation(description):
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


def getSectionSubTree(item):
    for tag in item:
        if config(tag.tag) == "section":
            return getSectionSubTree(tag)
    return item


def numberOfWords(text):
    '''
    :param text: Text from one paragraph
    :return: Number of words in text exclude the punctuation
    '''
    try:
        text = nltk.word_tokenize(text)
        text = [word for word in text if word not in punctuation]
        return len(text)
    except:
        return 0


def initBookTables(book):
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
            bookAuthor = getAuthorInformation(child)
            bookItem["title"] = title
            bookItem["_id"] = db.book.count() + 1
            bookItem["author"] = bookAuthor["first_name"] + ' ' + bookAuthor["second_name"] + ' ' + bookAuthor["middle_name"]
            #FIXME move to the place after whole book processing, because if book processing failes, we don't need it in database
            #db.books.insert_one(bookItem)
            bookTable = db[title]

        # Process book paragraphs
        elif re.sub('{[^>]+}', '', child.tag) == "body":

            section = 0
            id = 1
            position = 1
            partInSection = 1
            viewSectionElement = False # we don't need parsing 'p' tags if we not in the section
            for item in child.iter():
                print(item.tag)
                if config(item.tag) == "section":
                    section += 1
                    viewSectionElement = True

                if config(item.tag) == 'p':
                    if not viewSectionElement:
                        continue
                    p = item
                    pItem = dict()
                    pItem["section"] = section
                    pItem["begin_position"] = position
                    pItem["pos_in_section"] = partInSection
                    pItem["text"] = p.text
                    pItem["num_of_words"] = numberOfWords(p.text)
                    try:
                        text_len = len(p.text)
                    except:
                        text_len = 0
                    pItem["numOfSymbols"] = text_len
                    pItem["_id"] = id
                    pItem["end_position"] = position + text_len
                    pItem["pos_in_section"] = partInSection

                    if text_len != 0:
                        try:
                            bookTable.insert_one(pItem)
                            id += 1
                            position += text_len + 1
                            partInSection += 1
                        except:
                            continue
    return




def main():

    parser = argparse.ArgumentParser(description='Book(s) processing script')
    parser.add_argument("-file", type=str, help='Path to file with fb2 book source')
    #FIXME leave folder for a future
    parser.add_argument("-folder", type=str, help="Path to folder with fb2 books sources")
    args = parser.parse_args()
    # e = ET.parse(open(args.file))
    # for item in e.iter():
    #     print(item)
    book = ET.XML(open(args.file).read())

    global db
    db = connect_to_database_books_collection()
    initBookTables(book)

    return


if __name__ == "__main__":
    main()