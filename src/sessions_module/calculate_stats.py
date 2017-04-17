import pickle
import pandas as pd

books_users_dict = pickle.load(open("dumps/book-users-dict.pk", "rb"))
# users_books_dict = pickle.load(open("../dumps/user_books-dict.pk", "rb"))


def get_most_popular_books(n):
    global books_users_dict
    global users_books_dict

    # sort book - users by the number of users per each book
    # build a list of tuple (number_of_users, book_id) and sort by the first element
    book_users_tuples_list = []
    for book_id in books_users_dict:
        book_users = (len(books_users_dict[book_id]), book_id)
        if type(book_users[0]) is int and type(book_users[1] is int):
            book_users_tuples_list.append(book_users)
        else:
            print ('Exception in ' + str(book_users))
    book_users_tuples_list = sorted(book_users_tuples_list, key = lambda by_first_element: by_first_element[0], reverse = True)
    return book_users_tuples_list[:n]


def get_most_active_users(n):
    global books_users_dict
    global users_books_dict

    # sort users - book by the number of book per each user
    # build a list of tuple (number_of_books, user_id) and sort by the first element
    users_books_tuples_list = []
    for user_id in users_books_dict:
        user_books = (len(users_books_dict[user_id]), user_id)
        if type(user_books[0]) is int and type(user_books[1]) is int:
            users_books_tuples_list.append(user_books)
        else:
            print ('Exception in ' + str(user_books))
    users_books_tuples_list = sorted(users_books_tuples_list, key = lambda by_first_element: by_first_element[0], reverse = True)
    print ("Total number of users [%s]" % len(users_books_tuples_list))
    return users_books_tuples_list[:n]


def get_core_books_users(n_books, m_users):
    global books_users_dict
    global users_books_dict

    core_books = get_most_popular_books(n_books)
    active_users_tuples = get_most_active_users(m_users)
    active_users = []
    for active_user_tuple in active_users_tuples:
        active_users.append(active_user_tuple[1])

    core_books_users = dict()
    for core_book in core_books:
        book_id = core_book[1]
        book_readers_all = books_users_dict[book_id]
        book_readers_active = []
        for reader in book_readers_all:
            if reader in active_users:
                book_readers_active.append(reader)
        core_books_users[book_id] = book_readers_active

    for book in core_books_users:
        used_book_id = book
        result = set(core_books_users[book])
        break

    for book in core_books_users:
        if book != used_book_id:
            result.intersection_update(core_books_users[book])

    for book in core_books_users:
        print ("book_id [%s] with [%s] core_users" % (book, len(core_books_users[book])))
    print(len(result))


most_popular_books = get_most_popular_books(1000)
all_books = pd.read_csv('meta/books.csv')
place = 1
for book in most_popular_books:
    book_id = book[1]
    if book_id != None:
        book_row = all_books[all_books['id'] == book_id]
        try:
            book_row = list(book_row.values)[0]
            print ('%d. %s, %s, %s, %s, %d' % (place, str(book_row[0]), str(book_row[1]), str(book_row[2]), str(book_row[3]), book[0]))
            place += 1
        except:
            continue