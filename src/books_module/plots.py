from pymongo import MongoClient
import pymongo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly
import plotly.graph_objs as go
import numpy as np
plt.style.use('ggplot')


BOOKMATE_DB = 'bookmate'
FULL_SESSIONS_DB = 'sessions'

def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def plot_data(dbs, book_id, field):
    plot_data = []
    for db in dbs:
        database = connect_to_mongo_database(db)
        data = database['%s_pages' % book_id].find().distinct(field)
        data = np.convolve(data, np.ones(10)/10)
        trace = go.Scatter(
            x = list(range(0, len(data))),
            y = data,
            mode = 'lines',
            name = db.title()
        )
        plot_data.append(trace)

    plotly.offline.iplot(plot_data, filename='%s_%s.png' % (book_id, field))


def plot_user_sessions(book_id, sessions, read_percents, save_path):
    plt.clf()
    figure, axes = plt.subplots(dpi=600)
    max_y, max_x = 0, connect_to_mongo_database(BOOKMATE_DB)['books'].find_one({'_id': book_id})['symbols_num']
    colors = ['royalblue', 'gold', 'firebrick']
    read_fragments = dict()
    for session in sessions:
        if '%d_%d' % (session['symbol_from'], session['symbol_to']) not in read_fragments:
            read_fragments['%d_%d' % (session['symbol_from'], session['symbol_to'])] = 0
        else:
            read_fragments['%d_%d' % (session['symbol_from'], session['symbol_to'])] += 1
        if read_fragments['%d_%d' % (session['symbol_from'], session['symbol_to'])] < len(colors):
            color = colors[read_fragments['%d_%d' % (session['symbol_from'], session['symbol_to'])]]
        else:
            color='crimson'
        axes.add_patch(
            patches.Rectangle(
                (session['symbol_from'], 0),
                session['symbol_to'] - session['symbol_from'],
                session['abs_speed'],
                color=color
            )
        )
        max_y = max(max_y, session['abs_speed'])
    axes.set_xlim(0, max_x)
    axes.set_ylim(0, max_y)
    axes.set_xlabel('Symbols')
    axes.set_ylabel('User Relative Speed')
    axes.set_title('%.3f percents of coverage' % read_percents)
    figure.savefig(save_path, bbox_inches = 'tight')
    plt.close(figure)


def plot_users(book_id, min_percent=50.0, max_percent=80.0, imgs_num=75):
    db = connect_to_mongo_database(BOOKMATE_DB)
    users = db['%s_users' % book_id].find().sort([('read_percents', pymongo.DESCENDING)])
    print ('Drawing plots for %d users' % users.count())

    imgs = 0
    for user in users:
        if min_percent <= user['read_percents'] <= max_percent:
            user_sessions = db[book_id].find({'user_id': user['_id']})
            if user_sessions.count() > 0:
                plot_user_sessions(book_id, user_sessions, user['read_percents'],
                                   save_path='images/users/%s/%s.png' % (book_id, user['_id']))
                imgs += 1
                if imgs > imgs_num:
                    return


def plot_speed_distribution(book_id):
    db = connect_to_mongo_database(BOOKMATE_DB)
    speeds = db[book_id].find().distinct('speed')
    plt.clf()
    plt.hist(speeds)
    plt.title("book_id=%s speed distribution" % book_id)
    plt.xlabel('Speed sym/min')
    plt.ylabel('Number of sessions')
    plt.savefig('images/%s_speed_distr.png' % book_id)


def plot_books_distribution():
    db = connect_to_mongo_database(FULL_SESSIONS_DB)
    all_users = db['sessions'].find().distinct()


def main():
    # book_ids = ['2206', '2207', '2543', '2289', '135089']
    # fields = ['page_speed', 'page_skip_percent', 'page_unusual_percent']
    # dbs = ['bookmate', 'bookmate_male', 'bookmate_female', 'bookmate_paid', 'bookmate_free']
    #
    # for book_id in book_ids:
    #     for field in fields:
    #         plot_data(dbs, book_id, field)
    # plot_users(book_id='2289')
    plot_speed_distribution('2289')
    plot_speed_distribution('210901')

if __name__ == "__main__":
    main()
