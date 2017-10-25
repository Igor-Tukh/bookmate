from pymongo import MongoClient
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import numpy as np
plt.style.use('ggplot')


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def smooth_points(Y, N=10):
    new_Y = []
    for i in range(0, len(Y)):
        smooth_N = N
        if i - N < 0:
            new_Y.append(Y[i])
            continue
        elif i + N >= len(Y):
            new_Y.append(Y[i])
            continue
        else:
            sum = 0
            for j in range(-smooth_N, smooth_N):
                sum += Y[i + j]
            sum /= ((2 * smooth_N) + 1)
            new_Y.append(sum)

    return new_Y


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

def main():
    book_ids = ['2206', '2207', '2543', '2289', '135089']
    fields = ['page_speed', 'page_skip_percent', 'page_unusual_percent']
    dbs = ['bookmate', 'bookmate_male', 'bookmate_female', 'bookmate_paid', 'bookmate_free']

    for book_id in book_ids:
        for field in fields:
            plot_data(dbs, book_id, field)


if __name__ == "__main__":
    main()
