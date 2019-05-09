import logging
import matplotlib.pyplot as plt
import numpy as np


from src.metasessions_module.user_utils import *
from src.metasessions_module.sessions_utils import *
from src.metasessions_module.utils import *
from src.metasessions_module.text_utils import *
from src.metasessions_module.config import *


def get_batches(book_id, batches_amount):
    logging.info('Creating {} batches for users of book {}'.format(batches_amount, book_id))
    user_ids = get_good_users_info(book_id).keys()
    batches = np.zeros(len(user_ids), batches_amount)
    batch_percent = 100.0 / batches_amount

    for user_ind, user_id in tqdm(enumerate(user_ids)):
        batch_to = batch_percent
        batch_ind = 0
        current_speeds = []

        document_id = get_user_document_id(book_id, user_id)
        unique_sessions = get_user_sessions_by_place_in_book(book_id, document_id, user_id)

        places = list(unique_sessions.keys())
        places.sort(key=lambda val: val[0])
        for place in places:
            session = unique_sessions[place]
            book_from, book_to = place
            while book_from > batch_to and batch_ind < batches_amount:
                if len(current_speeds) > 0:
                    batches[user_ind][batch_ind] = sum(current_speeds) / len(current_speeds)
                    current_speeds = []
                batch_to, batch_ind = batch_to + batch_percent, batch_ind + 1
            if batches_amount > batch_ind and is_target_speed(session['speed']):
                current_speeds.append(session['speed'])
        if len(current_speeds) > 0:
            batches[user_ind][batch_ind] = sum(current_speeds) / len(current_speeds)

    return batches, user_ids


def load_batches(book_id, batches_amount, rebuild=False):
    """
    Uploads batches file for certain book id and batches amount if it exists and flag rebuild is not selected.
    Otherwise splits all user sessions to batches_amount sequent slices.
    :return: tuple of np.array of shape (users_cnt, batches_amount) and np.array of shape (users_cnt) -- batches average
    speeds and user ids respectively
    """
    output_batches_path = os.path.join('resources', 'batches', '{}_{}.pkl'.format(book_id, batches_amount))
    if rebuild or not os.path.isfile(output_batches_path):
        batches_and_users = get_batches(book_id, batches_amount)
        save_via_pickle(batches_and_users, output_batches_path)
        return batches_and_users
    return load_from_pickle(output_batches_path)


def cluster_users_by_batches_speed_sklearn(book_id, batches_amount, algo):
    batches, user_ids = load_batches(book_id, batches_amount)
    algo.fit(batches)
    return batches, algo.labels_


def cluster_users_by_batches_speed_sklearn_k_means(book_id, batches_amount, clusters_amount, random_state=23923):
    return cluster_users_by_batches_speed_sklearn(book_id, batches_amount, KMeans(n_clusters=clusters_amount,
                                                                                  random_state=random_state))


def visualize_batches_speed_clusters(book_id, batches, labels, plot_title, plot_name, color_function):
    batches_amount = batches.shape[1]
    inds = np.argsort(labels)
    batches = batches[inds, :]
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Users')
    ax.set_title(plot_title)
    ax.set_xlim(0.0, 100.0)
    batch_percent = 100.0 / batches_amount
    ax.set_ylim(batch_percent * batches.shape[1] + batch_percent / 2)

    for user_ind, speeds in tqdm(enumerate(batches)):
        batch_from = batch_percent / 2
        for ind in range(batches_amount):
            users_y = batch_percent * user_ind + batch_percent / 2
            if speeds[ind] is None:
                circle = plt.Circle((batch_from, users_y), batch_percent / 2, color='white')
            else:
                circle = plt.Circle((batch_from, users_y), batch_percent / 2, color=color_function(user_ind, speeds))
            ax.add_artist(circle)
            batch_from += batch_percent

    chapters_lens = get_chapter_percents(book_id, DOCUMENTS[book_id][0])
    prev_len = 0
    ticks_pos = []
    for chapter_len in chapters_lens:
        ticks_pos.append((chapter_len + prev_len) / 2)
        prev_len = chapter_len
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(BOOK_LABELS[book_id], rotation=90)

    plot_path = os.path.join('resources', 'plots', 'batches_clusters', plot_name)
    plt.savefig(plot_path)


if __name__ == '__main__':
    pass
