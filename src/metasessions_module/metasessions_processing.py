import argparse
import logging
import pickle
import math
import sys
import csv
import os

import matplotlib.pyplot as plt


from collections import defaultdict
from tqdm import tqdm
from enum import Enum

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.config import *
from src.metasessions_module.utils import *
from src.metasessions_module.sessions_utils import *
from src.metasessions_module.user_utils import *
from src.metasessions_module.text_utils import *
from src.metasessions_module.item_utils import *


logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'metasessions_processing.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000


def get_metasessions_breaks(book_id, document_id, user_id, break_time_seconds=1800):
    logging.info('Looking for metasessions break for user {} and document {} of book {}'.format(user_id,
                                                                                                document_id,
                                                                                                book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    if len(user_sessions) == 0:
        return []
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    previous_session = user_sessions[0]
    breaks = []
    for session in user_sessions[1:]:
        time = (date_from_timestamp(session['read_at']) - date_from_timestamp(previous_session['read_at'])) \
            .total_seconds()
        if time > break_time_seconds:
            breaks.append(session)
        previous_session = session
    return breaks


def save_metasessions_by_reading_style(book_id, document_id, user_id):
    logging.info('Start saving metasessions by reading style for user {} and document {} of book {}'.format(user_id,
                                                                                                            document_id,
                                                                                                            book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    if len(user_sessions) == 0:
        return
    if 'speed' not in user_sessions[0]:
        save_user_sessions_speed(book_id, document_id, user_id)
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    metasessions = [[user_sessions[0]]]
    previous_reading_style = ReadingStyle.get_reading_style_by_speed(user_sessions[0]['speed'])
    for session in user_sessions[1:]:
        current_reading_style = ReadingStyle.get_reading_style_by_speed(session['speed'])
        if current_reading_style == previous_reading_style:
            metasessions[-1].append(session)
        else:
            previous_reading_style = current_reading_style
            metasessions.append([session])

    logging.info('Total found {} metasessions'.format(len(metasessions)))
    metasessions_path = os.path.join('resources', 'metasessions_style', '{}_{}_{}.pkl'.format(book_id,
                                                                                              document_id,
                                                                                              user_id))
    with open(metasessions_path, 'wb') as file:
        pickle.dump(metasessions, file)
    logging.info('Metasessions by style of document {} for user {} saved to {}'.format(document_id,
                                                                                       user_id,
                                                                                       metasessions_path))


def save_metasessions_by_deviant_percent(book_id, document_id, user_id, deviant_percent=100, first_sessions_amount=3):
    logging.info('Start saving metasessions by reading style for user {} and document {} of book {}'.format(user_id,
                                                                                                            document_id,
                                                                                                            book_id))
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    if len(user_sessions) == 0:
        return
    if 'speed' not in user_sessions[0]:
        save_user_sessions_speed(book_id, document_id, user_id)
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    metasessions = [[user_sessions[0]]]
    for session in user_sessions[1:]:
        if len(metasessions[-1]) < 3:
            if len(metasessions[-1]) > 0:
                if abs(session['book_from'] - metasessions[-1][-1]['book_from']) < 0.3:
                    metasessions[-1].append(session)
                else:
                    metasessions.append([session])
            else:
                metasessions.append([session])
        else:
            base_speed = sum([metasessions[-1][i]['speed']
                              for i in range(first_sessions_amount)]) / first_sessions_amount
            big_skip = abs(session['book_from'] - metasessions[-1][-1]['book_from']) > 0.3
            if not big_skip and session['speed'] != INFINITE_SPEED and session['speed'] != UNKNOWN_SPEED \
                    and session['speed'] > 1e-6 and \
                    abs(base_speed - session['speed']) / session['speed'] < deviant_percent / 100:
                metasessions[-1].append(session)
            else:
                metasessions.append([session])

    logging.info('Total found {} metasessions'.format(len(metasessions)))
    metasessions_path = os.path.join('resources', 'metasessions_deviant_percent', '{}_{}_{}.pkl'.format(book_id,
                                                                                                        document_id,
                                                                                                        user_id))
    with open(metasessions_path, 'wb') as file:
        pickle.dump(metasessions, file)
    logging.info('Metasessions by style of document {} for user {} saved to {}'.format(document_id,
                                                                                       user_id,
                                                                                       metasessions_path))


def get_metasessions_by_reading_style(book_id, document_id, user_id):
    metasessions_path = os.path.join('resources', 'metasessions_style', '{}_{}_{}.pkl'.format(book_id,
                                                                                              document_id,
                                                                                              user_id))
    if not os.path.isfile(metasessions_path):
        logging.error('Unable to find metasessions by reading style of document {} for user {}'.format(document_id,
                                                                                                       user_id))
        save_metasessions_by_reading_style(book_id, document_id, user_id)

    with open(metasessions_path, 'rb') as file:
        return pickle.load(file)


def get_metasessions_by_deviant_percent(book_id, document_id, user_id):
    metasessions_path = os.path.join('resources', 'metasessions_deviant_percent', '{}_{}_{}.pkl'.format(book_id,
                                                                                                        document_id,
                                                                                                        user_id))
    if not os.path.isfile(metasessions_path):
        logging.error('Unable to find metasessions by reading style of document {} for user {}'.format(document_id,
                                                                                                       user_id))
        save_metasessions_by_deviant_percent(book_id, document_id, user_id)

    with open(metasessions_path, 'rb') as file:
        return pickle.load(file)


def visualize_metasessions_by_reading_style(book_id, document_id, user_id):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 10))
    # plt.xlabel('Book percent')
    # plt.ylabel('Session speed')
    # plt.title('Metasessions visualization')
    # plt.ylim(50.0, 6000.0)
    # plt.xlim(0.0, 100.0)
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Session speed')
    ax.set_title('Metasessions visualization')
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(50.0, 6000.0)
    metasessions = get_metasessions_by_reading_style(book_id, document_id, user_id)
    chapters_lens = get_chapter_percents(book_id, document_id)
    prev_len = 0
    ticks_pos = []
    for chapter_len in chapters_lens:
        ticks_pos.append((chapter_len + prev_len) / 2)
        prev_len = chapter_len
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(BOOK_LABELS[book_id], rotation=70)
    for chapter_len in chapters_lens[:-1]:
        # plt.axvline(x=chapter_len, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=chapter_len, color='black', linestyle='--', linewidth=0.5)
    breaks = get_metasessions_breaks(book_id, document_id, user_id)
    logging.info('Found {} timebreaks'.format(len(breaks)))
    for time_break in breaks:
        # plt.axvline(x=time_break['book_from'], color='red', linestyle='--', linewidth=0.4)
        ax.axvline(x=time_break['book_from'], color='red', linestyle='--', linewidth=0.4)
    for metasession in metasessions:
        avg_speed = sum([session['speed'] for session in metasession]) / len(metasession)
        x = [session['book_from'] for session in metasession]
        y = [avg_speed for _ in metasession]
        # plt.plot(x, y, ReadingStyle.get_reading_style_by_speed(metasession[0]['speed']).get_color(), markersize=2)
        ax.plot(x, y, ReadingStyle.get_reading_style_by_speed(metasession[0]['speed']).get_color(), markersize=2)
    # plt.tick_params(axis='x', rotation=70)
    plt.savefig(os.path.join('resources', 'plots', 'reading_style', '{}_{}_{}_metasessions_by_reading_style.png')
                .format(book_id, document_id, user_id))


def visualize_metasessions_by_deviant_percent(book_id, document_id, user_id):
    fig, ax = plt.subplots(figsize=(12, 10))
    # plt.xlabel('Book percent')
    # plt.ylabel('Session speed')
    # plt.title('Metasessions visualization')
    # plt.ylim(50.0, 6000.0)
    # plt.xlim(0.0, 100.0)
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Session speed')
    ax.set_title('Metasessions visualization')
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(50.0, 6000.0)
    metasessions = get_metasessions_by_deviant_percent(book_id, document_id, user_id)
    chapters_lens = get_chapter_percents(book_id, document_id)
    prev_len = 0
    ticks_pos = []
    for chapter_len in chapters_lens:
        ticks_pos.append((chapter_len + prev_len) / 2)
        prev_len = chapter_len
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(BOOK_LABELS[book_id], rotation=90)
    for chapter_len in chapters_lens[:-1]:
        # plt.axvline(x=chapter_len, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=chapter_len, color='black', linestyle='--', linewidth=0.5)
    breaks = get_metasessions_breaks(book_id, document_id, user_id)
    logging.info('Found {} timebreaks'.format(len(breaks)))
    for time_break in breaks:
        # plt.axvline(x=time_break['book_from'], color='red', linestyle='--', linewidth=0.4)
        ax.axvline(x=time_break['book_from'], color='red', linestyle='--', linewidth=0.4)
    for metasession in metasessions:
        max_speed = max([session['speed'] for session in metasession])
        if max_speed > 6000:
            continue
        avg_speed = sum([session['speed'] for session in metasession]) / len(metasession)
        x = [session['book_from'] for session in metasession]
        y = [avg_speed for _ in metasession]
        # plt.plot(x, y, ReadingStyle.get_reading_style_by_speed(metasession[0]['speed']).get_color(), markersize=2)
        ax.plot(x, y, markersize=2)
    plt.savefig(os.path.join('resources', 'plots', 'deviant_percent', '{}_{}_{}_metasessions_by_deviant_percent.png')
                .format(book_id, document_id, user_id))


def save_metasessions(book_id, document_id, user_id):
    user_sessions = load_user_sessions(book_id, document_id, user_id)
    logging.info('Found {} sessions for document_id {} and user_id {}'.format(len(user_sessions), document_id, user_id))
    user_sessions.sort(key=lambda session: date_from_timestamp(session['read_at']))
    if len(user_sessions) == 0:
        return
    metasessions = [[(user_sessions[0], 0)]]
    for ind, session in enumerate(user_sessions[1:], 1):
        previous_session = user_sessions[ind - 1]
        time = (date_from_timestamp(session['read_at']) - date_from_timestamp(previous_session['read_at'])) \
            .total_seconds()
        if time < 0.01:
            speed = 3000  # TODO inf
        else:
            speed = previous_session['size'] * 60 / time
        if speed < 100:  # TODO threshold
            metasessions.append([(session, speed)])
        else:
            metasessions[-1].append((session, speed))
    logging.info('Total found {} metasessions'.format(len(metasessions)))
    metasessions_path = os.path.join('resources', 'metasessions', '{}_{}_{}.pkl'.format(book_id, document_id, user_id))
    with open(metasessions_path, 'wb') as file:
        pickle.dump(metasessions, file)
    logging.info('Metasessions of document {} for user {} saved to {}'.format(document_id, user_id, metasessions_path))


def get_metassesions(book_id, document_id, user_id):
    metasessions_path = os.path.join('resources', 'metasessions', '{}_{}_{}.pkl'.format(book_id, document_id, user_id))
    if not os.path.isfile(metasessions_path):
        logging.error('Unable to find metasessions of document {} for user {}'.format(document_id, user_id))
    with open(metasessions_path, 'rb') as file:
        return pickle.load(file)


def visualize_metassesions(book_id, document_id, user_id):
    plt.clf()
    plt.xlabel('Book percent')
    plt.ylabel('Session speed')
    plt.title('Metasessions visualization')
    plt.ylim(50.0, 5000.0)
    plt.xlim(0.0, 100.0)
    metassesions = get_metassesions(book_id, document_id, user_id)
    for metassesion in metassesions:
        x = [session['book_from'] for session, _ in metassesion]
        y = [speed for _, speed in metassesion]
        plt.plot(x, y)
    plt.savefig(os.path.join('resources', 'plots', '{}_{}_{}_metasessions.png').format(book_id, document_id, user_id))


def visualize_user_speed_spectrum(book_id, document_id, user_id, batches_amount):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Speed color')
    ax.set_title('Sessions batches')
    ax.set_xlim(0.0, 100.0)

    unique_sessions = get_user_sessions_by_place_in_book(book_id, document_id, user_id)

    batch_percent = 100.0 / batches_amount
    batches_speed = [None for _ in range(batches_amount)]
    batch_to = batch_percent
    batch_ind = 0
    current_speeds = []

    ax.set_ylim(0.0, batch_percent)

    places = list(unique_sessions.keys())
    places.sort(key=lambda val: val[0])
    for place in places:
        session = unique_sessions[place]
        book_from, book_to = place
        while book_from > batch_to:
            if len(current_speeds) > 0:
                batches_speed[batch_ind] = sum(current_speeds) / len(current_speeds)
                current_speeds = []
            batch_to, batch_ind = batch_to + batch_percent, batch_ind + 1
        if session['speed'] != INFINITE_SPEED and session['speed'] != UNKNOWN_SPEED:
            current_speeds.append(session['speed'])
    if len(current_speeds) > 0:
        batches_speed[batch_ind] = sum(current_speeds) / len(current_speeds)

    not_null_speeds = [speed for speed in batches_speed if speed is not None]
    max_speed = max(not_null_speeds)
    min_speed = min(not_null_speeds)
    rgb_sum = lambda first, second, weight: (first[0] * (1 - weight) + second[0] * weight,
                                             first[1] * (1 - weight) + second[1] * weight,
                                             first[2] * (1 - weight) + second[2] * weight)
    get_speed_ratio = lambda speed: (speed - min_speed) / (max_speed - min_speed)
    red_color = (1, 0, 0)
    blue_color = (0, 0, 1)
    batch_from = batch_percent / 2
    for ind in range(batches_amount):
        if batches_speed[ind] is None:
            circle = plt.Circle((batch_from, batch_percent / 2), batch_percent / 2, color='white')
            # ax.plot(batch_from, 5, markersize=20, color='w')
        else:
            circle = plt.Circle((batch_from, batch_percent / 2), batch_percent / 2, color=rgb_sum(blue_color, red_color,
                                                                                                  get_speed_ratio(
                                                                                                      batches_speed[
                                                                                                          ind])))
            # ax.plot(batch_from, 5, markersize=20, color=rgb_sum(blue_color, red_color,
            #                                                     get_speed_ratio(batches_speed[ind])))
        ax.add_artist(circle)
        batch_from += batch_percent

    chapters_lens = get_chapter_percents(book_id, document_id)
    prev_len = 0
    ticks_pos = []
    for chapter_len in chapters_lens:
        ticks_pos.append((chapter_len + prev_len) / 2)
        prev_len = chapter_len
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(BOOK_LABELS[book_id], rotation=90)

    plot_path = os.path.join('resources', 'plots', 'batches_spectrum', '{}_{}_{}.png'.format(book_id,
                                                                                             document_id,
                                                                                             user_id))
    plt.savefig(plot_path)


def visualize_users_speed_spectrum(book_id, user_ids, batches_amount, book_name=None, absolute_colors=False,
                                   sort_by_colors=False, smooth_limits=False, relative_speed=False,
                                   plot_name_suffix=None):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Speed color')
    ax.set_title('Sessions batches for book \'{}\''.format(book_name if book_name is not None else book_id))
    ax.set_xlim(0.0, 100.0)
    batch_percent = 100.0 / batches_amount
    ax.set_ylim(batch_percent * len(user_ids) + batch_percent / 2)
    batches_speeds = []

    for user_id in tqdm(user_ids):
        batches_speed = [None for _ in range(batches_amount)]
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
            while book_from > batch_to:
                if len(current_speeds) > 0:
                    batches_speed[batch_ind] = sum(current_speeds) / len(current_speeds)
                    current_speeds = []
                batch_to, batch_ind = batch_to + batch_percent, batch_ind + 1
            if session['speed'] != INFINITE_SPEED and session['speed'] != UNKNOWN_SPEED:
                current_speeds.append(session['speed'])
        if len(current_speeds) > 0:
            batches_speed[batch_ind] = sum(current_speeds) / len(current_speeds)
        batches_speeds.append(batches_speed)

    avg_speed = None
    if relative_speed and absolute_colors:
        speeds = [speed for batches_speed in batches_speeds for speed in batches_speed if speed is not None]
        if not smooth_limits:
            avg_speed = 1.0 * sum(speeds) / len(speeds)
        else:
            percent = int(1.0 * len(speeds) / 100 + 1)
            speeds = sorted(speeds)
            avg_speed = 1.0 * (sum(speeds[percent:-percent]) +
                               percent * speeds[percent] + percent * speeds[-percent]) / len(speeds)

    if not smooth_limits:
        max_speed = max([max([speed for speed in batches_speed if speed is not None])
                         for batches_speed in batches_speeds])
        min_speed = min([min([speed for speed in batches_speed if speed is not None])
                         for batches_speed in batches_speeds])
    else:
        speeds = sorted([speed for batches_speed in batches_speeds for speed in batches_speed if speed is not None
                         and not math.isnan(speed)])
        percent = int(1.0 * len(speeds) / 100 + 1)
        max_speed = speeds[-percent]
        min_speed = speeds[percent]

    print(max_speed, min_speed, percent)

    rgb_sum = lambda first, second, weight: (first[0] * (1 - weight) + second[0] * weight,
                                             first[1] * (1 - weight) + second[1] * weight,
                                             first[2] * (1 - weight) + second[2] * weight)
    get_speed_ratio = lambda speed: (speed - min_speed) / (max_speed - min_speed) if max_speed - min_speed > 0.0 \
        else 0.0
    red_color = (1, 0, 0)
    blue_color = (0, 0, 1)

    sorted_batches = sorted_batches_speeds(batches_speeds, sort_by_colors) if not relative_speed \
        else sorted_batches_speeds(batches_speeds, sort_by_colors, relative_speed=True, avg_speed=avg_speed)
    for user_ind, batches_speed in tqdm(enumerate(sorted_batches)):
        not_null_speeds = [speed for speed in batches_speed if speed is not None and not math.isnan(speed)]
        if not absolute_colors:
            if smooth_limits:
                percent = int(1.0 * len(not_null_speeds) / 100 + 1)
                not_null_speeds = sorted(not_null_speeds)
                max_speed = not_null_speeds[-percent]
                min_speed = not_null_speeds[percent]
            else:
                max_speed = max(not_null_speeds)
                min_speed = min(not_null_speeds)
            if relative_speed:
                if not smooth_limits:
                    avg_speed = 1.0 * sum(not_null_speeds) / len(not_null_speeds)
                else:
                    percent = int(1.0 * len(not_null_speeds) / 100 + 1)
                    not_null_speeds = sorted(not_null_speeds)
                    avg_speed = (sum(not_null_speeds[percent:-percent]) + percent * not_null_speeds[percent]
                                 + percent * not_null_speeds[-percent]) / len(not_null_speeds)
        batch_from = batch_percent / 2
        for ind in range(batches_amount):
            users_y = batch_percent * user_ind + batch_percent / 2
            if batches_speed[ind] is None:
                circle = plt.Circle((batch_from, users_y), batch_percent / 2, color='white')
            else:
                speed = min_speed if batches_speed[ind] < min_speed else max_speed if batches_speed[ind] > max_speed \
                    else batches_speed[ind]
                if relative_speed:
                    if speed > avg_speed:
                        percent = (speed - avg_speed) / (max_speed - avg_speed) if max_speed > avg_speed else 1
                        color = (0.5 * percent,
                                 0,
                                 0.5 * percent + 1.0 * (1 - percent))
                    else:
                        percent = (speed - min_speed) / (avg_speed - min_speed) if avg_speed > min_speed else 0
                        color = (0.5 * (1 - percent) + 1.0 * percent,
                                 0,
                                 0.5 * (1 - percent))
                    # logging.info('Percent : {}, color: {}, avg speed: {}, min speed: {}, max speed: {}'
                    #              .format(percent, color, avg_speed, min_speed, max_speed))

                    circle = plt.Circle((batch_from, users_y), batch_percent / 2,
                                        color=color)
                else:
                    circle = plt.Circle((batch_from, users_y), batch_percent / 2,
                                        color=rgb_sum(blue_color, red_color, get_speed_ratio(speed)))
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

    if not absolute_colors:
        if sort_by_colors:
            plot_path = os.path.join('resources', 'plots', 'users_batches_spectrum',
                                     '{}_{}_sorted_by_colors'.format(book_id, batches_amount))
        else:
            plot_path = os.path.join('resources', 'plots', 'users_batches_spectrum', '{}_{}'.format(book_id,
                                                                                                        batches_amount))
    else:
        plot_path = os.path.join('resources', 'plots', 'users_batches_spectrum', '{}_{}_absolute_colors'
                                 .format(book_id, batches_amount))

    plot_path += '_relative_speeds' if relative_speed else ''
    plot_path += '_smoothed_limits' if smooth_limits else ''
    plot_path += plot_name_suffix if plot_name_suffix is not None else ''
    plot_path += '.png'

    logging.info('Saving plot {}'.format(plot_path))
    plt.savefig(plot_path)


def sorted_batches_speeds(batches_speeds, by_colors=False, smooth_limits=False, relative_speed=False,
                          avg_speed=None):
    logging.info('Batches sorting started')
    users_count = len(batches_speeds)

    if by_colors:
        rgb_sum = lambda first, second, weight: (first[0] * (1 - weight) + second[0] * weight,
                                                 first[1] * (1 - weight) + second[1] * weight,
                                                 first[2] * (1 - weight) + second[2] * weight)
        get_speed_ratio = lambda speed: (speed - min_speed) / (max_speed - min_speed)
        red_color = (1, 0, 0)
        blue_color = (0, 0, 1)
        sessions_colors = []
        for user_ind, batches_speed in enumerate(batches_speeds):
            not_null_speeds = [speed for speed in batches_speed if speed is not None]
            if smooth_limits:
                percent = int(1.0 * len(not_null_speeds) / 100)
                not_null_speeds = sorted(not_null_speeds)
                max_speed = not_null_speeds[-percent]
                min_speed = not_null_speeds[percent]
            else:
                max_speed = max(not_null_speeds)
                min_speed = min(not_null_speeds)
            session_colors = []
            for ind in range(batches_amount):
                speed = None if batches_speed[ind] is None \
                    else min_speed if batches_speed[ind] < min_speed else max_speed if batches_speed[ind] > max_speed \
                    else batches_speed[ind]
                session_color = rgb_sum(blue_color, red_color, get_speed_ratio(speed)) \
                    if batches_speed[ind] is not None else (0, 0, 0)
                if relative_speed and avg_speed is None:
                    avg_speed = 1.0 * sum(not_null_speeds) / len(not_null_speeds)
                if relative_speed and batches_speed[ind] is not None:
                    if speed > avg_speed:
                        percent = (speed - avg_speed) / (max_speed - avg_speed) if max_speed > avg_speed else 0
                        session_color = (0.5 * percent,
                                         0,
                                         0.5 * percent + 1.0 * (1 - percent))
                    else:
                        percent = (speed - min_speed) / (avg_speed - min_speed) if avg_speed > min_speed else 0
                        session_color = (0.5 * (1 - percent) + 1.0 * percent,
                                         0,
                                         0.5 * (1 - percent))
                session_colors.append(session_color)
            sessions_colors.append(session_colors)

        triple_tuple_dist = lambda first, second: sum([(first[i] - second[i]) ** 2 for i in range(3)])
        first_ind = -1
        min_colors_sum = 255 ** 2 * 3 * len(sessions_colors[0])
        for ind, session_colors in enumerate(sessions_colors):
            first_sum = sum([triple_tuple_dist(color, (0, 0, 0)) for color in session_colors])
            if min_colors_sum < first_sum:
                min_colors_sum = first_sum
                first_ind = ind
        sorted_batches = [batches_speeds[first_ind]]
        sorted_colors = [sessions_colors[first_ind]]
        added = [False for _ in range(users_count)]
        added[first_ind] = True

        for _ in tqdm(range(users_count - 1)):
            min_dist = 255 ** 2 * 3 * len(sessions_colors[0]) * 1000000
            if relative_speed:
                min_dist = (max_speed / avg_speed) * 255 ** 2 * 3 * 100 * len(sessions_colors[0])
            next_ind = -1
            for second_ind, session_colors in enumerate(sessions_colors):
                if added[second_ind]:
                    continue
                current_dist = 0
                for first_color, second_color in zip(sorted_colors[-1], session_colors):
                    current_dist += triple_tuple_dist(first_color, second_color)

                if current_dist < min_dist:
                    min_dist = current_dist
                    next_ind = second_ind
            added[next_ind] = True
            sorted_batches.append(batches_speeds[next_ind])
            sorted_colors.append(sessions_colors[next_ind])

        return sorted_batches

    first_ind = -1
    min_batches = len(batches_speeds[0]) + 1
    for ind, batches_speed in enumerate(batches_speeds):
        not_null_count = len([speed for speed in batches_speed if speed is not None])
        if not_null_count < min_batches:
            min_batches = not_null_count
            first_ind = ind
    sorted_batches = [batches_speeds[first_ind]]
    added = [False for _ in range(users_count)]
    added[first_ind] = True

    if relative_speed and (avg_speed is None or math.isnan(avg_speed)):
        speeds = [speed for batches_speed in batches_speeds for speed in batches_speed if speed is not None
                  and not math.isnan(speed)]
        logging.info(len(speeds))
        if not smooth_limits:
            avg_speed = 1.0 * sum(speeds) / len(speeds)
        else:
            percent = int(1.0 * len(speeds) / 100)
            speeds = sorted(speeds)
            avg_speed = 1.0 * (sum(speeds[percent:-percent]) +
                               percent * speeds[percent] + percent * speeds[-percent]) / len(speeds)

    logging.info(avg_speed)

    for _ in tqdm(range(users_count - 1)):
        min_dist = (1.0 * INFINITE_SPEED) ** 2 * len(sorted_batches[-1]) * users_count * 10000
        next_ind = -1
        for second_ind, batches_speed in enumerate(batches_speeds):
            if added[second_ind]:
                continue
            current_dist = 0
            for first_speed, second_speed in zip(sorted_batches[-1], batches_speeds[second_ind]):
                if first_speed is not None and second_speed is not None:
                    if relative_speed:
                        first_speed /= avg_speed
                        second_speed /= avg_speed
                    current_dist += (first_speed - second_speed) ** 2
                elif first_speed is None and second_speed is None:
                    continue
                if relative_speed:
                    current_dist += INFINITE_SPEED ** 2
                else:
                    current_dist += (1.0 * INFINITE_SPEED / avg_speed) ** 2

            # logging.info('{} {}'.format(current_dist, min_dist))
            if current_dist < min_dist:
                min_dist = current_dist
                next_ind = second_ind
        added[next_ind] = True
        sorted_batches.append(batches_speeds[next_ind])

    return sorted_batches


def visualize_users_speed_classes(book_id, user_ids, batches_amount, book_name=None, suffix=''):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlabel('Book percent')
    ax.set_ylabel('Speed color')
    ax.set_title('Sessions batches for book \'{}\''.format(book_name if book_name is not None else book_id))
    ax.set_xlim(0.0, 100.0)
    batch_percent = 100.0 / batches_amount
    ax.set_ylim(batch_percent * len(user_ids) + batch_percent / 2)
    batches_speeds = []

    for user_id in tqdm(user_ids):
        batches_speed = [None for _ in range(batches_amount)]
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
            while book_from > batch_to:
                if len(current_speeds) > 0:
                    batches_speed[batch_ind] = sum(current_speeds) / len(current_speeds)
                    current_speeds = []
                batch_to, batch_ind = batch_to + batch_percent, batch_ind + 1
            if session['speed'] != INFINITE_SPEED and session['speed'] != UNKNOWN_SPEED:
                current_speeds.append(session['speed'])
        if len(current_speeds) > 0:
            batches_speed[batch_ind] = sum(current_speeds) / len(current_speeds)
        batches_speeds.append(batches_speed)

    colors = {ReadingStyle.SCANNING: (1, 0, 0),
              ReadingStyle.SKIMMING: (2.0 / 3, 0, 1.0 / 3),
              ReadingStyle.NORMAL: (1.0 / 3, 0, 2.0 / 3),
              ReadingStyle.DETAILED: (0, 0, 1)}

    for user_ind, batches_speed in tqdm(enumerate(sorted_batches_by_class(batches_speeds, colors))):
        batch_from = batch_percent / 2
        for ind in range(batches_amount):
            users_y = batch_percent * user_ind + batch_percent / 2
            if batches_speed[ind] is None:
                circle = plt.Circle((batch_from, users_y), batch_percent / 2, color='white')
            else:
                circle = plt.Circle((batch_from, users_y), batch_percent / 2,
                                     color=colors[ReadingStyle.get_reading_style_by_speed(batches_speed[ind])])
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

    plot_path = os.path.join('resources', 'plots', 'users_batches_speed_class', '{}_{}_sorted_{}.png'
                             .format(book_id, batches_amount, suffix))
    plt.savefig(plot_path)


def sorted_batches_by_class(batches_speeds, colors):
    users_amount = len(batches_speeds)
    first_ind = -1
    min_batches = len(batches_speeds[0]) + 1
    for ind, batches_speed in enumerate(batches_speeds):
        not_null_count = len([speed for speed in batches_speed if speed is not None])
        if not_null_count < min_batches:
            min_batches = not_null_count
            first_ind = ind
    sorted_batches = [batches_speeds[first_ind]]
    added = [False for _ in range(users_amount)]
    added[first_ind] = True

    color_dist = lambda first, second: sum([(first[i] - second[i]) ** 2 for i in range(3)])
    get_color = lambda speed: colors[ReadingStyle.get_reading_style_by_speed(speed)]
    for _ in tqdm(range(users_amount - 1)):
        min_dist = 10 * len(sorted_batches[-1])
        next_ind = -1
        for second_ind, batches_speed in enumerate(batches_speeds):
            if added[second_ind]:
                continue
            current_dist = 0
            for first_speed, second_speed in zip(sorted_batches[-1], batches_speeds[second_ind]):
                if first_speed is not None and second_speed is not None:
                    current_dist += color_dist(get_color(first_speed), get_color(second_speed))
                elif first_speed is None and second_speed is None:
                    continue
                current_dist += 3

            if current_dist < min_dist:
                min_dist = current_dist
                next_ind = second_ind
        added[next_ind] = True
        sorted_batches.append(batches_speeds[next_ind])

    return sorted_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_sessions', help='Save BOOKS sessions', action='store_true')
    parser.add_argument('--save_users', help='Save BOOKS users', action='store_true')
    parser.add_argument('--save_users_sessions', help='Save sessions for users of BOOKS', action='store_true')
    parser.add_argument('--calculate_session_percents', help='Calculate book_from and book_to', action='store_true')
    parser.add_argument('--save_common_users', help='Save common users of BOOKS', action='store_true')
    parser.add_argument('--process_items', help='process DOCUMENTS items', action='store_true')
    parser.add_argument('--save_metasessions', help='Save BOOKS metasessions', action='store_true')
    parser.add_argument('--find_the_best_users', help='Find the best users', action='store_true')
    parser.add_argument('--visualize_metasessions_by_style', help='Visualize metasessions by reading style',
                        action='store_true')
    parser.add_argument('--visualize_metasessions_by_deviant_percent', help='Visualize metasessions by reading style',
                        action='store_true')
    parser.add_argument('--prepare_users', help='Find users which red a lot of books',
                        action='store_true')
    parser.add_argument('--save_users_csv', help='Save selected good users as csv',
                        action='store_true')
    parser.add_argument('--select_users', help='Select users with the most amount of sessions',
                        action='store_true')
    parser.add_argument('--save_speed', help='Save speed for users sessions',
                        action='store_true')
    parser.add_argument('--pass', help='pass',
                        action='store_true')
    parser.add_argument('--visualize_user_speed_spectrum',
                        help='Visualize sessions by the place in the book with N batches for one user',
                        type=int, metavar='N')
    parser.add_argument('--visualize_users_speed_spectrum',
                        help='Visualize sessions by the place in the book with N batches for several users',
                        type=int, metavar='N')
    parser.add_argument('--visualize_users_speed_classes', help='Visualize speed classes for N batches',
                        type=int, metavar='N')

    args = parser.parse_args()
    if args.save_sessions:
        save_sessions(BOOKS.values())
    if args.save_users:
        save_users(BOOKS.values())
    if args.save_users_sessions:
        save_books_users_sessions(BOOKS.values())
    if args.save_common_users:
        save_common_users(list(BOOKS.values()))
    if args.calculate_session_percents:
        for book_id in BOOKS.values():
            calculate_session_percents(book_id, DOCUMENTS[book_id])
    if args.find_the_best_users:
        for book_id in BOOKS.values():
            users = load_users(book_id)[100:]
            for user_id, _ in zip(users, tqdm(range(len(users)))):
                user_document_id = get_user_document_id(book_id, user_id)
                if user_document_id not in DOCUMENTS[book_id]:
                    continue
                if user_id is '':
                    continue
                user_sessions = load_user_sessions(book_id, user_document_id, user_id)
                if len(user_sessions) < 500:
                    continue
                print(book_id, user_document_id, user_id, len(user_sessions))
    if args.process_items:
        for book_id in BOOKS.values():
            for document_id in DOCUMENTS[book_id]:
                print('Book {}, document {}'.format(book_id, document_id))
                chapters = load_chapters(book_id, document_id)
                text = load_text(book_id, document_id)
                items = get_items(document_id)
                print('Items:')
                for item in items:
                    print('{},{},{},{}'.format(book_id, document_id, item['position'], item['_to'] - item['_from']))
                print('Chapters')
                for ind, chapter in enumerate(chapters):
                    print('{},{}'.format(ind + 1, 100.0 * len(chapter) / len(text)))
    if args.visualize_metasessions_by_style:
        for book_id in BOOKS.values():
            user_ids = get_good_users_info(book_id).keys()
            logging.info('Found {} users for book {}'.format(len(user_ids), book_id))
            for user_id in tqdm(user_ids):
                logging.info('Visualizing metasessions by speed of user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                if document_id not in DOCUMENTS[book_id]:
                    continue
                visualize_metasessions_by_reading_style(book_id, document_id, user_id)
    if args.visualize_metasessions_by_deviant_percent:
        for book_id in BOOKS.values():
            user_ids = get_good_users_info(book_id).keys()
            logging.info('Found {} users for book {}'.format(len(user_ids), book_id))
            for user_id in tqdm(user_ids):
                logging.info('Visualizing metasessions by deviant percent of user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                if document_id not in DOCUMENTS[book_id]:
                    continue
                visualize_metasessions_by_deviant_percent(book_id, document_id, user_id)
    if args.save_speed:
        for book_id in BOOKS.values():
            user_ids = get_good_users_info(book_id).keys()
            logging.info('Found {} users for book {}'.format(len(user_ids), book_id))
            for user_id in tqdm(user_ids):
                logging.info('Saving speed for sessions of user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                save_user_sessions_speed(book_id, document_id, user_id)
    if args.save_metasessions:
        for book_id in BOOKS.values():
            for user_id in tqdm(get_user_selection(book_id)):
                logging.info('Saving metasessions for user {}'.format(user_id))
                document_id = get_user_document_id(book_id, user_id)
                save_metasessions_by_reading_style(book_id, document_id, user_id)
                save_metasessions_by_deviant_percent(book_id, document_id, user_id)
    if args.prepare_users:
        good_users = {}
        for book_id in BOOKS.values():
            book_users = load_users(book_id)
            good_users[book_id] = []
            docs = set(DOCUMENTS[book_id])
            for user_id in tqdm(book_users):
                if user_id == '':
                    continue
                document_id = get_user_document_id(book_id, user_id)
                if document_id not in docs:
                    continue
                read_percent = get_book_percent(book_id, document_id, user_id)
                logging.info('User {} has red {} percents of book {}'.format(user_id, read_percent, book_id))
                if read_percent < 80.0:
                    continue
                books_amount = get_users_books_amount(user_id)
                if books_amount >= 3:
                    good_users[book_id].append((user_id, books_amount, read_percent))
                    logging.info('Found good user {}, who has red {} books'.format(user_id, books_amount))
            output_path = os.path.join('resources', 'users', '{}_good_users_id_amount_percent.pkl'.format(book_id))
            with open(output_path, 'wb') as file:
                pickle.dump(good_users[book_id], file)
    if args.save_users_csv:
        users_info = get_users_extra_information()
        for book_id in BOOKS.values():
            output_path = os.path.join('resources', 'users', 'csv', '{}.csv'.format(book_id))
            users = upload_good_users_with_percents(book_id)
            with open(output_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['book_id', 'user_id', 'books_amount',
                                                             'read_percent', 'gender', 'sub_level',
                                                             'birthday_at'])
                writer.writeheader()
                for user_id, amount, read_percent in tqdm(users):
                    document_id = get_user_document_id(book_id, user_id)
                    if str(user_id) in users_info:
                        user_info = users_info[str(user_id)]
                        gender = user_info['gender']
                        sub_level = user_info['sub_level']
                        birthday_at = user_info['birthday_at']
                    else:
                        gender = '?'
                        sub_level = '?'
                        birthday_at = '?'
                    user_description = {'book_id': book_id, 'user_id': user_id, 'books_amount': amount,
                                        'read_percent': read_percent, 'gender': gender, 'sub_level': sub_level,
                                        'birthday_at': birthday_at}
                    writer.writerow(user_description)
    if args.select_users:
        users_for_book = {}
        for book_id in BOOKS.values():
            users = upload_good_users(book_id)
            users_for_book[book_id] = []
            for user_id, amount in users:
                document_id = get_user_document_id(book_id, user_id)
                sessions_amount = len(load_user_sessions(book_id, document_id, user_id))
                users_for_book[book_id].append((user_id, sessions_amount))
            users_for_book[book_id].sort(key=lambda x: -x[1])
            output_path = os.path.join('resources', 'users', '{}_users_selection.pkl'.format(book_id))
            logging.info('Total found {} good users for book {}, we will select 100'
                         .format(len(users_for_book[book_id]), book_id))
            with open(output_path, 'wb') as file:
                pickle.dump([x[0] for x in users_for_book[book_id][:100]], file)
    if args.visualize_user_speed_spectrum is not None:
        batches_amount = args.visualize_user_speed_spectrum
        for book_id in BOOKS.values():
            for user_id in tqdm(get_user_selection(book_id)):
                document_id = get_user_document_id(book_id, user_id)
                visualize_user_speed_spectrum(book_id, document_id, user_id, batches_amount)
                break
            break
    if args.visualize_users_speed_spectrum is not None:
        amount = args.visualize_users_speed_spectrum
        for name, book_id in BOOKS.items():
            if amount == -1:
                batches_amounts = [20, 100, 200]
            else:
                batches_amounts = [amount]
            for batches_amount in batches_amounts:
                user_ids = get_good_users_info(book_id).keys()
                logging.info('Found {} users for book {}'.format(len(user_ids), book_id))
                visualize_users_speed_spectrum(book_id, user_ids, batches_amount, book_name=name, smooth_limits=True,
                                               relative_speed=True)
                visualize_users_speed_spectrum(book_id, user_ids, batches_amount, book_name=name, smooth_limits=True,
                                               sort_by_colors=True, relative_speed=True)
                visualize_users_speed_spectrum(book_id, user_ids, batches_amount, book_name=name, absolute_colors=True,
                                               smooth_limits=True, relative_speed=True)
    if args.visualize_users_speed_classes is not None:
        amount = args.visualize_users_speed_classes
        for name, book_id in BOOKS.items():
            if amount == -1:
                batches_amounts = [50, 100, 200]
            else:
                batches_amounts = [amount]
            for batches_amount in batches_amounts:
                info = get_good_users_info(book_id)
                user_ids = info.keys()
                logging.info('Found {} users for book {}'.format(len(user_ids), book_id))
                visualize_users_speed_classes(book_id, user_ids, batches_amount, book_name=name)
                visualize_users_speed_classes(book_id, [user_id for user_id in user_ids
                                                        if info[user_id]['gender'] == 'f'],
                                              batches_amount, book_name=name, suffix='f')
                visualize_users_speed_classes(book_id, [user_id for user_id in user_ids
                                                        if info[user_id]['gender'] == 'm'],
                                              batches_amount, book_name=name, suffix='m')
