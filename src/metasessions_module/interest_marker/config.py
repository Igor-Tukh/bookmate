import os
import sys

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from enum import IntEnum, Enum


class InterestMarker(Enum):
    def __str__(self):
        return self.value

    QUIT_MARKER = 'quit_marker'
    UNUSUAL_HOURS_MARKER = 'unusual_hours_marker'
    RE_READING_MARKER = 're_reading_marker'
    READING_INTERRUPT_MARKER = 'reading_interrupt_marker'
    SPEED_MARKER = 'speed_marker'
    SCROLLING_MARKER = 'scrolling_marker'


class MarkerManifestation(IntEnum):
    NON_INTERESTING = -1
    NEUTRAL = 0
    INTERESTING = 1


MARKER_TITLE_TO_RUSSIAN = {
    'quit_marker': 'Сигнал бросаний чтения',
    'unusual_hours_marker': 'Сигнал чтений в необычное время',
    're_reading_marker': 'Сигнал повторных прочтений',
    'reading_interrupt_marker': 'Сигнал прерываний в процессе чтения',
    'speed_marker': 'Силгнал выской скорости',
    'scrolling_marker': 'Сигнал пролистываний'
}


def _map(int_list):
    return list(map(MarkerManifestation, int_list))


MANIFESTATIONS_NUMBER_TO_MARKER_MANIFESTATION = {
    InterestMarker.READING_INTERRUPT_MARKER: _map([1, 0, -1]),
    InterestMarker.RE_READING_MARKER: _map([-1, 0, 1]),
    InterestMarker.QUIT_MARKER: _map([1, 0, -1]),
    InterestMarker.UNUSUAL_HOURS_MARKER: _map([-1, 0, 1]),
    InterestMarker.SPEED_MARKER: _map([1, 0, -1])
}


def get_marker_plots_path(book_id, marker_description):
    dir_path = os.path.join('resources', 'plots', 'markers', str(book_id), marker_description)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_markers_path_for_book(book_id):
    dir_path = os.path.join('resources', 'all_markers', str(book_id))
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_stats_path():
    return os.path.join('resources', 'stats')


def get_marker_dumps_path(marker_description, book_id, user_id):
    dir_path = os.path.join(get_markers_path_for_book(book_id), marker_description)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f'{user_id}.pkl')


def get_book_stats_path(book_id, stats_filename):
    dir_path = os.path.join('resources', 'stats', str(book_id))
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, stats_filename)
