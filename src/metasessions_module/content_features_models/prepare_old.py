import logging

from src.metasessions_module.content_features.feautres_builder import TextFeaturesBuilder
from src.metasessions_module.interest_marker.interest_marker_utils import load_saved_markers, calculate_cumulative_markers
from sklearn.model_selection import train_test_split


def prepare_regression_markers(book_id, common=False):
    all_markers = load_saved_markers()

    if book_id not in all_markers.keys():
        logging.warning(f'No markers saved for book {book_id}')
        return None, None, None, None

    quit_markers = all_markers[book_id]['quit_markers']
    unusual_hours_markers = all_markers[book_id]['unusual_hours_markers']
    re_reading_markers = all_markers[book_id]['re_reading_markers']
    reading_interrupt_markers = all_markers[book_id]['reading_interrupt_markers']

    if common:
        return calculate_cumulative_markers(quit_markers), calculate_cumulative_markers(unusual_hours_markers), \
               calculate_cumulative_markers(re_reading_markers), calculate_cumulative_markers(reading_interrupt_markers)
    else:
        return quit_markers, unusual_hours_markers, re_reading_markers, reading_interrupt_markers


def prepare_features():
    return TextFeaturesBuilder.load_features()


def prepare_regression_data(book_id, shuffle=False):
    q, u, r, i = prepare_regression_markers(book_id, common=True)
    f = prepare_features()
    data = []
    for m, description in zip([q, u, r, i], ['Quit marker', 'Unusual reading hours', 'Re-Reading', 'Interrupts']):
        data.append((train_test_split(f, m, test_size=0.7, shuffle=shuffle), description))
    return data

