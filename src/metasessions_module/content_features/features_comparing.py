import os
import sys
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt

from src.metasessions_module.interest_marker.interest_marker_utils import smooth_marker, \
    load_normalized_interest_markers

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.interest_marker.integrated_interest_marker import load_integrated_markers
from src.metasessions_module.content_features.feautres_builder import TextFeaturesBuilder
from src.metasessions_module.utils import remove_underscores, join_with_underscores, min_max_scale


def compare_integrated_signals_with_features(book=None, features=None, features_names=None, integrated_signals=None,
                                             smoothed_only=True):
    if book is None and (features is None or features_names is None or integrated_signals is None):
        logging.error('compare_integrated_signals_with_features: provide either book or all remaining arguments')
    if features is None:
        features = TextFeaturesBuilder.load_features()
    if features_names is None:
        features_names = TextFeaturesBuilder.STR_FEATURES_NAMES
    if integrated_signals is None:
        integrated_signals = load_integrated_markers(book)
    if smoothed_only:
        integrated_signals = {k: v for k, v in integrated_signals.items() if 'smoothed' in k}

    output_path = os.path.join('resources', 'plots', str(book), 'comparing')
    os.makedirs(output_path, exist_ok=True)
    fragments = np.arange(features.shape[0])

    for signal_name, signal in integrated_signals.items():
        for feature_ind, feature_name in enumerate(features_names):
            plt.clf()
            plt.plot(fragments, min_max_scale(signal), label=remove_underscores(signal_name))
            plt.plot(fragments, min_max_scale(smooth_marker(features[:, feature_ind])),
                     label=f'smoothed {remove_underscores(feature_name)}')
            plt.xlabel('Fragment number')
            plt.ylabel('Value')
            plt.legend()
            cur_output_path = os.path.join(output_path, signal_name)
            os.makedirs(cur_output_path, exist_ok=True)
            plt.savefig(os.path.join(cur_output_path, join_with_underscores(signal_name, feature_name)))


def compare_signals_with_features(book=None, features=None, features_names=None, signals=None,
                                  smoothed_only=False):
    if book is None and (features is None or features_names is None or signals is None):
        logging.error('compare_signals_with_features: provide either book or all remaining arguments')
    if features is None:
        features = TextFeaturesBuilder.load_features()
    if features_names is None:
        features_names = TextFeaturesBuilder.STR_FEATURES_NAMES
    if signals is None:
        signals = load_normalized_interest_markers(book)
    if smoothed_only:
        signals = {k: v for k, v in signals.items() if 'smoothed' in k}

    output_path = os.path.join('resources', 'plots', str(book), 'comparing')
    os.makedirs(output_path, exist_ok=True)
    fragments = np.arange(features.shape[0])

    for signal_name, signal in signals.items():
        for feature_ind, feature_name in enumerate(features_names):
            plt.clf()
            plt.plot(fragments, min_max_scale(smooth_marker(signal)),
                     label=f'smoothed{remove_underscores(signal_name)}')
            plt.plot(fragments, min_max_scale(smooth_marker(features[:, feature_ind])),
                     label=f'smoothed {remove_underscores(feature_name)}')
            plt.xlabel('Fragment number')
            plt.ylabel('Value')
            plt.legend()
            cur_output_path = os.path.join(output_path, signal_name)
            os.makedirs(cur_output_path, exist_ok=True)
            plt.savefig(os.path.join(cur_output_path, join_with_underscores(signal_name, feature_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    compare_integrated_signals_with_features(book_id)
    # compare_signals_with_features(book_id)
