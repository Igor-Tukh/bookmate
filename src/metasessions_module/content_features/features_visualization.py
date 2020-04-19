import os
import numpy as np
import matplotlib.pyplot as plt

from src.metasessions_module.content_features.feautres_builder import TextFeaturesBuilder
from scipy.interpolate import make_interp_spline


def plot_feature(feature, path, feature_name='', smooth=False):
    plt.clf()
    x = np.arange(feature.shape[0])
    smooth_x = x[::5]
    if smooth:
        plt.plot(smooth_x, make_interp_spline(x, feature, k=3)(smooth_x))
    else:
        plt.plot(x, feature)
    plt.title(f'Feature \'{feature_name}\' plot')
    plt.xlabel('Fragment number')
    plt.ylabel(feature_name)
    plt.savefig(path)


if __name__ == '__main__':
    builder = TextFeaturesBuilder(os.path.join('resources', 'texts'))
    plots_path = os.path.join('resources', 'plots', 'features')
    # for pos in ['noun', 'verb', 'adjective']:
    #     features = builder.get_pos_percent(pos)
    #     plot_feature(features, os.path.join(plots_path, f'{pos}_percent_smooth.png'), f'The proportion of {pos}s',
    #                  smooth=True)
    # plot_feature(builder.get_bodyparts_percent(), os.path.join(plots_path, 'bodyparts_smooth.png'),
    #              'The proportion of bodyparts', smooth=True)
    # plot_feature(builder.get_bodyparts_percent(), os.path.join(plots_path, 'bodyparts.png'),
    #              'The proportion of bodyparts', smooth=False)
    # plot_feature(builder.get_characters_names_percent(), os.path.join(plots_path, 'characters_smooth.png'),
    #              'The proportion of characters', smooth=True)
    # plot_feature(builder.get_main_characters_names_percent(), os.path.join(plots_path, 'main_characters_smooth.png'),
    #              'The proportion of main characters', smooth=True)
    # plot_feature(builder.get_emotional_verbs_percent(), os.path.join(plots_path, 'emotional_verbs_smooth.png'),
    #              'The proportion of emotional verbs', smooth=True)
    # plot_feature(builder.get_sentiment(), os.path.join(plots_path, 'sentiment_smooth.png'),
    #              'Smoothed average sentiment', smooth=True)
    # plot_feature(builder.get_average_word_len(), os.path.join(plots_path, 'average_len.png'),
    #              'Average word length')
    # plot_feature(builder.get_average_word_len(), os.path.join(plots_path, 'average_len_smooth.png'),
    #              'Smoothed average word length', smooth=True)
    # plot_feature(builder.get_personal_pronouns_percent(), os.path.join(plots_path, 'personal_pronouns.png'),
    #              'The proportion of personal pronouns')
    # plot_feature(builder.get_personal_pronouns_percent(), os.path.join(plots_path, 'personal_pronouns_smooth.png'),
    #              'The proportion of personal pronouns', smooth=True)
    plot_feature(builder.get_average_dependency_tree_depth(), os.path.join(plots_path, 'average_dependency_tree.png'),
                 'Average dependency depth tree depth')
    plot_feature(builder.get_personal_pronouns_percent(),
                 os.path.join(plots_path, 'average_dependency_tree_smooth.png'),
                 'Average dependency depth tree depth', smooth=True)

