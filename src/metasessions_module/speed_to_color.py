import numpy as np
import logging


from src.metasessions_module.config import *


def is_known_speed(speed):
    return not np.isclose(speed, UNKNOWN_SPEED)


def to_matplotlib_color(color):
    return tuple(color)


def get_red_to_blue_color_by_ratio(ratio):
    ratio = 0 if ratio < 0 else 1 if ratio > 1 else ratio
    return np.array([ratio, 0, 1 - ratio])


def get_colors_speed_using_absolute_min_max_scale(speeds, min_speed=None, max_speed=None):
    is_known = np.vectorize(is_known_speed)
    known_speeds = speeds[np.where(is_known(speeds))]
    max_speed = np.max(known_speeds) if max_speed is None else max_speed
    min_speed = np.min(known_speeds) if min_speed is None else min_speed

    if np.isclose(max_speed - min_speed, 0):
        logging.info('Max speed is close to min speed, colors calculating canceled')
        return np.zeros(*speeds.shape)

    colors = np.full((*speeds.shape, 3), 1, dtype=np.float64)
    for user_ind in range(speeds.shape[0]):
        for speed_ind in range(speeds.shape[1]):
            speed = speeds[user_ind, speed_ind]
            if is_known_speed(speed):
                colors[user_ind, speed_ind] = get_red_to_blue_color_by_ratio((speed - min_speed) /
                                                                             (max_speed - min_speed))

    return colors


def get_colors_speed_using_users_min_max_scale(speeds, min_speed=None, max_speed=None):
    colors = np.full((*speeds.shape, 3), 1, dtype=np.float64)
    for user_ind in range(speeds.shape[0]):
        current_speeds = speeds[user_ind]
        is_known = np.vectorize(is_known_speed)
        known_speeds = current_speeds[np.where(is_known(current_speeds))]
        cur_max_speed = np.max(known_speeds) if max_speed is None else max_speed
        cur_min_speed = np.min(known_speeds) if min_speed is None else min_speed

        if np.isclose(cur_max_speed, cur_min_speed):
            logging.info('Max speed is close to min speed for user #{}, color set to blue')
            for speed_ind in range(speeds.shape[1]):
                if is_known_speed(speeds[user_ind][speed_ind]):
                    colors[user_ind][speed_ind][1] = 0
                    colors[user_ind][speed_ind][2] = 0
            continue

        for speed_ind in range(speeds.shape[1]):
            speed = speeds[user_ind, speed_ind]
            if is_known_speed(speed):
                ratio = 1.0 * (speed - cur_min_speed) / (cur_max_speed - cur_min_speed)
                colors[user_ind, speed_ind] = get_red_to_blue_color_by_ratio(ratio)
                # print(speed, cur_min_speed, cur_max_speed, ratio, colors[user_ind, speed_ind])

    return colors
