import pickle
import numpy as np
import pandas as pd
from functools import lru_cache

history = None
last_bandit = None
last_reward_total = None
second_last_opposite = None
third_last_opposite = None

# paste model binary here
MODEL_B = b''

model = pickle.loads(MODEL_B)


def repeat_features(x):
    temp = ''.join((np.array(x) != -1).astype(int).astype(str)).split('0')
    return len(temp[0]), len(temp[-1])


def repeat_get_reward_features(x):
    temp = ''.join((np.array(x) == 1).astype(int).astype(str)).split('0')
    return len(temp[0]), len(temp[-1])


def repeat_non_reward_features(x):
    temp = ''.join((np.array(x) == 0).astype(int).astype(str)).split('0')
    return len(temp[0]), len(temp[-1])


def repeat_features_opponent(x):
    temp = ''.join((np.array(x) == -1).astype(int).astype(str)).split('0')
    return len(temp[0]), len(temp[-1])


@lru_cache(maxsize=2048)
def calc_empirical_win_rate_cache(x):

    if len(x) == 0:
        return 49.5, 9.1, 19.2, 29.3, 39.4, 49.5, 59.6, 69.7, 79.8, 89.9

    arr_x = np.array(x)

    win_table = np.tile(np.arange(0, 1, 0.01), (len(arr_x), 1)).T * 0.97 ** np.arange(len(arr_x))
    lose_table = 1 - win_table

    result_table = np.concatenate([win_table[:, arr_x == 1], lose_table[:, arr_x == 0]], axis=1)
    log_likelihood = np.sum(np.log(result_table), axis=1)

    shift_ll = log_likelihood + np.min(log_likelihood[np.isfinite(log_likelihood)])

    weights = np.exp(shift_ll)

    weights[np.isinf(weights)] = 0.0
    weights = weights / np.sum(weights)
    weights_cumsum = np.cumsum(weights)

    quantile_10 = np.argmin(np.abs(weights_cumsum - 0.1)) * 0.97 ** len(x)
    quantile_20 = np.argmin(np.abs(weights_cumsum - 0.2)) * 0.97 ** len(x)
    quantile_30 = np.argmin(np.abs(weights_cumsum - 0.3)) * 0.97 ** len(x)
    quantile_40 = np.argmin(np.abs(weights_cumsum - 0.4)) * 0.97 ** len(x)
    quantile_50 = np.argmin(np.abs(weights_cumsum - 0.5)) * 0.97 ** len(x)
    quantile_60 = np.argmin(np.abs(weights_cumsum - 0.6)) * 0.97 ** len(x)
    quantile_70 = np.argmin(np.abs(weights_cumsum - 0.7)) * 0.97 ** len(x)
    quantile_80 = np.argmin(np.abs(weights_cumsum - 0.8)) * 0.97 ** len(x)
    quantile_90 = np.argmin(np.abs(weights_cumsum - 0.9)) * 0.97 ** len(x)

    maximum_posterior = np.argmin(-log_likelihood) * 0.97 ** len(x)

    return maximum_posterior, quantile_10, quantile_20, quantile_30, quantile_40, quantile_50, \
        quantile_60, quantile_70, quantile_80, quantile_90


def calc_empirical_win_rate(x):
    return calc_empirical_win_rate_cache(tuple(x))


def feature_extraction(history_df):

    history_df['num_chosen'] = history_df['history'].map(len)
    history_df['num_chosen_mine'] = history_df['history'].map(lambda x: np.sum(np.array(x) != -1))
    history_df['num_chosen_opponent'] = history_df['num_chosen'] - history_df['num_chosen_mine']
    history_df['num_get_reward'] = history_df['history'].map(lambda x: np.sum(np.array(x) == 1))
    history_df['num_non_reward'] = history_df['history'].map(lambda x: np.sum(np.array(x) == 0))

    history_df['rate_mine'] = history_df['num_chosen_mine'] / history_df['num_chosen']
    history_df['rate_opponent'] = history_df['num_chosen_opponent'] / history_df['num_chosen']
    history_df['rate_get_reward'] = history_df['num_get_reward'] / history_df['num_chosen_mine']

    history_df['empirical_win_rate'], history_df['quantile_10'], history_df['quantile_20'], history_df['quantile_30'], \
    history_df['quantile_40'], history_df['quantile_50'], history_df['quantile_60'], history_df['quantile_70'], \
    history_df['quantile_80'], history_df['quantile_90'] = zip(*history_df['history'].map(calc_empirical_win_rate))

    history_df['repeat_head'], history_df['repeat_tail'] = zip(
        *history_df['history'].map(repeat_features))
    history_df['repeat_get_reward_head'], history_df['repeat_get_reward_tail'] = zip(
        *history_df['history'].map(repeat_get_reward_features))
    history_df['repeat_non_reward_head'], history_df['repeat_non_reward_tail'] = zip(
        *history_df['history'].map(repeat_non_reward_features))
    history_df['opponent_repeat_head'], history_df['opponent_repeat_tail'] = zip(
        *history_df['history'].map(repeat_features_opponent))


def agent(observation, configuration):
    global history, last_bandit, last_reward_total, second_last_opposite, third_last_opposite

    if observation.step == 0:
        history = [[] for _ in range(configuration.banditCount)]
        last_opposite = None

    else:

        if observation.lastActions[0] == last_bandit:
            last_opposite = int(observation.lastActions[1])
        else:
            last_opposite = int(observation.lastActions[0])

        reward = observation.reward - last_reward_total

        history[last_bandit].append(int(reward))
        history[last_opposite].append(-1)

    history_df = pd.DataFrame()
    history_df['round'] = np.repeat(observation.step, configuration.banditCount)
    history_df['history'] = history

    last_opponent_chosen = np.zeros(100)
    if observation.step >= 1:
        last_opponent_chosen[last_opposite] = 1
    history_df['last_opponent_chosen'] = last_opponent_chosen

    second_last_opponent_chosen = np.zeros(100)
    if observation.step >= 2:
        second_last_opponent_chosen[second_last_opposite] = 1
    history_df['second_last_opponent_chosen'] = second_last_opponent_chosen

    third_last_opponent_chosen = np.zeros(100)
    if observation.step >= 3:
        third_last_opponent_chosen[third_last_opposite] = 1
    history_df['third_last_opponent_chosen'] = third_last_opponent_chosen

    history_df['opponent_repeat_twice'] = last_opponent_chosen * second_last_opponent_chosen
    history_df['opponent_repeat_three_times'] = history_df['opponent_repeat_twice'] * third_last_opponent_chosen

    feature_extraction(history_df)

    history_df = history_df[[
        'round', 'last_opponent_chosen', 'second_last_opponent_chosen', 'third_last_opponent_chosen',
        'opponent_repeat_twice', 'opponent_repeat_three_times',
        'num_chosen', 'num_chosen_mine', 'num_chosen_opponent', 'num_get_reward', 'num_non_reward',
        'rate_mine', 'rate_opponent', 'rate_get_reward',
        'empirical_win_rate', 'quantile_10', 'quantile_20', 'quantile_30', 'quantile_40',
        'quantile_50', 'quantile_60', 'quantile_70', 'quantile_80', 'quantile_90',
        'repeat_head', 'repeat_tail', 'repeat_get_reward_head', 'repeat_get_reward_tail',
        'repeat_non_reward_head', 'repeat_non_reward_tail', 'opponent_repeat_head', 'opponent_repeat_tail'
    ]]

    pred = model.predict(history_df)

    act = int(np.random.choice(np.flatnonzero(pred == pred.max())))

    third_last_opposite = second_last_opposite
    second_last_opposite = last_opposite
    last_bandit = act
    last_reward_total = observation.reward

    return act
