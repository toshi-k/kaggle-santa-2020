import copy
import json
import traceback
from pathlib import Path
from logging import getLogger
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from submit_main import feature_extraction
from log import init_logger


def preprocess(path):

    logger = getLogger('root')

    try:
        with open(path, 'r') as f:
            steps = json.load(f)['steps']

        rewards = np.array([step[0]['reward'] for step in steps])
        rewards = np.diff(np.append(0, rewards))

        history = [[] for _ in range(100)]

        list_history_df = []

        for r in range(0, 2000):

            if r >= 1:
                history[steps[r][0]['action']].append(rewards[r])
                history[steps[r][1]['action']].append(-1)

            new_df = pd.DataFrame()
            new_df['history'] = copy.deepcopy(history)
            new_df['round'] = r
            new_df['game_id'] = path.stem
            new_df['threshold'] = steps[r][0]['observation']['thresholds']

            last_opponent_chosen = np.zeros(100)
            if r >= 1:
                last_opponent_chosen[steps[r][1]['action']] = 1
            new_df['last_opponent_chosen'] = last_opponent_chosen

            second_last_opponent_chosen = np.zeros(100)
            if r >= 2:
                second_last_opponent_chosen[steps[r-1][1]['action']] = 1
            new_df['second_last_opponent_chosen'] = second_last_opponent_chosen

            third_last_opponent_chosen = np.zeros(100)
            if r >= 3:
                third_last_opponent_chosen[steps[r-2][1]['action']] = 1
            new_df['third_last_opponent_chosen'] = third_last_opponent_chosen

            new_df['opponent_repeat_twice'] = last_opponent_chosen * second_last_opponent_chosen
            new_df['opponent_repeat_three_times'] = new_df['opponent_repeat_twice'] * third_last_opponent_chosen

            list_history_df.append(new_df)

        history_df = pd.concat(list_history_df, axis=0)
        history_df = history_df.sample(frac=0.05)

        feature_extraction(history_df)

    except json.decoder.JSONDecodeError:
        logger.info(f'json decode error occurred in {path}')
        return None
    except:
        logger.info(f'unknown error occurred in {path}')
        logger.debug(traceback.format_exc())
        return None

    return history_df


def main():

    logger = init_logger('_log/preprocess.log')

    dir_data = Path('../../data')

    path_all = list(dir_data.glob("*/*.json"))

    logger.info(f'number of games: {len(path_all)}')

    # multi process
    with Pool(20) as p:
        list_dataset_all = list(tqdm(p.imap(preprocess, path_all), total=len(path_all)))

    # single process
    # list_dataset_all = list(tqdm(map(preprocess, path_all), total=len(path_all)))

    list_dataset_all = [d for d in list_dataset_all if d is not None]

    dataset_all = pd.concat(list_dataset_all, axis=0)

    logger.info(f'size of created dataset: {dataset_all.shape}')

    print(dataset_all[["last_opponent_chosen", "second_last_opponent_chosen", "third_last_opponent_chosen",
                       "opponent_repeat_twice", "opponent_repeat_three_times"]].sum(0))

    dataset_all.drop('history', axis=1, inplace=True)
    dataset_all.sort_values(['game_id', 'round'], inplace=True)
    dataset_all.to_csv('train.csv', index=False)


if __name__ == '__main__':
    main()
