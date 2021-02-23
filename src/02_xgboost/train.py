import random
import pickle
from logging import getLogger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import DMatrix

from log import init_logger

NUM_ROUND = 1500
LEARNING_RATE = 0.10
NUM_LEAVES = 8
MAX_DEPTH = 6
LAMBDA_L1 = 0.1
LAMBDA_L2 = 0.1
BAGGING_FRACTION = 0.9
FEATURE_FRACTION = 0.9
SEED = 1000

pd.options.display.max_columns = 100
pd.options.display.width = 200

plt.rcParams["figure.subplot.left"] = 0.4
plt.rcParams["figure.subplot.right"] = 0.95


def log_evaluation(period=1, show_stdv=True):
    """Create a callback that logs evaluation result with logger.
    https://stackoverflow.com/questions/46619974/save-the-output-of-xgb-train-of-xgboost-as-a-log-file-with-python-logging

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that logs evaluation every period iterations into logger.
    """

    def _fmt_metric(value, show_stdv=True):
        """format metric string"""
        if len(value) == 2:
            return '%s:%g' % (value[0], value[1])
        elif len(value) == 3:
            if show_stdv:
                return '%s:%g+%g' % (value[0], value[1], value[2])
            else:
                return '%s:%g' % (value[0], value[1])
        else:
            raise ValueError("wrong metric value")

    def callback(env):
        logger = getLogger('root')
        if env.rank != 0 or len(env.evaluation_result_list) == 0 or period is False:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            logger.info('[%d]\t%s\n' % (i, msg))

    return callback


def main():

    logger = init_logger('_log/train.log')

    random.seed(SEED)

    train_all = pd.read_csv('train.csv')

    game_ids = sorted(list(set(train_all['game_id'].tolist())))
    random.shuffle(game_ids)

    train_game_ids = game_ids[:len(game_ids) // 2]
    valid_game_ids = game_ids[len(game_ids) // 2:]

    print(train_all.head())

    logger.info(f'train_game_ids: {train_game_ids[:5]} ...')
    logger.info(f'valid_game_ids: {valid_game_ids[:5]} ...')

    list_features = train_all.columns.tolist()
    list_features.remove('threshold')
    list_features.remove('game_id')

    train = train_all.query('game_id in @train_game_ids')
    valid = train_all.query('game_id in @valid_game_ids')

    logger.info(f'train shape: {train.shape}')
    logger.info(f'valid shape: {valid.shape}')

    train_targets = train['threshold']
    valid_targets = valid['threshold']

    train_features = train[list_features]
    valid_features = valid[list_features]

    train_group = np.full(len(train_features) // 100, 100)
    valid_group = np.full(len(valid_features) // 100, 100)

    train_features.drop('game_id_round', axis=1, inplace=True)
    valid_features.drop('game_id_round', axis=1, inplace=True)

    xgb_train_set = DMatrix(train_features, train_targets)
    xgb_valid_set = DMatrix(valid_features, valid_targets)

    xgb_train_set.set_group(train_group)
    xgb_valid_set.set_group(valid_group)

    params = {
        'objective': 'rank:pairwise',
        'eta': LEARNING_RATE,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'max_depth': MAX_DEPTH,
        'max_leaves': NUM_LEAVES,
        'lambda': LAMBDA_L2,
        'alpha': LAMBDA_L1,
        'subsample': BAGGING_FRACTION,
        'colsample_bytree': FEATURE_FRACTION
    }
    callbacks = [log_evaluation(100, True)]
    model = xgb.train(params, xgb_train_set, num_boost_round=NUM_ROUND,
                      evals=[(xgb_valid_set, 'validation')],
                      callbacks=callbacks,
                      verbose_eval=100)

    _, ax = plt.subplots(figsize=(12, 12))
    xgb.plot_importance(model,
                        ax=ax,
                        importance_type='gain',
                        show_values=False)
    plt.savefig('importance.png')

    model_b = pickle.dumps(model)

    with open('model.txt', 'w') as f:
        f.write(str(model_b))

    with open("submit_main.py") as f:
        submit_main = f.read()

    with open("model.txt", "r") as model_file:

        with open(f"submission_round{NUM_ROUND}.py", "w") as f:

            f.write("\nMODEL_B = ")
            f.write(model_file.read())
            f.write("\n\n")

            f.write(submit_main)

    list_features.remove('game_id_round')
    pred = model.predict(DMatrix(train_all.query('round == 0')[list_features]))
    logger.info(f'predict for non-touched machines: {pred}')


if __name__ == '__main__':
    main()
