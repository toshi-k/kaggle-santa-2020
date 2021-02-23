import random
import pickle
import logging

import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm.callback import _format_eval_result

from log import init_logger

ALPHA = 0.65
NUM_ROUND = 4000
LEARNING_RATE = 0.05
NUM_LEAVES = 8
MAX_DEPTH = 8
LAMBDA_L1 = 0.2
LAMBDA_L2 = 0.2
BAGGING_FRACTION = 0.9
FEATURE_FRACTION = 0.9
SEED = 1002

pd.options.display.max_columns = 100
pd.options.display.width = 200

plt.rcParams["figure.subplot.left"] = 0.4
plt.rcParams["figure.subplot.right"] = 0.95


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    """https://amalog.hateblo.jp/entry/lightgbm-logging-callback
    """
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback


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

    train_all = train_all.sample(frac=1.0)

    train = train_all.query('game_id in @train_game_ids')
    valid = train_all.query('game_id in @valid_game_ids')

    logger.info(f'train shape: {train.shape}')
    logger.info(f'valid shape: {valid.shape}')

    train_targets = train['threshold']
    valid_targets = valid['threshold']

    train_features = train[list_features]
    valid_features = valid[list_features]

    lgb_train_set = lgb.Dataset(train_features, label=train_targets, free_raw_data=False)
    lgb_valid_set = lgb.Dataset(valid_features, label=valid_targets, free_raw_data=False)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'quantile',
        'alpha': ALPHA,
        'n_jobs': -1,
        'num_leaves': NUM_LEAVES,
        'learning_rate': LEARNING_RATE,
        'max_depth': MAX_DEPTH,
        'lambda_l1': LAMBDA_L1,
        'lambda_l2': LAMBDA_L2,
        'bagging_fraction': BAGGING_FRACTION,
        'bagging_freq': 1,
        'feature_fraction': FEATURE_FRACTION,
        'verbose': 1,
    }

    callbacks = [log_evaluation(logger, period=100)]

    model = lgb.train(
        params, lgb_train_set,
        num_boost_round=NUM_ROUND,
        early_stopping_rounds=100,
        valid_sets=[lgb_train_set, lgb_valid_set],
        verbose_eval=100,
        callbacks=callbacks
    )

    lgb.plot_importance(model)
    plt.savefig('importance.png')

    model_b = pickle.dumps(model)

    with open('model.txt', 'w') as f:
        f.write(str(model_b))

    pred = model.predict(train_all.query('round == 0')[list_features])
    logger.info(f'predict for non-touched machines: {pred}')


if __name__ == '__main__':
    main()
