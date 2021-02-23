import random

import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool

from log import init_logger

ALPHA = 0.65
NUM_ROUND = 4000
LEARNING_RATE = 0.03
DEPTH = 4
LAMBDA_L2 = 0.1
SEED = 1000

pd.options.display.max_columns = 100
pd.options.display.width = 200

plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["figure.subplot.right"] = 0.95


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
    # train_all = train_all.sample(frac=0.5)

    train = train_all.query('game_id in @train_game_ids')
    valid = train_all.query('game_id in @valid_game_ids')

    logger.info(f'train shape: {train.shape}')
    logger.info(f'valid shape: {valid.shape}')

    train_targets = train['threshold']
    valid_targets = valid['threshold']

    train_features = train[list_features]
    valid_features = valid[list_features]

    cat_train_set = Pool(train_features, label=train_targets)
    cat_valid_set = Pool(valid_features, label=valid_targets)

    model = CatBoostRegressor(
        loss_function=f'Quantile:alpha={ALPHA}',
        num_boost_round=NUM_ROUND,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        l2_leaf_reg=LAMBDA_L2,
    )

    model.fit(
        cat_train_set,
        eval_set=[cat_valid_set],
        verbose_eval=100,
        early_stopping_rounds=100
    )

    feature_importance = model.get_feature_importance()
    plt.figure(figsize=(10, 10))
    plt.barh(range(len(feature_importance)),
             feature_importance,
             tick_label=train_features.columns)

    plt.xlabel('importance')
    plt.ylabel('features')
    plt.grid()
    plt.savefig('importance.png')

    with open("submit_main.py") as f:
        submit_main = f.read()

    model.save_model('model.json', format='json')

    with open("model.json", "r") as model_file:

        with open(f"submission_alpha{ALPHA:.2f}.py", "w") as f:

            f.write("\nMODEL_B = r\"\"\"")
            f.write(model_file.read())
            f.write("\"\"\"\n\n")

            f.write(submit_main)

    pred = model.predict(train_all.query('round == 0')[list_features])
    logger.info(f'predict for non-touched machines: {pred}')


if __name__ == '__main__':
    main()
