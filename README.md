# Santa 2020 - The Candy Cane Contest

My solution in this Kaggle competition ["Santa 2020 - The Candy Cane Contest"](https://www.kaggle.com/c/santa-2020), 9th place.

## Basic Strategy
In this competition, the reward was decided by comparing the threshold and random generated number. It was easy to calculate the probability of getting reward if we knew the thresholds. But the agents can't see the threshold during the game, we had to estimate it.

Like other teams, I also downloaded the history by Kaggle API and created a dataset for supervised learning. We can see the true value of `threshold` at each round in the response of API. So, I used it as the target variable.

In the middle of the competition, I found out that quantile regression is much better than conventional L2 regression. I think it can adjust the balance between **Explore** and **Exploit** by the percentile parameter.

## Features

| &nbsp; &nbsp; &nbsp; &nbsp; # &nbsp; &nbsp; &nbsp; &nbsp; | Name | Explanation |
| --- | --- | --- |
| #1 | round | number of round in the game (0-1999)
| #2 | last_opponent_chosen | whether the opponent agent chose this machine in the last step or not
| #3 | second_last_opponent_chosen | whether the opponent agent chose this machine in the second last step or not
| #4 | third_last_opponent_chosen | whether the opponent agent chose this machine in the third last step or not
| #5 | opponent_repeat_twice | whether the opponent agent continued to choose this machine in the last two rounds (#2 x #3)
| #6 | opponent_repeat_three_times | whether the opponent agent continued to choose this machine in the last three rounds (#2 x #3 x #4)
| #7 | num_chosen | how many times the opponent and my agent chose this machine
| #8 | num_chosen_mine | how many times my agent chose this machine
| #9 | num_chosen_opponent | how many time the opponent agent chose this machine (#7 - #8)
| #10 | num_get_reward | how many time my agent got rewards from this machine
| #11 | num_non_reward | how many time my agent didn't get rewarded from this machine
| #12 | rate_mine | ratio of my choices against the total number of choices (#8 / #7)
| #13 | rate_opponent | ratio of opponent choices against the total number of choices (#9 / #7)
| #14 | rate_get_reward | ratio of my rewarded choices against the total number of choices (#10 / #7)
| #15 | empirical_win_rate | posterior expectation of threshold value based on my choices and rewords
| #16 | quantile_10 | 10% point of posterior distribution of threshold based on my choices and rewords
| #17 | quantile_20 | 20% point of posterior distribution of threshold based on my choices and rewords
| #18 | quantile_30 | 30% point of posterior distribution of threshold based on my choices and rewords
| #19 | quantile_40 | 40% point of posterior distribution of threshold based on my choices and rewords
| #20 | quantile_50 | 50% point of posterior distribution of threshold based on my choices and rewords
| #21 | quantile_60 | 60% point of posterior distribution of threshold based on my choices and rewords
| #22 | quantile_70 | 70% point of posterior distribution of threshold based on my choices and rewords
| #23 | quantile_80 | 80% point of posterior distribution of threshold based on my choices and rewords
| #24 | quantile_90 | 90% point of posterior distribution of threshold based on my choices and rewords
| #25 | repeat_head | how many times my agent chose this machine before the opponent agent chose this agent for the first time
| #26 | repeat_tail | how many times my agent chose this machine after the opponent agent chose this agent last time
| #27 | repeat_get_reward_head | how many times my agent got reward from this machine before my agent didn't get rewarded or the opponent agent chose this agent for the first time
| #28 | repeat_get_reward_tail | how many times my agent got reward from this machine after my agent didn't get rewarded or the opponent agent chose this agent last time
| #29 | repeat_non_reward_head | how many times my agent didn't get rewarded from this machine before my agent got reward or the opponent agent chose this agent for the first time
| #30 | repeat_non_reward_tail | how many times my agent didn't get rewarded from this machine after my agent got reward or the opponent agent chose this agent last time
| #31 | opponent_repeat_head | how many times the opponent agent chose this machine before my agent chose this machine for the first time
| #32 | opponent_repeat_tail | how many times the opponent agent chose this machine after my agent chose this machine last time



## Software

* Python 3.7.8
* numpy==1.18.5
* pandas==1.0.5
* matplotlib==3.2.2
* lightgbm==3.1.1
* catboost==0.24.4
* xgboost==1.2.1
* tqdm==4.47.0

## Usage

1. download data from Kaggle by `/src/01_downlaod/download.py`

2. create a dataset by `/src/02_[regressor]/preprocess.py`

3. train a model by `/src/02_[regressor]/train.py`

## Top Agents

| Regressor | Loss | NumRound | LearningRate | LB Score | SubmissionID |
| --- | --- | --- | --- | --- | --- |
| LightBGM | Quantile (0.65)	| 4000 | 0.05 | 1449.4 | [19318812](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19318812)|
| LightBGM | Quantile (0.65)	| 4000 | 0.10 | 1442.1 | [19182047](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19182047)|
| LightBGM | Quantile (0.65)	| 3000 | 0.03 | 1438.8 | [19042049](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19042049)|
| LightBGM | Quantile (0.66) | 3500 | 0.04 | 1433.9 | [19137024](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19137024)|
| CatBoost | Quantile (0.65) | 4000 | 0.05 | 1417.6 | [19153745](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19153745)|
| CatBoost | Quantile (0.67) | 3000 | 0.10 | 1344.5 | [19170829](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19170829)|
| LightGBM | MSE | 4000 | 0.03 | 1313.3 | [19093039](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19093039)|
| XGBoost | Pairwised | 1500 | 0.10 | 1173.5 | [19269952](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19269952)|
