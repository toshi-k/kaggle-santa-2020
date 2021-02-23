import json
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

from log import init_logger


def main():

    logger = init_logger('_log/download.log')

    # TODO describe target submission IDs here
    target_submission_ids = [
    ]

    dir_data = Path('../../data')
    dir_data.mkdir(exist_ok=True)

    for submission_id in tqdm(target_submission_ids, desc='submission'):

        logger.info(f'submission_id: {submission_id}')

        r = requests.post('https://www.kaggle.com/requests/EpisodeService/ListEpisodes',
                          json={"submissionId": submission_id})
        df = pd.DataFrame(r.json()['result']['episodes']).sort_values('id')

        for i, game_id in enumerate(tqdm(df['id'], desc='game')):

            r = requests.post('https://www.kaggle.com/requests/EpisodeService/GetEpisodeReplay',
                              json={"EpisodeId": game_id})

            if r.status_code != 200:
                print(r.text)
                raise Exception('Status code is not 200')

            replay = json.loads(r.json()['result']['replay'])

            dir_save = dir_data / str(submission_id)
            dir_save.mkdir(exist_ok=True)

            with open(dir_save / f'{game_id}.json', 'w') as f:
                json.dump(replay, f, indent=2)


if __name__ == '__main__':
    main()
