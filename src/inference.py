from pathlib import Path

import pandas as pd
from surprise import Reader, Dataset, SVD

if __name__ == '__main__':
    data_dir = Path('../data')
    train_df = pd.read_csv(data_dir / 'train.csv', compression='gzip')
    test_df = pd.read_csv(data_dir / 'test.csv', compression='gzip')

    reader = Reader(rating_scale=(1, 10))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    X_test = test_df[['userId', 'movieId']].values
    test_df['rating'] = [algo.predict(*row).est for row in X_test]
    test_df.to_csv(path_or_buf='submission.csv', compression='gzip')
