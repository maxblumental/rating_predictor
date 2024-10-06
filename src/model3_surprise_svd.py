import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from surprise import Reader, Dataset, SVD

if __name__ == '__main__':
    train_df = pd.read_parquet('../data/dev.parquet')
    test_df = pd.read_parquet('../data/holdout.parquet')

    reader = Reader(rating_scale=(1, 10))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    y_true = test_df['rating'].values
    X_test = test_df[['userId', 'movieId']].values
    y_pred = np.array([algo.predict(*row).est for row in X_test])
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f'test RMSE = {rmse:.4f}')
