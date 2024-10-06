import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from surprise import Reader, Dataset, BaselineOnly

if __name__ == '__main__':
    train_df = pd.read_parquet('../data/dev.parquet')
    test_df = pd.read_parquet('../data/holdout.parquet')

    reader = Reader(rating_scale=(1, 10))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()

    algo = BaselineOnly()
    algo.fit(trainset)

    y_true = test_df['rating'].values
    X_test = test_df[['userId', 'movieId', 'rating']].values
    y_pred = np.array([p.est for p in (algo.test(X_test))])
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f'test RMSE = {rmse:.4f}')
