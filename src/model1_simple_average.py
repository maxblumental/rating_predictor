import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

if __name__ == '__main__':
    train_df = pd.read_parquet('../data/dev.parquet')
    test_df = pd.read_parquet('../data/holdout.parquet')

    avg_rating = train_df['rating'].mean()
    y_true = test_df['rating'].values
    y_pred = np.full_like(y_true, fill_value=avg_rating)
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f'test RMSE = {rmse:.4f}')
