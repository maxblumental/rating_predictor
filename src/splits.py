from pathlib import Path
from typing import Tuple

import pandas as pd


def split_by_timestamp(df: pd.DataFrame, second_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    second_start = int(df['timestamp'].quantile(1 - second_size))
    first = df.query(f'timestamp < {second_start}').copy()
    second = df.query(f'timestamp >= {second_start}').copy()
    return first, second


def make_dev_holdout_splits(data_dir: Path, holdout_size: float):
    df = pd.read_csv(data_dir / 'train.csv', compression='gzip')
    dev, holdout = split_by_timestamp(df, second_size=holdout_size)
    dev.to_parquet(data_dir / 'dev.parquet', compression='gzip')
    holdout.to_parquet(data_dir / 'holdout.parquet', compression='gzip')


def make_dev_splits(data_dir: Path, second_size: float):
    df = pd.read_parquet(data_dir / 'dev.parquet')
    dev1, dev2 = split_by_timestamp(df, second_size)
    dev1.to_parquet(data_dir / 'dev1.parquet', compression='gzip')
    dev2.to_parquet(data_dir / 'dev2.parquet', compression='gzip')


if __name__ == '__main__':
    data_dir = Path('../data')
    make_dev_holdout_splits(data_dir, holdout_size=0.1)
    make_dev_splits(data_dir, second_size=0.1)
