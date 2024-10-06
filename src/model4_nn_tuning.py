import gc
from pathlib import Path
from typing import Dict, Tuple

import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

##### DATA #####

GENRE_TO_CODE = {
    'Adventure': 1,
    'Animation': 2,
    'Children': 3,
    'Comedy': 4,
    'Fantasy': 5,
    'Romance': 6,
    'Drama': 7,
    'Action': 8,
    'Crime': 9,
    'Thriller': 10,
    'Horror': 11,
    'Mystery': 12,
    'Sci-Fi': 13,
    'IMAX': 14,
    'Documentary': 15,
    'War': 16,
    'Musical': 17,
    'Western': 18,
    'Film-Noir': 19,
}
GENRE_PADDING = len(GENRE_TO_CODE) + 1

EDGE_YEARS = [
    1995, 1996, 1998, 1999, 2001, 2003, 2005, 2007, 2010, 2013, 2015
]


def encode_top_values(df: pd.DataFrame, value_col: str, frequency_col: str, threshold: int):
    value_freq = df.groupby(value_col)[frequency_col].nunique().sort_values(ascending=False)
    top_values = value_freq[value_freq >= threshold].index.tolist()
    return {value: code for code, value in enumerate(top_values, start=1)}


def encode_movies(df: pd.DataFrame, min_users_per_movie: int) -> Dict[int, int]:
    return encode_top_values(df, value_col='movieId', frequency_col='userId', threshold=min_users_per_movie)


def encode_users(df: pd.DataFrame, min_movies_per_user: int) -> Dict[int, int]:
    return encode_top_values(df, value_col='userId', frequency_col='movieId', threshold=min_movies_per_user)


def make_dataset(df: pd.DataFrame) -> Dataset:
    user_code = torch.tensor(df['userId'].values, dtype=torch.long)
    movie_code = torch.tensor(df['movieId'].values, dtype=torch.long)
    genres = torch.tensor(df['genres'].tolist(), dtype=torch.long)
    year = torch.tensor(df['year_code'].tolist(), dtype=torch.long)
    rating = torch.tensor(df['rating'].values, dtype=torch.float)
    return TensorDataset(user_code, movie_code, genres, year, rating)


def encode_year(year: float):
    if pd.isna(year):
        return 0

    for i, edge in enumerate(EDGE_YEARS):
        if year <= edge:
            return i + 1

    return len(EDGE_YEARS) + 1


def preprocess_data(movies_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    min_movies_per_user: int, min_users_per_movie: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_genres_num = movies_df['genres'].str.split('|').str.len().max()
    movies_df['genres'] = [[GENRE_TO_CODE.get(g, 0) for g in genres.split('|')] for genres in movies_df['genres']]
    movies_df['genres'] = [g + [GENRE_PADDING] * (max_genres_num - len(g)) for g in movies_df['genres']]

    movies_df['year'] = movies_df['title'].str.rsplit('(', n=1).str[1].str.strip(')').str.strip(') ')
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce').astype('UInt64')
    movies_df['year_code'] = [encode_year(year) for year in movies_df['year']]

    user_codes = encode_users(train_df, min_movies_per_user)
    movie_codes = encode_movies(train_df, min_users_per_movie)

    def preprocess(df):
        df = df.merge(movies_df[['movieId', 'genres', 'year_code']])
        df['userId'] = [user_codes.get(userId, 0) for userId in df['userId']]
        df['movieId'] = [movie_codes.get(movieId, 0) for movieId in df['movieId']]
        return df

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    return train_df, test_df


##### Model #####

class RecommenderNN(pl.LightningModule):
    def __init__(self, num_users: int, num_movies: int, embedding_dim: int,
                 genre_embedding_dim: int, year_embedding_dim: int,
                 reg: float, reg_u: float, reg_i: float):
        super(RecommenderNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.user_bias = nn.Embedding(num_users + 1, embedding_dim=1)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)
        self.movie_bias = nn.Embedding(num_users + 1, embedding_dim=1)
        self.genre_embedding = nn.Embedding(len(GENRE_TO_CODE) + 2, genre_embedding_dim, padding_idx=GENRE_PADDING)
        self.year_embedding = nn.Embedding(len(EDGE_YEARS) + 2, year_embedding_dim)
        self.fc_net = nn.Sequential(
            nn.Linear(2 * embedding_dim + genre_embedding_dim + year_embedding_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),

        )
        self.bias = nn.Parameter(torch.tensor(7.0))
        self.reg = reg
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.criterion = nn.MSELoss()

    def forward(self, features):
        user_idx, movie_idx, genre_idx, year_idx = features
        user = self.user_embedding(user_idx)
        features = torch.cat(tensors=[
            user,
            self.movie_embedding(movie_idx),
            self.embed_genres(genre_idx),
            self.year_embedding(year_idx),
        ], dim=-1)
        b_u = self.user_bias(user_idx).squeeze()
        b_i = self.movie_bias(movie_idx).squeeze()
        cross = self.fc_net(features).squeeze()
        rating = cross + self.bias + b_u + b_i
        return rating

    def embed_genres(self, genre_idx):
        genre_embeds = self.genre_embedding(genre_idx)
        weights = torch.ones_like(genre_idx, dtype=torch.float32)
        weights = weights.masked_fill(genre_idx == GENRE_PADDING, float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        n = (genre_idx != GENRE_PADDING).sum(dim=-1, keepdim=True)
        genre = torch.sum(weights.unsqueeze(-1) * genre_embeds, dim=-2) * torch.sqrt(n)
        return genre

    def training_step(self, batch, batch_idx):
        features, ratings = batch[:-1], batch[-1]
        predictions = self(features)
        loss = self.criterion(predictions, ratings)
        self.log('train_loss', loss)
        self.log('train_rmse', torch.sqrt(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, ratings = batch[:-1], batch[-1]
        predictions = self(features)
        loss = self.criterion(predictions, ratings)
        self.log('val_loss', loss)
        self.log('val_rmse', torch.sqrt(loss), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.user_embedding.parameters(), 'weight_decay': self.reg_u},
            {'params': self.user_bias.parameters(), 'weight_decay': self.reg_u},
            {'params': self.movie_embedding.parameters(), 'weight_decay': self.reg_i},
            {'params': self.movie_bias.parameters(), 'weight_decay': self.reg_i},
            {'params': self.genre_embedding.parameters(), 'weight_decay': self.reg_i},
            {'params': self.year_embedding.parameters(), 'weight_decay': self.reg_i},
            {'params': self.fc_net.parameters(), 'weight_decay': self.reg},
            {'params': self.bias, 'weight_decay': 0.0}
        ], lr=3e-4)

        return optimizer

    def evaluate(self, data_loader):
        self.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                features, ratings = batch[:-1], batch[-1]
                predictions = self(features)
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item() * len(ratings)
                total_samples += len(ratings)

        if total_samples == 0:
            return 0

        return (total_loss / total_samples) ** 0.5


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, params: Dict):
    train_loader = DataLoader(make_dataset(train_df), shuffle=True, **params['dataloader'])
    test_loader = DataLoader(make_dataset(test_df), **params['dataloader'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=params['checkpoint_path'],
        filename=params['model_name'],
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
    )
    model = RecommenderNN(num_users=train_df['userId'].max(),
                          num_movies=train_df['movieId'].max(),
                          **params['model'])
    logger = TensorBoardLogger(save_dir="lightning_logs", name=params['model_name'])
    trainer = pl.Trainer(
        max_epochs=params['n_epochs'],
        accelerator=get_device(),
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model, train_loader, test_loader)

    print('RMSE:', model.evaluate(test_loader))

    del trainer
    del train_loader
    del test_loader
    gc.collect()


def test_model(train_df: pd.DataFrame, test_df: pd.DataFrame, params: Dict) -> float:
    checkpoint_path = Path(params['checkpoint_path']) / (params['model_name'] + '.ckpt')
    model = RecommenderNN.load_from_checkpoint(checkpoint_path,
                                               num_users=train_df['userId'].max(),
                                               num_movies=train_df['movieId'].max(),
                                               **params['model'])
    model.to(torch.device('cpu'))

    test_loader = DataLoader(make_dataset(test_df), **params['dataloader'])
    rmse = model.evaluate(test_loader)

    del test_loader
    gc.collect()

    return rmse


def objective(trial):
    params = {
        'model_name': f'trial{trial.number}',
        'n_epochs': 64,
        'model': {
            'reg': 10 ** trial.suggest_float('reg', -6, 1),
            'reg_u': 10 ** trial.suggest_float('reg_u', -6, 1),
            'reg_i': 10 ** trial.suggest_float('reg_i', -6, 1),
            'embedding_dim': 16,
            'genre_embedding_dim': 8,
            'year_embedding_dim': 8,
        },
        'dataloader': {
            'batch_size': trial.suggest_categorical('batch_size', [512, 2048, 8192]),
            'num_workers': 64,
            'persistent_workers': True,
        },
        'checkpoint_path': 'checkpoints/',
    }

    train_model(train_df, val_df, params)
    return test_model(train_df, val_df, params)


if __name__ == '__main__':
    movies_df = pd.read_csv('../data/movies.csv')
    train_df = pd.read_parquet('../data/dev1.parquet')
    val_df = pd.read_parquet('../data/dev2.parquet')
    train_df, val_df = preprocess_data(movies_df, train_df, val_df,
                                       min_movies_per_user=10, min_users_per_movie=10)

    study = optuna.create_study(direction='minimize', study_name="rating_prediction",
                                storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(objective, n_trials=100)
