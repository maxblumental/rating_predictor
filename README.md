# Предсказание рейтинга

## Валидация

Деление датасета делалось в [splits.py](src/splits.py) по таймстемпу.

Обучающая выборка была разделена на `dev` и `holdout` выборки. В свою очередь, `dev` поделен на `dev1` и `dev2`.

* `dev1` - обучение
* `dev2` - тюнинг гиперпараметров
* `holdout` - выбор между подходами

## Подходы

* [model1_simple_average.py](src/model1_simple_average.py) - средний рейтинг в качестве предсказания.
* [model2_surprise_baseline.py](src/model2_surprise_baseline.py) - модель `BaselineOnly` из библиотеки `surprise`.
* [model3_surprise_svd.py](src/model3_surprise_svd.py) - модель `SVD` из библиотеки `surprise`.
* [model4_nn.py](src/model4_nn.py) - нейронная сеть на PyTorch (тюнинг гиперпараметров
  в [model4_nn_tuning.py](src/model4_nn_tuning.py)).

## Результаты

| Model             | Test RMSE |
|-------------------|-----------|
| simple average    | 2.1836    |
| surprise baseline | 2.0332    |
| surprise svd      | 2.0227    |
| neural network    | 2.0325    |

Нейронная сеть показала качества лишь немного лучше чем `surprise.BaselineOnly` и хуже чем `surprise.SVD`.
Нужны дальнейшие эксперименты с архитектурой и тюнинг гиперпараметров.
