# Итог сравнения ALS baseline vs ALS + CatBoost

Все сравнения сделаны на одинаковых `train/test/candidates` внутри каждого эксперимента. `ALS baseline` использует только `user_id`, `item_id`, `value`; `ALS + CatBoost` дополнительно использует user/item признаки, если они есть в подготовленном train-файле.

## Сводная таблица

| Датасет | Сценарий | HitRate@10 ALS | HitRate@10 Hybrid | Delta HitRate | NDCG@10 ALS | NDCG@10 Hybrid | Delta NDCG | Вывод |
|---|---|---:|---:|---:|---:|---:|---:|---|
| personalized | eval_100 | 0.107400 | 0.104600 | -0.002800 | 0.049797 | 0.047072 | -0.002724 | ALS лучше по HitRate |
| personalized | full_catalog | 0.005000 | 0.005200 | 0.000200 | 0.002262 | 0.002459 | 0.000197 | Hybrid лучше по HitRate |
| ecommerce | eval_100 | 0.111400 | 0.124000 | 0.012600 | 0.050208 | 0.056773 | 0.006565 | Hybrid лучше по HitRate |
| shopping | eval_100 | 0.095238 | 0.161905 | 0.066667 | 0.060653 | 0.062727 | 0.002074 | Hybrid лучше по HitRate |
| ecommerce_consumer | full_5000 | 0.628400 | 0.648000 | 0.019600 | 0.284905 | 0.290137 | 0.005232 | Hybrid лучше по HitRate |
| hm_small | last_30d_eval_100 | 0.186200 | 0.094400 | -0.091800 | 0.062813 | 0.031710 | -0.031103 | ALS лучше по HitRate |
| hm_180d | last_180d_eval_100 | 0.088200 | 0.039400 | -0.048800 | 0.024389 | 0.010924 | -0.013466 | ALS лучше по HitRate |

## Детальная таблица

| Датасет | Сценарий | Модель | Precision@10 | Recall@10 | HitRate@10 | MAP@10 | NDCG@10 | Coverage | Eval users | Candidate pairs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| personalized | eval_100 | ALS baseline | 0.010740 | 0.107400 | 0.107400 | 0.032718 | 0.049797 | 1.000000 | 5000.000000 | 500000.000000 |
| personalized | eval_100 | ALS + CatBoost | 0.010460 | 0.104600 | 0.104600 |  | 0.047072 | 1.000000 | 5000.000000 | 500000.000000 |
| personalized | full_catalog | ALS baseline | 0.000500 | 0.005000 | 0.005000 | 0.001448 | 0.002262 | 0.999000 | 5000.000000 | 10000000.000000 |
| personalized | full_catalog | ALS + CatBoost | 0.000520 | 0.005200 | 0.005200 |  | 0.002459 | 0.999500 | 5000.000000 | 10000000.000000 |
| ecommerce | eval_100 | ALS baseline | 0.011140 | 0.111400 | 0.111400 | 0.032132 | 0.050208 | 1.000000 | 5000.000000 | 500000.000000 |
| ecommerce | eval_100 | ALS + CatBoost | 0.012400 | 0.124000 | 0.124000 |  | 0.056773 | 0.998473 | 5000.000000 | 500000.000000 |
| shopping | eval_100 | ALS baseline | 0.009524 | 0.095238 | 0.095238 | 0.050329 | 0.060653 | 0.413091 | 105.000000 | 10500.000000 |
| shopping | eval_100 | ALS + CatBoost | 0.016190 | 0.161905 | 0.161905 |  | 0.062727 | 0.220572 | 105.000000 | 10500.000000 |
| ecommerce_consumer | full_5000 | ALS baseline | 0.147400 | 0.403911 | 0.628400 | 0.189236 | 0.284905 | 0.917910 | 5000.000000 | 670000.000000 |
| ecommerce_consumer | full_5000 | ALS + CatBoost | 0.151960 | 0.421512 | 0.648000 | 0.190959 | 0.290137 | 0.798507 | 5000.000000 | 670000.000000 |
| hm_small | last_30d_eval_100 | ALS baseline | 0.023640 | 0.100912 | 0.186200 | 0.038874 | 0.062813 | 0.095058 | 5000.000000 | 500000.000000 |
| hm_small | last_30d_eval_100 | ALS + CatBoost | 0.011420 | 0.045670 | 0.094400 | 0.020404 | 0.031710 | 0.041502 | 5000.000000 | 500000.000000 |
| hm_180d | last_180d_eval_100 | ALS baseline | 0.010960 | 0.041157 | 0.088200 | 0.013345 | 0.024389 | 0.066180 | 5000.000000 | 500000.000000 |
| hm_180d | last_180d_eval_100 | ALS + CatBoost | 0.004800 | 0.016679 | 0.039400 | 0.006191 | 0.010924 | 0.034448 | 5000.000000 | 500000.000000 |

## Warm / Cold Items

Warm/cold split посчитан по train-файлам гибридной модели. Правило: item считается warm, если число взаимодействий в train `>= min_warm_interactions`; иначе item считается cold.

| Датасет | Сценарий | Train rows | Threshold | Total items | Warm items | Cold items | Warm share | Cold share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| personalized | eval_100 / full_catalog | 145000 | 5 | 2000 | 2000 | 0 | 1.000000 | 0.000000 |
| ecommerce | eval_100 | 202012 | 5 | 1964 | 1964 | 0 | 1.000000 | 0.000000 |
| shopping | eval_100 | 3795 | 2 | 1805 | 1043 | 762 | 0.577839 | 0.422161 |
| ecommerce_consumer | full_5000 | 1470178 | 30 | 134 | 134 | 0 | 1.000000 | 0.000000 |
| hm_small | last_30d_eval_100 | 882591 | 30 | 27201 | 5572 | 21629 | 0.204845 | 0.795155 |
| hm_180d | last_180d_eval_100 | 7782138 | 30 | 50338 | 21887 | 28451 | 0.434801 | 0.565199 |


# Training Cost Summary

Скрипт прогоняет обучение двух моделей на одинаковых train-файлах:

- `hybrid` = ALS + CatBoost из `cold-item`
- `als` = обычный ALS baseline из `recommender`

Метрики затрат:

- `wall_time_sec` — полное wall-clock время обучения
- `peak_rss_mb` — максимальный RSS процесса и его дочерних процессов

| Dataset | Model | Status | Wall time (sec) | Peak RSS (MB) | Train rows | Users | Items | Warm items | Cold items |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| personalized | hybrid | ok | 20.94 | 505.03 | 145000 | 5000 | 2000 | 2000 | 0 |
| personalized | als | ok | 2.93 | 117.90 | 144980 | 5000 | 2000 |  |  |
| ecommerce | hybrid | ok | 45.48 | 495.36 | 201975 | 49673 | 1964 | 1964 | 0 |
| ecommerce | als | ok | 13.62 | 139.05 | 201975 | 49673 | 1964 |  |  |
| shopping | hybrid | ok | 2.92 | 168.39 | 3795 | 105 | 1805 | 1043 | 762 |
| shopping | als | ok | 1.67 | 88.01 | 3794 | 105 | 1805 |  |  |
| ecommerce_consumer | hybrid | ok | 73.98 | 1886.60 | 1059949 | 105273 | 134 | 134 | 0 |
| ecommerce_consumer | als | ok | 32.70 | 478.50 | 891763 | 105273 | 134 |  |  |
| hm_small | hybrid | ok | 263.31 | 2204.12 | 782721 | 205159 | 27201 | 5187 | 22014 |
| hm_small | als | ok | 54.46 | 407.26 | 782721 | 205159 | 27201 |  |  |

## Configs

| Dataset | Hybrid config | ALS config |
|---|---|---|
| personalized | min_warm=5, als_iter=150, reg_iter=300, neg_per_user=3 | min_user=1, min_item=1, iter=150, reg=0.01 |
| ecommerce | min_warm=5, als_iter=50, reg_iter=100, neg_per_user=1 | min_user=1, min_item=1, iter=150, reg=0.01 |
| shopping | min_warm=2, als_iter=50, reg_iter=100, neg_per_user=1 | min_user=1, min_item=1, iter=150, reg=0.01 |
| ecommerce_consumer | min_warm=30, als_iter=50, reg_iter=100, neg_per_user=1 | min_user=1, min_item=1, iter=150, reg=0.01 |
| hm_small | min_warm=30, als_iter=50, reg_iter=100, neg_per_user=1 | min_user=1, min_item=1, iter=150, reg=0.01 |

## Ключевые выводы

- `ecommerce`: гибрид немного лучше ALS: HitRate@10 вырос с 0.111400 до 0.124000, NDCG@10 с 0.050208 до 0.056773.
- `personalized`: значимого улучшения нет. На `eval_100` ALS немного лучше, на полном каталоге гибрид лучше только на 0.000200 по HitRate@10, что практически несущественно.
- `shopping`: гибрид заметно лучше по попаданиям: HitRate@10 вырос с 0.095238 до 0.161905, но coverage ниже. Важно: оценка по сегментам пользователей, а не по исходным Customer ID.
- `ecommerce_consumer`: гибрид умеренно лучше ALS: HitRate@10 вырос с 0.628400 до 0.648000, Recall@10 с 0.403911 до 0.421512, но coverage ниже.
- `H&M`: обычный ALS существенно лучше гибрида и на 30-дневном окне, и на 180-дневном. Увеличение окна с 30 до 180 дней ухудшило обе модели, что указывает на важность свежести данных в fashion-домене.

## Общий итог

Гибрид ALS + CatBoost даёт прирост на `ecommerce`, `shopping` и `ecommerce_consumer`, но эффект зависит от датасета. На `personalized` прирост отсутствует или находится на уровне шума. На H&M текущая гибридная схема ухудшает качество относительно обычного ALS, вероятно потому что сильнейший сигнал находится в свежей user-item истории, а доступные user/item признаки не компенсируют сезонность и динамику модных товаров.

Файл с полной машинно-читаемой таблицей: `tests/comparison_summary.csv`.
