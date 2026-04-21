# Домашнее задание 2: Neural Transition Ranker

## Основная идея

Я реализовал для Botify двухстадийную рекомендательную систему. В контрольной группе используется исходный бейзлайн `SasRec-I2I`. В treatment используется обученный neural transition ranker: сначала для текущего трека формируется компактный набор кандидатов, затем DLRM-style модель переупорядочивает этих кандидатов на основе score, обученного по логам взаимодействий. Это не ручная эвристика по артисту, жанру или популярности: итоговый порядок рекомендаций в treatment меняется supervised ML-моделью, обученной на наблюдаемых переходах между треками.

## Детали

DLRM transition model обучался offline на логах взаимодействий, взятых из Google Drive который скидывали на семинаре (DLRM_Transition_I2I_HW2.ipynb), обучал на T4 GPU в Colab. Модель оценивает качество перехода `current_track -> candidate_track`. После обучения ее ranking сохраняется в `botify/data/dlrm_transition_i2i.jsonl`.

Для online-сервиса я заранее строю файл `botify/data/dlrm_sasrec_rerank_i2i.jsonl`: беру кандидатов из `sasrec_i2i.jsonl` и переупорядочиваю их DLRM-сигналом по формуле `sasrec_rank + 0.055 * dlrm_rank`. Поэтому во время пользовательского запроса Botify не запускает модель, а быстро читает готовые рекомендации из Redis через `I2IRecommender`.

В A/B-тесте `HOMEWORK2` контроль `C` использует исходный `sasrec_i2i_recommender`, а treatment `T1` использует `dlrm_sasrec_rerank_i2i_recommender`. Например, для `item_id = 0` процесс выглядит так:

1. Берем SasRec список кандидатов: `[2, 7133, 6, 7, 1, 7134, 9, 7104, 8, 3]`.
2. Берем DLRM transition ranking для того же `item_id = 0`.
3. Для каждого кандидата из SasRec смотрим, на каком месте он у SasRec и на каком месте он у DLRM.
4. Считаем общий score: `score = sasrec_rank + 0.055 * dlrm_rank`.
5. Сортируем кандидатов по этому score.
6. Получаем новый список: `[7133, 2, 6, 7, 1, 7134, 9, 7104, 8, 3]`.

```text
                  OFFLINE TRAINING / PRECOMPUTE

  Логи взаимодействий из Google диска (был на семинаре)
          |
          v
  DLRM transition model
          |
          v
  dlrm_transition_i2i.jsonl
          |
          |        sasrec_i2i.jsonl
          |              |
          |              v
          +------> Candidate retrieval
                         |
                         v
                  Neural reranking
                         |
                         v
            dlrm_sasrec_rerank_i2i.jsonl


                         ONLINE A/B TEST

        Control C                          Treatment T1
   sasrec_i2i_recommender        dlrm_sasrec_rerank_i2i_recommender
        |                                      |
        v                                      v
  следующий трек от SasRec          следующий трек после ML-reranking
```

## Результаты A/B эксперимента

A/B-тест был проведен в эксперименте `HOMEWORK2`: в контроле использовался неизмененный `SasRec-I2I`, в treatment — neural transition reranker. Целевая метрика статистически значимо улучшилась: `mean_session_time` выросла на `1.33%` при `pvalue = 0.0456` на CI-like запуске с `30000` эпизодов и `seed=31312`.

| metric | control_mean | treatment_mean | effect_% | pvalue | significant |
|---|---:|---:|---:|---:|---|
| mean_session_time | 6.78623 | 6.87640 | 1.32873 | 0.0455989 | True |
| mean_tracks_per_session | 11.7062 | 11.8001 | 0.802095 | 0.069579 | False |
| mean_request_latency | 0.000397676 | 0.000397842 | 0.0418469 | 0.700238 | False |
| sessions | 15200 | 15196 | -0.0263158 | nan | False |
