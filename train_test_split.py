import pandas as pd
import numpy as np

def holdout(transactions_train):
    train = transactions_train.query('t_dat < "2019-09-02"')
    test = transactions_train.query('t_dat >= "2019-09-02"')\
        .groupby('customer_id')['article_id'].apply(list).to_frame('target').reset_index()
    return train, test


# def time_series_split(transactions_train=transactions_train):
#     # пытаюсь сделать кросс валидацию по временному таргету с учетом сезонности
#     train = None
#     test = None
#     for part in range(6):
#         # part1
#         if part == 0:
#             train = transactions_train.query('t_dat < "2020-09-16"')
#             test = transactions_train.query('t_dat >= "2020-09-16"')\
#                 .groupby('customer_id')['article_id'].apply(list).to_frame('target').reset_index()
#         # part2
#         elif part == 1:
#             train = transactions_train.query('t_dat < "2020-09-09"')
#             test = transactions_train.query('t_dat >= "2020-09-09"')\
#                 .groupby('customer_id')['article_id'].apply(list).to_frame('target').reset_index()
#         # part3
#         elif part == 2:
#             train = transactions_train.query('t_dat < "2020-09-02"')
#             test = transactions_train.query('t_dat >= "2020-09-02"')\
#                 .groupby('customer_id')['article_id'].apply(list).to_frame('target').reset_index()
#         # part4
#         elif part == 3:
#             train = transactions_train.query('t_dat < "2019-09-16"')
#             test = transactions_train.query('t_dat >= "2019-09-16"')\
#                 .groupby('customer_id')['article_id'].apply(list).to_frame('target').reset_index()
#         # part5
#         elif part == 4:
#             train = transactions_train.query('t_dat < "2019-09-09"')
#             test = transactions_train.query('t_dat >= "2019-09-09"')\
#                 .groupby('customer_id')['article_id'].apply(list).to_frame('target').reset_index()
#         # part6
#         elif part == 5:
#             train = transactions_train.query('t_dat < "2019-09-02"')
#             test = transactions_train.query('t_dat >= "2019-09-02"')\
#                 .groupby('customer_id')['article_id'].apply(list).to_frame('target').reset_index()
#         yield train, test
# for train, test in time_series_split:
#     print(train, test)