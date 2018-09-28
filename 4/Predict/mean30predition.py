#!/usr/bin/env python
# -*- coding:UTF-8 -*-

"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

# import os
# os.chdir("C:\\Users\\Administrator\\Desktop\\work\\jdata\\GOC_Semifinal_Inventory\\submission\\part")

df_train = pd.read_csv(
    'sku_sales.csv',
    converters={'quantity': lambda u: float(u) if float(u) < 10000 else 10000},
    parse_dates=["date"]  # 2016-01-01
)

df_prom = pd.read_csv(
    'sku_prom_train.csv',
    parse_dates=["date"]  # 2016-01-01
)

df_test = pd.read_csv(
    'test_prom_type.csv',
    parse_dates=["date"]  # , date_parser=parser
).set_index(['dc_id', 'item_sku_id', 'date']
            )  ##item_sku_id,date,promotion_type

df_test2 = pd.read_csv(
    'sku_discount_testing_2018MarApr1.csv', usecols=[0, 1, 2, 3, 4],
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['dc_id', 'item_sku_id', 'date']
)  # dc_id,item_sku_id,date,original_price,discount

sku_info = pd.read_csv(
    'sku_info.csv'
).set_index("item_sku_id")

attrr = pd.read_csv('sku_attr.csv').groupby(['item_sku_id', 'attr_cd']).mean().reset_index()
attrcd = np.array((np.unique(attrr['attr_cd'])))
df_rep = pd.DataFrame([[attrid, skuid] for attrid in attrcd for skuid in range(1, 1000 + 1)],
                      columns=['attr_cd', 'item_sku_id'])
df_rep = pd.merge(df_rep, attrr[['attr_cd', 'item_sku_id', 'attr_value_cd']], on=['attr_cd', 'item_sku_id'], how='left')
df_rep.fillna(value=0, inplace=True)
sku_attr = np.asarray([df_rep.loc[df_rep.item_sku_id == d].attr_value_cd.values for d in range(1, 1000 + 1)], dtype=int)
sku_attr = pd.DataFrame(sku_attr, index=range(1, 1001))

DC_id = pd.DataFrame(np.arange(6000) % 6, columns=['dc_id'])

a = df_train.date < pd.datetime(2016, 7, 1)
b = df_train.date > pd.datetime(2018, 1, 17)
c = (pd.datetime(2017, 10, 27) <= df_train.date) * (df_train.date <= pd.datetime(2017, 10, 31))
d = (pd.datetime(2016, 10, 27) <= df_train.date) * (df_train.date <= pd.datetime(2016, 10, 31))
e = (pd.datetime(2017, 5, 27) <= df_train.date) * (df_train.date <= pd.datetime(2017, 5, 31))
f = (pd.datetime(2016, 5, 27) <= df_train.date) * (df_train.date <= pd.datetime(2016, 5, 31))

dele_row = (1 - a) * (1 - b) * (1 - c) * (1 - d) * (1 - e) * (1 - f)
dele_row = dele_row.astype(bool)
df_2017 = df_train.loc[dele_row]

a = df_prom.date < pd.datetime(2016, 7, 1)
b = df_prom.date > pd.datetime(2018, 1, 17)
c = (pd.datetime(2017, 10, 27) <= df_prom.date) * (df_prom.date <= pd.datetime(2017, 10, 31))
d = (pd.datetime(2016, 10, 27) <= df_prom.date) * (df_prom.date <= pd.datetime(2016, 10, 31))
e = (pd.datetime(2017, 5, 27) <= df_prom.date) * (df_prom.date <= pd.datetime(2017, 5, 31))
f = (pd.datetime(2016, 5, 27) <= df_prom.date) * (df_prom.date <= pd.datetime(2016, 5, 31))

dele_row = (1 - a) * (1 - b) * (1 - c) * (1 - d) * (1 - e) * (1 - f)
dele_row = dele_row.astype(bool)
df_2017_prom = df_prom.loc[dele_row]

del df_train
del df_prom

discount_train = df_2017.set_index(
    ["dc_id", "item_sku_id", "date"])[["discount"]].unstack(
    level=-1).fillna(10)
discount_train.columns = discount_train.columns.get_level_values(1)
discount_test = df_test2[["discount"]].unstack(level=-1).fillna(10)
discount_test.columns = discount_test.columns.get_level_values(1)
discount_test = discount_test.reindex(discount_train.index).fillna(10)

discount_train.columns = pd.date_range(end='2018-02-28', periods=discount_train.shape[1], freq='D')
discount_2017 = pd.concat([discount_train, discount_test], axis=1)

price_train = df_2017.set_index(
    ["dc_id", "item_sku_id", "date"])[["original_price"]].unstack(
    level=-1).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
price_train.columns = price_train.columns.get_level_values(1)
price_test = df_test2[["original_price"]].unstack(level=-1).fillna(method='ffill', axis=1).fillna(method='bfill',
                                                                                                  axis=1)
price_test.columns = price_test.columns.get_level_values(1)
price_test = price_test.reindex(price_train.index).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

price_train.columns = pd.date_range(end='2018-02-28', periods=price_train.shape[1], freq='D')
price_2017 = pd.concat([price_train, price_test], axis=1)

promo_2017_train = df_2017.set_index(["dc_id", "item_sku_id", "date"])[["discount"]].unstack(level=-1).fillna(False)
t1_2017_test_prom = df_test[["type1"]].unstack(level=-1).fillna(False)
t1_2017_test_prom.columns = t1_2017_test_prom.columns.get_level_values(1)
t1_2017_test_prom = t1_2017_test_prom.reindex(promo_2017_train.index).fillna(False)

t4_2017_test_prom = df_test[["type4"]].unstack(level=-1).fillna(False)
t4_2017_test_prom.columns = t4_2017_test_prom.columns.get_level_values(1)
t4_2017_test_prom = t4_2017_test_prom.reindex(promo_2017_train.index).fillna(False)

t6_2017_test_prom = df_test[["type6"]].unstack(level=-1).fillna(False)
t6_2017_test_prom.columns = t6_2017_test_prom.columns.get_level_values(1)
t6_2017_test_prom = t6_2017_test_prom.reindex(promo_2017_train.index).fillna(False)

t10_2017_test_prom = df_test[["type10"]].unstack(level=-1).fillna(False)
t10_2017_test_prom.columns = t10_2017_test_prom.columns.get_level_values(1)
t10_2017_test_prom = t10_2017_test_prom.reindex(promo_2017_train.index).fillna(False)

df_2017_prom1 = df_2017_prom.set_index(
    ['dc_id', 'item_sku_id', 'date'])

t1_2017_train_prom = df_2017_prom1[["type1"]].unstack(level=-1).fillna(False)
t1_2017_train_prom.columns = t1_2017_train_prom.columns.get_level_values(1)
t1_2017_train_prom = t1_2017_train_prom.reindex(promo_2017_train.index).fillna(False)

t4_2017_train_prom = df_2017_prom1[["type4"]].unstack(level=-1).fillna(False)
t4_2017_train_prom.columns = t4_2017_train_prom.columns.get_level_values(1)
t4_2017_train_prom = t4_2017_train_prom.reindex(promo_2017_train.index).fillna(False)

t6_2017_train_prom = df_2017_prom1[["type6"]].unstack(level=-1).fillna(False)
t6_2017_train_prom.columns = t6_2017_train_prom.columns.get_level_values(1)
t6_2017_train_prom = t6_2017_train_prom.reindex(promo_2017_train.index).fillna(False)

t10_2017_train_prom = df_2017_prom1[["type10"]].unstack(level=-1).fillna(False)
t10_2017_train_prom.columns = t10_2017_train_prom.columns.get_level_values(1)
t10_2017_train_prom = t10_2017_train_prom.reindex(promo_2017_train.index).fillna(False)

t1_2017_train_prom.columns = pd.date_range(end='2018-02-28', periods=t1_2017_train_prom.shape[1], freq='D')
t4_2017_train_prom.columns = pd.date_range(end='2018-02-28', periods=t4_2017_train_prom.shape[1], freq='D')
t6_2017_train_prom.columns = pd.date_range(end='2018-02-28', periods=t6_2017_train_prom.shape[1], freq='D')
t10_2017_train_prom.columns = pd.date_range(end='2018-02-28', periods=t10_2017_train_prom.shape[1], freq='D')
promo_2017_1 = pd.concat([t1_2017_train_prom, t1_2017_test_prom], axis=1)
promo_2017_4 = pd.concat([t4_2017_train_prom, t4_2017_test_prom], axis=1)
promo_2017_6 = pd.concat([t6_2017_train_prom, t6_2017_test_prom], axis=1)
promo_2017_10 = pd.concat([t10_2017_train_prom, t10_2017_test_prom], axis=1)

df_2017_ven = df_2017.set_index(
    ["dc_id", "item_sku_id", "date"])[["vendibility"]].unstack(
    level=-1).fillna(-1)
df_2017_price = df_2017.set_index(
    ["dc_id", "item_sku_id", "date"])[["original_price"]].unstack(
    level=-1).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)  #
df_2017 = df_2017.set_index(
    ["dc_id", "item_sku_id", "date"])[["quantity"]].unstack(
    level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)
df_2017_ven.columns = df_2017_ven.columns.get_level_values(1)
df_2017_price.columns = df_2017_price.columns.get_level_values(1)
# df_2017_columns = df_2017.columns.get_level_values(1)
sku_info = sku_info.reindex(df_2017.index.get_level_values(1))
print(sku_info.shape)
sku_attr = sku_attr.reindex(df_2017.index.get_level_values(1))
print(sku_attr.shape)
df_2017_group_sku = df_2017.groupby('item_sku_id')[df_2017.columns].sum()
prom_2017_group_sku_1 = promo_2017_1.groupby('item_sku_id')[promo_2017_1.columns].sum()
df_2017_store_class = df_2017.reset_index()
df_2017_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_store_class_index = df_2017_store_class[['dc_id', 'first']]
df_2017_store_class = df_2017_store_class.groupby(['dc_id', 'first'])[df_2017.columns].sum()
df_2017_promo_store_class = promo_2017_1.reset_index()
df_2017_promo_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_promo_store_class_index = df_2017_promo_store_class[['dc_id', 'first']]
df_2017_promo_store_class_1 = df_2017_promo_store_class.groupby(['dc_id', 'first'])[promo_2017_1.columns].sum()

# df_2017.columns = df_2017.columns.get_level_values(1)
# sku_info = sku_info.reindex(df_2017.index.get_level_values(1))
# df_2017_group_sku = df_2017.groupby('item_sku_id')[df_2017.columns].sum()
prom_2017_group_sku_4 = promo_2017_4.groupby('item_sku_id')[promo_2017_4.columns].sum()
df_2017_store_class = df_2017.reset_index()
df_2017_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_store_class_index = df_2017_store_class[['dc_id', 'first']]
df_2017_store_class = df_2017_store_class.groupby(['dc_id', 'first'])[df_2017.columns].sum()
df_2017_promo_store_class = promo_2017_4.reset_index()
df_2017_promo_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_promo_store_class_index = df_2017_promo_store_class[['dc_id', 'first']]
df_2017_promo_store_class_4 = df_2017_promo_store_class.groupby(['dc_id', 'first'])[promo_2017_4.columns].sum()

# df_2017.columns = df_2017.columns.get_level_values(1)
# sku_info = sku_info.reindex(df_2017.index.get_level_values(1))
# df_2017_group_sku = df_2017.groupby('item_sku_id')[df_2017.columns].sum()
prom_2017_group_sku_6 = promo_2017_6.groupby('item_sku_id')[promo_2017_6.columns].sum()
df_2017_store_class = df_2017.reset_index()
df_2017_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_store_class_index = df_2017_store_class[['dc_id', 'first']]
df_2017_store_class = df_2017_store_class.groupby(['dc_id', 'first'])[df_2017.columns].sum()
df_2017_promo_store_class = promo_2017_6.reset_index()
df_2017_promo_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_promo_store_class_index = df_2017_promo_store_class[['dc_id', 'first']]
df_2017_promo_store_class_6 = df_2017_promo_store_class.groupby(['dc_id', 'first'])[promo_2017_6.columns].sum()

# df_2017.columns = df_2017.columns.get_level_values(1)
# sku_info = sku_info.reindex(df_2017.index.get_level_values(1))
# df_2017_group_sku = df_2017.groupby('item_sku_id')[df_2017.columns].sum()
prom_2017_group_sku_10 = promo_2017_10.groupby('item_sku_id')[promo_2017_10.columns].sum()
df_2017_store_class = df_2017.reset_index()
df_2017_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_store_class_index = df_2017_store_class[['dc_id', 'first']]
df_2017_store_class = df_2017_store_class.groupby(['dc_id', 'first'])[df_2017.columns].sum()
df_2017_promo_store_class = promo_2017_10.reset_index()
df_2017_promo_store_class['first'] = sku_info['item_first_cate_cd'].values
df_2017_promo_store_class_index = df_2017_promo_store_class[['dc_id', 'first']]
df_2017_promo_store_class_10 = df_2017_promo_store_class.groupby(['dc_id', 'first'])[promo_2017_10.columns].sum()

df_2017_ven.columns = pd.date_range(end='2018-02-28', periods=df_2017_ven.shape[1], freq='D')
df_2017.columns = pd.date_range(end='2018-02-28', periods=df_2017.shape[1], freq='D')
df_2017_price.columns = pd.date_range(end='2018-02-28', periods=df_2017_price.shape[1], freq='D')
# df_2017_promo_store_class_1.columns=pd.date_range(end='2018-02-28',periods = df_2017_promo_store_class_1.shape[1],freq = 'D')
# df_2017_promo_store_class_4.columns=pd.date_range(end='2018-02-28',periods = df_2017_promo_store_class_4.shape[1],freq = 'D')
# df_2017_promo_store_class_6.columns=pd.date_range(end='2018-02-28',periods = df_2017_promo_store_class_6.shape[1],freq = 'D')
# df_2017_promo_store_class_10.columns=pd.date_range(end='2018-02-28',periods = df_2017_promo_store_class_10.shape[1],freq = 'D')

# prom_2017_group_sku_1.columns=pd.date_range(end='2018-02-28',periods = prom_2017_group_sku_1.shape[1],freq = 'D')
# prom_2017_group_sku_4.columns=pd.date_range(end='2018-02-28',periods = prom_2017_group_sku_4.shape[1],freq = 'D')
# prom_2017_group_sku_6.columns=pd.date_range(end='2018-02-28',periods = prom_2017_group_sku_6.shape[1],freq = 'D')
# prom_2017_group_sku_10.columns=pd.date_range(end='2018-02-28',periods = prom_2017_group_sku_10.shape[1],freq = 'D')
df_2017_store_class.columns = pd.date_range(end='2018-02-28', periods=df_2017_store_class.shape[1], freq='D')

df_2017_group_sku.columns = pd.date_range(end='2018-02-28', periods=df_2017_group_sku.shape[1], freq='D')


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


def prepare_prom(df, promo_df, t2017, is_train=True, name_prefix=None):
    time_index = [3, 7, 14, 31, 61, 90, 140, 180, 300]
    jdata_period = 61
    X = {}
    for i in time_index:
        tmp1 = get_timespan(df, t2017, i, i)
        tmp2 = (get_timespan(promo_df, t2017, i, i) > 0) * 1
        X['has_promo_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values
        X['has_promo_mean_%s_decay' % i] = (tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(
            axis=1).values
        X['no_promo_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values
        X['no_promo_mean_%s_decay' % i] = (
                    tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
    for i in time_index:
        tmp = get_timespan(promo_df, t2017, i, i)
        X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        # X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        # X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values
    tmp = get_timespan(promo_df, t2017 + timedelta(days=jdata_period), jdata_period - 1, jdata_period - 1)
    X['has_promo_days_in_after_jdata_period-1_days'] = (tmp > 0).sum(axis=1).values
    # X['last_has_promo_day_in_after_jdata_period-1_days'] = i - ((tmp > 0) * np.arange(jdata_period-1)).max(axis=1).values
    # X['first_has_promo_day_in_after_jdata_period-1_days'] = ((tmp > 0) * np.arange(jdata_period-1, 0, -1)).max(axis=1).values
    # for i in range(-jdata_period, jdata_period):
    # X["promo_{}".format(i)] = promo_df[t2017 + timedelta(days=i)].values.astype(np.uint8)
    X = pd.DataFrame(X)
    if is_train:
        y = df[
            pd.date_range(t2017 + timedelta(31), periods=30)
        ].values
        y = np.mean(y, axis=1)
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


def prepare_quantity(df, t2017, is_train=True, name_prefix=None):
    time_index = [3, 7, 14, 30, 31, 60, 61, 70, 90, 140, 180, 300]
    jdata_period = 61
    X = {}
    for i in time_index:
        tmp = get_timespan(df, t2017, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values
    for i in time_index:
        tmp = get_timespan(df, t2017, i, i)
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
    # X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values#
    # X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values
    for i in range(1, jdata_period):  #
        X['day_%s_2017' % i] = get_timespan(df, t2017, i, 1).values.ravel()
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df, t2017, 28 - i, 4, freq='7D').mean(axis=1).values
        # X['mean_20_dow{}_2017'.format(i)] = get_timespan(df, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        X['mean_10_dow{}_2017'.format(i)] = get_timespan(df, t2017, 70 - i, 10, freq='7D').mean(axis=1).values
    X = pd.DataFrame(X)
    if is_train:
        y = df[
            pd.date_range(t2017 + timedelta(31), periods=30)
        ].values
        y = np.mean(y, axis=1)
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


def prepare_price(df, t2017, is_train=True, name_prefix=None):
    X = {}
    time_index = [3, 7, 14, 30, 60, 90, 140, 180, 300]
    jdata_period = 61
    for i in time_index:
        tmpp = get_timespan(df, t2017, i, i)
        # X['price_mean_%s_decay' % i] = (tmpp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['price_mean_%s' % i] = tmpp.mean(axis=1).values
        X['price_median_%s' % i] = tmpp.median(axis=1).values
        X['price_min_%s' % i] = tmpp.min(axis=1).values
        X['price_max_%s' % i] = tmpp.max(axis=1).values
        X['price_std_%s' % i] = tmpp.std(axis=1).values
    X = pd.DataFrame(X)
    if is_train:
        y = df[
            pd.date_range(t2017 + timedelta(31), periods=30)
        ].values
        y = np.mean(y, axis=1)
        return X
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


def prepare_discount(df, t2017, is_train=True, name_prefix=None):
    X = {}
    time_index = [3, 7, 14, 30, 60, 90, 140, 180, 300]
    jdata_period = 61
    for i in time_index:
        tmppp = get_timespan(df, t2017, i, i)
        # X['discount_mean_%s_decay' % i] = (tmppp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['discount_mean_%s' % i] = tmppp.mean(axis=1).values
        # X['discount_median_%s' % i] = tmppp.median(axis=1).values
        X['discount_min_%s' % i] = tmppp.min(axis=1).values
        X['discount_max_%s' % i] = tmppp.max(axis=1).values
        X['discount_std_%s' % i] = tmppp.std(axis=1).values
    X = pd.DataFrame(X)
    if is_train:
        y = df[
            pd.date_range(t2017 + timedelta(31), periods=30)
        ].values
        y = np.mean(y, axis=1)
        return X
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


print("Preparing dataset...")
t2017 = date(2017, 10, 26)
num_days = 60
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=1 * i)
    X_tmp1, y_tmp = prepare_quantity(df_2017, t2017 + delta)  # 6´Î»¬¶¯·Ö±ðµÃµ½µÄÌØÕ÷
    X_tmp1_prom1 = prepare_prom(df_2017, promo_2017_1, t2017 + delta, is_train=False, name_prefix='prom1')
    X_tmp1_prom4 = prepare_prom(df_2017, promo_2017_4, t2017 + delta, is_train=False, name_prefix='prom4')
    X_tmp1_prom6 = prepare_prom(df_2017, promo_2017_6, t2017 + delta, is_train=False, name_prefix='prom6')
    X_tmp1_prom10 = prepare_prom(df_2017, promo_2017_10, t2017 + delta, is_train=False, name_prefix='prom10')
    X_tmp2 = prepare_quantity(df_2017_group_sku, t2017 + delta, is_train=False, name_prefix='sku_quantity')
    X_tmp2_prom1 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_1, t2017 + delta, is_train=False,
                                name_prefix='sku1')
    X_tmp2_prom4 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_4, t2017 + delta, is_train=False,
                                name_prefix='sku4')
    X_tmp2_prom6 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_6, t2017 + delta, is_train=False,
                                name_prefix='sku6')
    X_tmp2_prom10 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_10, t2017 + delta, is_train=False,
                                 name_prefix='sku10')
    X_tmp2.index = df_2017_group_sku.index
    X_tmp2_prom1.index = df_2017_group_sku.index
    X_tmp2_prom4.index = df_2017_group_sku.index
    X_tmp2_prom6.index = df_2017_group_sku.index
    X_tmp2_prom10.index = df_2017_group_sku.index
    X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
    X_tmp2_prom1 = X_tmp2_prom1.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
    X_tmp2_prom4 = X_tmp2_prom4.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
    X_tmp2_prom6 = X_tmp2_prom6.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
    X_tmp2_prom10 = X_tmp2_prom10.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
    X_tmp3 = prepare_quantity(df_2017_store_class, t2017 + delta, is_train=False, name_prefix='store_class')
    X_tmp3_class1 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_1, t2017 + delta, is_train=False,
                                 name_prefix='store_class_1')
    X_tmp3_class4 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_4, t2017 + delta, is_train=False,
                                 name_prefix='store_class_4')
    X_tmp3_class6 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_6, t2017 + delta, is_train=False,
                                 name_prefix='store_class_6')
    X_tmp3_class10 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_10, t2017 + delta, is_train=False,
                                  name_prefix='store_class_10')
    X_tmp3.index = df_2017_store_class.index
    X_tmp3_class1.index = df_2017_store_class.index
    X_tmp3_class4.index = df_2017_store_class.index
    X_tmp3_class6.index = df_2017_store_class.index
    X_tmp3_class10.index = df_2017_store_class.index
    X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)
    X_tmp3_class1 = X_tmp3_class1.reindex(df_2017_store_class_index).reset_index(drop=True)
    X_tmp3_class4 = X_tmp3_class4.reindex(df_2017_store_class_index).reset_index(drop=True)
    X_tmp3_class6 = X_tmp3_class6.reindex(df_2017_store_class_index).reset_index(drop=True)
    X_tmp3_class10 = X_tmp3_class10.reindex(df_2017_store_class_index).reset_index(drop=True)
    # X_tmp4 = prepare_ven(df_2017_ven,t2017 + delta, is_train=False, name_prefix='ven')
    X_tmp5 = prepare_price(price_2017, t2017 + delta, is_train=False, name_prefix='price')
    X_tmp6 = prepare_discount(discount_2017, t2017 + delta, is_train=False, name_prefix='discount')
    X_tmp = pd.concat(
        [X_tmp1, X_tmp1_prom1, X_tmp1_prom4, X_tmp1_prom6, X_tmp1_prom10, X_tmp2, X_tmp2_prom1, X_tmp2_prom4,
         X_tmp2_prom6, X_tmp2_prom10, X_tmp3, X_tmp3_class1, X_tmp3_class4, X_tmp3_class6, X_tmp3_class10, X_tmp5,
         X_tmp6, sku_info.reset_index(), DC_id['dc_id']], axis=1)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp2
    gc.collect()

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
print(X_train.shape)
print(y_train.shape)

t2017_val = date(2017, 12, 28)

# X_val, y_val = prepare_dataset(df_2017, promo_2017, t2017_val)
X_val1, y_val = prepare_quantity(df_2017, t2017_val)
X_val1_prom1 = prepare_prom(df_2017, promo_2017_1, t2017_val, is_train=False, name_prefix='prom1')
X_val1_prom4 = prepare_prom(df_2017, promo_2017_4, t2017_val, is_train=False, name_prefix='prom4')
X_val1_prom6 = prepare_prom(df_2017, promo_2017_6, t2017_val, is_train=False, name_prefix='prom6')
X_val1_prom10 = prepare_prom(df_2017, promo_2017_10, t2017_val, is_train=False, name_prefix='prom10')

X_val2 = prepare_quantity(df_2017_group_sku, t2017_val, is_train=False, name_prefix='sku_quantity')
X_val2_prom1 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_1, t2017_val, is_train=False, name_prefix='sku1')
X_val2_prom4 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_4, t2017_val, is_train=False, name_prefix='sku4')
X_val2_prom6 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_6, t2017_val, is_train=False, name_prefix='sku6')
X_val2_prom10 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_10, t2017_val, is_train=False, name_prefix='sku10')

X_val2.index = df_2017_group_sku.index
X_val2_prom1.index = df_2017_group_sku.index
X_val2_prom4.index = df_2017_group_sku.index
X_val2_prom6.index = df_2017_group_sku.index
X_val2_prom10.index = df_2017_group_sku.index

X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_val2_prom1 = X_val2_prom1.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_val2_prom4 = X_val2_prom4.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_val2_prom6 = X_val2_prom6.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_val2_prom10 = X_val2_prom10.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #

X_val3 = prepare_quantity(df_2017_store_class, t2017_val, is_train=False, name_prefix='store_class')
X_val3_class1 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_1, t2017_val, is_train=False,
                             name_prefix='store_class_1')
X_val3_class4 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_4, t2017_val, is_train=False,
                             name_prefix='store_class_4')
X_val3_class6 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_6, t2017_val, is_train=False,
                             name_prefix='store_class_6')
X_val3_class10 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_10, t2017_val, is_train=False,
                              name_prefix='store_class_10')

X_val3.index = df_2017_store_class.index
X_val3_class1.index = df_2017_store_class.index
X_val3_class4.index = df_2017_store_class.index
X_val3_class6.index = df_2017_store_class.index
X_val3_class10.index = df_2017_store_class.index

X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)
X_val3_class1 = X_val3_class1.reindex(df_2017_store_class_index).reset_index(drop=True)
X_val3_class4 = X_val3_class4.reindex(df_2017_store_class_index).reset_index(drop=True)
X_val3_class6 = X_val3_class6.reindex(df_2017_store_class_index).reset_index(drop=True)
X_val3_class10 = X_val3_class10.reindex(df_2017_store_class_index).reset_index(drop=True)

# X_val4 = prepare_ven(df_2017_ven,t2017_val, is_train=False, name_prefix='ven')

X_val5 = prepare_price(price_2017, t2017_val, is_train=False, name_prefix='price')

X_val6 = prepare_discount(discount_2017, t2017_val, is_train=False, name_prefix='discount')

X_val = pd.concat(
    [X_val1, X_val1_prom1, X_val1_prom4, X_val1_prom6, X_val1_prom10, X_val2, X_val2_prom1, X_val2_prom4, X_val2_prom6,
     X_val2_prom10, X_val3, X_val3_class1, X_val3_class4, X_val3_class6, X_val3_class10, X_val5, X_val6,
     sku_info.reset_index(), DC_id['dc_id']], axis=1)

print(X_val.shape)
print(y_val.shape)

t2017_test = date(2018, 3, 1)

# X_test = prepare_dataset(df_2017, promo_2017, t2017_test)
X_test1 = prepare_quantity(df_2017, t2017_test, is_train=False)
X_test1_prom1 = prepare_prom(df_2017, promo_2017_1, t2017_test, is_train=False, name_prefix='prom1')
X_test1_prom4 = prepare_prom(df_2017, promo_2017_4, t2017_test, is_train=False, name_prefix='prom4')
X_test1_prom6 = prepare_prom(df_2017, promo_2017_6, t2017_test, is_train=False, name_prefix='prom6')
X_test1_prom10 = prepare_prom(df_2017, promo_2017_10, t2017_test, is_train=False, name_prefix='prom10')

X_test2 = prepare_quantity(df_2017_group_sku, t2017_test, is_train=False, name_prefix='sku_quantity')
X_test2_prom1 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_1, t2017_test, is_train=False, name_prefix='sku1')
X_test2_prom4 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_4, t2017_test, is_train=False, name_prefix='sku4')
X_test2_prom6 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_6, t2017_test, is_train=False, name_prefix='sku6')
X_test2_prom10 = prepare_prom(df_2017_group_sku, prom_2017_group_sku_10, t2017_test, is_train=False,
                              name_prefix='sku10')

X_test2.index = df_2017_group_sku.index
X_test2_prom1.index = df_2017_group_sku.index
X_test2_prom4.index = df_2017_group_sku.index
X_test2_prom6.index = df_2017_group_sku.index
X_test2_prom10.index = df_2017_group_sku.index

X_test2 = X_test2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_test2_prom1 = X_test2_prom1.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_test2_prom4 = X_test2_prom4.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_test2_prom6 = X_test2_prom6.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #
X_test2_prom10 = X_test2_prom10.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)  #

X_test3 = prepare_quantity(df_2017_store_class, t2017_test, is_train=False, name_prefix='store_class')
X_test3_class1 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_1, t2017_test, is_train=False,
                              name_prefix='store_class_1')
X_test3_class4 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_4, t2017_test, is_train=False,
                              name_prefix='store_class_4')
X_test3_class6 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_6, t2017_test, is_train=False,
                              name_prefix='store_class_6')
X_test3_class10 = prepare_prom(df_2017_store_class, df_2017_promo_store_class_10, t2017_test, is_train=False,
                               name_prefix='store_class_10')

X_test3.index = df_2017_store_class.index
X_test3_class1.index = df_2017_store_class.index
X_test3_class4.index = df_2017_store_class.index
X_test3_class6.index = df_2017_store_class.index
X_test3_class10.index = df_2017_store_class.index

X_test3 = X_test3.reindex(df_2017_store_class_index).reset_index(drop=True)
X_test3_class1 = X_test3_class1.reindex(df_2017_store_class_index).reset_index(drop=True)
X_test3_class4 = X_test3_class4.reindex(df_2017_store_class_index).reset_index(drop=True)
X_test3_class6 = X_test3_class6.reindex(df_2017_store_class_index).reset_index(drop=True)
X_test3_class10 = X_test3_class10.reindex(df_2017_store_class_index).reset_index(drop=True)

# X_test4 = prepare_ven(df_2017_ven,t2017_test, is_train=False, name_prefix='ven')

X_test5 = prepare_price(price_2017, t2017_test, is_train=False, name_prefix='price')

X_test6 = prepare_discount(discount_2017, t2017_test, is_train=False, name_prefix='discount')

X_test = pd.concat(
    [X_test1, X_test1_prom1, X_test1_prom4, X_test1_prom6, X_test1_prom10, X_test2, X_test2_prom1, X_test2_prom4,
     X_test2_prom6, X_test2_prom10, X_test3, X_test3_class1, X_test3_class4, X_test3_class6, X_test3_class10, X_test5,
     X_test6, sku_info.reset_index(), DC_id['dc_id']], axis=1)

print(X_test.shape)

print("Training and predicting models...")

MAX_ROUNDS = 10000
val_pred = []
test_pred = []
cate_vars = []
importance = []

print("Training and predicting models...")
params = {
    'num_leaves': 70,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 40
}

MAX_ROUNDS = 5000

val_pred = []
test_pred = []
cate_vars = []
importance = []
for i in range(1):
    print("=" * 50)
    print("Step %d" % (i + 1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cate_vars,
        # weight=pd.concat([items["perishable"]] * num_days) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain,
        # weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=50
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    importance.append(bst.feature_importance("gain").reshape(bst.feature_importance("gain").shape[0], 1))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

print("Validation mse:", mean_squared_error(y_val, np.array(val_pred).transpose()))

# weight = items["perishable"] * 0.25 + 1
err = (y_val - np.array(val_pred).transpose()) ** 2
err = err.sum(axis=1)
err = np.sqrt(err.sum() / 61)
print('nwrmsle = {}'.format(err))

y_val = np.array(val_pred).transpose()
df_preds = pd.DataFrame(
    y_val, index=df_2017.index,
    columns=pd.date_range("2018-12-28", periods=1)
).stack().to_frame("quantity")
df_preds.index.set_names(["dc_id", "item_sku_id", "date"], inplace=True)
# df_preds["quantity"] = np.clip(np.expm1(df_preds["quantity"]), 0, 1000)
df_preds.reset_index().to_csv('meantwo2_val-cv23.csv', index=False)

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2018-03-01", periods=1)
).stack().to_frame("quantity")
df_preds.index.set_names(["dc_id", "item_sku_id", "date"], inplace=True)
# df_preds["quantity"] = np.clip(np.expm1(df_preds["quantity"]), 0, 1000)
df_preds.reset_index().to_csv('meantwo2_test23_cv.csv', index=False)

