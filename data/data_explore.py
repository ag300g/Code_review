#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
__author__ = 'ag300g'

import pandas as pd
pd.set_option('display.max_columns', 10)  ## 最多可以显示10列


sku_info = pd.read_csv('data/sku_info.csv')
sku_info.shape
sku_info.columns.tolist()
sku_info.head(2)



sku_attr = pd.read_csv('data/sku_attr.csv')
sku_attr.shape
sku_attr.columns.tolist()
sku_attr.head(2)





sku_prom = pd.read_csv('data/sku_prom.csv')
sku_prom.shape
sku_prom.columns.tolist()
sku_prom.head(2)






sku_discount_testing_2018MarApr = pd.read_csv('data/sku_discount_testing_2018MarApr.csv')
sku_discount_testing_2018MarApr.shape
sku_discount_testing_2018MarApr.columns.tolist()
sku_discount_testing_2018MarApr.head(2)




sku_prom_testing_2018MarApr = pd.read_csv('data/sku_prom_testing_2018MarApr.csv')
sku_prom_testing_2018MarApr.shape
sku_prom_testing_2018MarApr.columns.tolist()
sku_prom_testing_2018MarApr.head(2)





initial_inventory = pd.read_csv('data/initial_inventory.csv')
initial_inventory.shape
initial_inventory.columns.tolist()
initial_inventory.head(2)



sku_cost = pd.read_csv('data/sku_cost.csv')
sku_cost.shape
sku_cost.columns.tolist()
sku_cost.head(2)
