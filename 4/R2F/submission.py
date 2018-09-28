#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
Sample submission for 2nd round competition.

'''

import pandas as pd
import numpy as np


# import all modules been used

class UserPolicy:
    def __init__(self, initial_inventory, sku_cost):
        self.inv = initial_inventory
        self.costs = sku_cost
        self.extra_shipping_cost_per_unit = 0.01
        self.fixed_replenish_cost_per_order = 0.1
        self.sku_limit = np.asarray([300, 300, 300, 300, 300])
        self.capacity_limit = np.asarray([4000, 5000, 6000, 2000, 1000])
        self.abandon_rate = np.asarray([1. / 100, 7. / 100, 10. / 100, 9. / 100, 8. / 100])
        self.decision_record = []

    def daily_decision(self, t):
        '''
        daily decision of inventory allocation
        input values:
            t, decision date
        return values:
            inventory decision, 2-D numpy array, shape (6,1000), type integer
        '''
        # Your algorithms here
        # simple rule: no replenishment or transshipment
        capacity_limit = self.capacity_limit
        abandon_rate = self.abandon_rate
        df_inv = self.inv.sort_values(['dc_id', 'item_sku_id'])
        dc_list = list(range(6))

        array_inv = np.asarray(
            [df_inv.loc[df_inv.dc_id == dcid].stock_quantity.values for dcid in dc_list])  # 将初始库存数据整理成6*1000矩阵
        sort_abandon_rate = np.argsort(-self.abandon_rate)  # 对abandon rate按升序排列
        costs = self.costs.set_index(['item_sku_id'])
        costs = costs.sort_index().values
        stockout_cost = costs[:, 0]
        holding_cost = costs[:, 1]

        mean_history = pd.read_csv("mean_test_cv10.csv").set_index(['dc_id', 'item_sku_id', 'date'])
        mean_history = mean_history[["quantity"]].unstack(level=-1)
        mean_history = mean_history.values  # 读入预测的销量均值数据，并将其整理成6000*1矩阵，此数据后续将用于对我们预测的销量数据进行修正

        mean_history1 = pd.read_csv("meantwo_test_cv29.csv").set_index(['dc_id', 'item_sku_id', 'date'])
        mean_history1 = mean_history1[["quantity"]].unstack(level=-1)
        mean_history1 = mean_history1.values  # 读入预测的前31天的销量均值，并将其整理成6000*1矩阵，此数据后续将用于对我们预测的销量数据进行修正

        mean_history2 = pd.read_csv("meantwo2_test_cv23.csv").set_index(['dc_id', 'item_sku_id', 'date'])
        mean_history2 = mean_history2[["quantity"]].unstack(level=-1)
        mean_history2 = mean_history2.values  # 读入预测的后30天的销量均值，并将其整理成6000*1矩阵，此数据后续将用于对我们预测的销量数据进行修正

        sku_demand = pd.read_csv("lgb_test_cv049.csv").set_index(['dc_id', 'item_sku_id', 'date'])
        sku_demand = sku_demand[["quantity"]].unstack(level=-1)
        sku_demand = sku_demand.values
        array_sku_mean = sku_demand.reshape(6, 1000, 61)  # 读入预测的61天的销量数据，并将其整理成6*1000*61的矩阵，后续会对该数据进行修正

        a_std = np.std(array_sku_mean, axis=2)
        a_std = a_std.reshape(6, 1000, 1)
        a_std = a_std.repeat(61, axis=2)  # 计算预测的销量数据的标准差

        std_all = pd.read_csv("std.csv").values  # 读入计算的每个商品在所有历史数据上的标准差，此数据后续将用于对我们预测的销量数据进行修正
        std_all1 = std_all[:, :] + 1 - 1
        std_abn = std_all > 100  # 将标准差>100的商品记录为std_abn
        std_all[std_all > 30] = 30  # 在std_all中将大于30的数据记录为30，因为有些商品在历史数据上波动较大，但是在我们的预测时间范围内不一定会波动很大
        std_all = std_all[:, 1:].reshape(6, 1000, 1).repeat(61, axis=2)

        std_f = np.maximum(std_all / 17, a_std)  # 取修正之后的预测数据与历史数据的较大者
        std_f[std_f > 10] = 10  # 在std_f中将大于10的数据记录为10，此数据后续将用于对我们预测的销量数据进行修正

        history = mean_history.reshape(6, 1000, 1).repeat(61, axis=2)
        history1 = mean_history1.reshape(6, 1000, 1).repeat(31, axis=2)
        history2 = mean_history2.reshape(6, 1000, 1).repeat(30, axis=2)
        history0 = np.zeros((6, 1000, 61))
        history0[:, :, :31] = np.maximum(history[:, :, :31], history1)
        history0[:, :, 31:] = np.maximum(history[:, :, 31:], history2)  # history0记录了我们预测的销量均值数据信息

        array_sku_mean = 1.55 * np.maximum(array_sku_mean, history0) + 4 * std_f  # 根据预测的均值数据以及计算得到的标准差数据对我们的销量数据进行了修正

        sort_abandon_rate = np.argsort(-abandon_rate)  # 对abandon rate排序，得到的排序会作为调拨时FDC的优先级顺序
        inv_mean = np.zeros((6, 1000))
        sku_surplus = np.zeros(1000)  # 定义RDC可调商品数量矩阵
        sku_shortage = np.zeros((5, 1000))  # 定义FDC需求商品数量矩阵

        # 开始调拨
        if t < 59:
            alpha = 3
        else:
            alpha = 62 - t  # t<=59时，我们初步考虑一次给每个FDC调拨4天的需求量,t=60初步考虑一次给每个FDC调拨3天的需求量,t=61初步考虑一次给每个FDC调拨2天的需求量
        rdc_alpha = 2  # RDC给自己留下满足当天需求的量的2倍，其余的部分是可以用于调拨的
        inventory_decision = np.zeros((6, 1000)).astype(int)  # 定义调拨矩阵

        end_day = min(t + alpha, 61)
        sku_surplus = (array_inv[0, :] - np.minimum(array_inv[0, :], rdc_alpha * array_sku_mean[0, :, t - 1])).astype(
            int)  # 每天给RDC留下满足自己的需求量的2倍之后剩余的数量，这一部分是可以用来给FDC调拨的量
        inv_mean = np.minimum(array_inv[:, :], np.sum(array_sku_mean[:, :, t - 1:end_day], axis=2))
        sku_shortage = np.rint(np.sum(array_sku_mean[1:, :, t - 1:end_day], axis=2) - inv_mean[1:,
                                                                                      :])  # 计算在现有库存量的基础上，若要满足我们初步考虑的FDC的需求量，每个FDC需要RDC调拨的商品数量

        not_sat = sku_surplus - np.sum(sku_shortage, axis=0) < 0  # 记录下RDC中可调拨量不能满足5个FDC缺货量之和的SKU
        sku_shortage[:, not_sat] = sku_shortage[:, not_sat] * np.tile(
            sku_surplus[not_sat] / np.sum(sku_shortage[:, not_sat], axis=0),
            (5, 1))  # 对于RDC中可调拨量足以满足5个FDC缺货量之和的SKU，按照FDC缺货量进行调拨；对于RDC中可调拨量不能满足5个FDC缺货量之和的SKU，按照缺货比例进行调拨
        # 调整shortage，万一库存中无货可以

        sku_shortage_4day = np.rint(
            np.sum(array_sku_mean[1:, :, t - 1:t + 3], axis=2) - inv_mean[1:, :])  # case1:对于每个FDC每个SKU考虑4天的需求
        not_sat = sku_surplus - np.sum(sku_shortage_4day, axis=0) < 0
        sku_shortage[:, not_sat] = sku_shortage_4day[:, not_sat] * np.tile(
            sku_surplus[not_sat] / np.sum(sku_shortage_4day[:, not_sat], axis=0), (5, 1))

        sku_shortage_3day = np.rint(
            np.sum(array_sku_mean[1:, :, t - 1:t + 2], axis=2) - inv_mean[1:, :])  # case1:对于每个FDC每个SKU考虑3天的需求
        sku_shortage_3day[sku_shortage_3day < 0] = 0
        not_sat = sku_surplus - np.sum(sku_shortage_3day, axis=0) < 0
        sku_shortage[:, not_sat] = sku_shortage_3day[:, not_sat] * np.tile(
            sku_surplus[not_sat] / np.sum(sku_shortage_3day[:, not_sat], axis=0), (5, 1))

        sku_shortage_2day = np.rint(
            np.sum(array_sku_mean[1:, :, t - 1:t + 1], axis=2) - inv_mean[1:, :])  # case1:对于每个FDC每个SKU考虑2天的需求
        sku_shortage_2day[sku_shortage_2day < 0] = 0
        not_sat = sku_surplus - np.sum(sku_shortage_2day, axis=0) < 0
        sku_shortage[:, not_sat] = sku_shortage_2day[:, not_sat] * np.tile(
            sku_surplus[not_sat] / np.sum(sku_shortage_2day[:, not_sat], axis=0), (5, 1))

        sku_shortage_1day = np.rint(array_sku_mean[1:, :, t - 1] - inv_mean[1:, :])  # case1:对于每个FDC每个SKU考虑1天的需求
        sku_shortage_1day[sku_shortage_1day < 0] = 0
        not_sat = sku_surplus - np.sum(sku_shortage_1day, axis=0) < 0
        sku_shortage[:, not_sat] = sku_shortage_1day[:, not_sat]

        # 开始按照我们的排序进行调拨
        for i in sort_abandon_rate:  # 按照abandon_rate由高到低的顺序进行调货
            sku_shortage[i, :] = np.minimum(sku_shortage[i, :], sku_surplus)

            sku_shortage_1day = np.minimum(sku_shortage, sku_shortage_1day)
            sku_shortage_2day = np.minimum(sku_shortage, sku_shortage_2day)

            importance = (sku_shortage > 0) * (stockout_cost.reshape(1000, 1).repeat(5, axis=1).T) + \
                         7.5 * (stockout_cost.reshape(1000, 1).repeat(5, axis=1).T) * sku_shortage_1day + \
                         0.05 * sku_shortage_2day  # 按照缺货量以及缺货成本计算每个FDC每个SKU的importance
            importance[i, :] = importance[i, :] * (sku_shortage[i, :] > 0)

            sort_importance = np.argsort(-importance[i, :])  # 对FDCi每种商品的重要性进行降序排列，这个顺序就是我们调拨的顺序
            sku_cum = sku_shortage[i, sort_importance].cumsum()  # 对FDCi按照sort_importance进行商品数量的累加
            cum_more_than_cap = sku_cum <= capacity_limit[i]  # 记录是否超过FDCi可调拨商品总量限制
            kind_limit = min(sum(cum_more_than_cap), 300)  # 记录在不超过FDCi商品总量限制以及商品种类限制的条件下能调拨的商品种类
            inventory_decision[i + 1, sort_importance[:kind_limit]] = sku_shortage[
                i, sort_importance[:kind_limit]]  # 按照商品的shortage进行调拨
            sku_surplus = sku_surplus - inventory_decision[i + 1, :]  # 更新RDC可调拨商品数量

        importance = 0.001 * stockout_cost - holding_cost  # 按照缺货成本以及持有成本计算SKU的importance
        sort_importance = np.argsort(-importance).astype(int)  # 按照importance对商品进行排序

        if t < 55:

            demand_mean_3 = np.sum(array_sku_mean[:, :, t + 6:t + 7], axis=2)  # 计算RDC以及5个FDC在第t+6天对每个商品的需求总量

            demand_mean_hi = np.sum(array_sku_mean[:, :, t + 6:t + 9], axis=2)  # 计算RDC以及5个FDC在第t+6，t+7,t+8天对每个商品的需求总量
            demand_mean_hi1 = np.sum(array_sku_mean[:, :, t + 6:t + 10],
                                     axis=2)  # 计算RDC以及5个FDC在第t+6，t+7,t+8,t+9天对每个商品的需求总量
            demand_mean_hi2 = np.sum(array_sku_mean[:, :, t + 6:t + 11],
                                     axis=2)  # 计算RDC以及5个FDC在第t+6，t+7,t+8,t+9,t+10天对每个商品的需求总量

            demand_mean_3[:, sort_importance[:900]] = demand_mean_hi[:,
                                                      sort_importance[:900]]  # 对于优先级位于800-900的商品我们考虑补够3天的需求量
            demand_mean_3[:, sort_importance[:800]] = demand_mean_hi1[:,
                                                      sort_importance[:800]]  # 对于优先级位于400-800的商品我们考虑补够4天的需求量
            demand_mean_3[:, sort_importance[:400]] = demand_mean_hi2[:,
                                                      sort_importance[:400]]  # 对于优先级最高的400种商品我们考虑补够5天的需求量

            demand_mean_3[std_abn[:, 1:]] = demand_mean_3[std_abn[:, 1:]] + std_all1[:, 1:][
                std_abn[:, 1:]] / 10  # 对于历史上标准差>100的商品的补货量进行调整，考虑对这些商品多补一些

            demand_whole_3 = np.sum(demand_mean_3, axis=0)  # 6个dc需求之和
            demand_mean_7 = np.sum(array_sku_mean[:, :, t - 1:t + 6], axis=2)  # 7天需求总量6*1000

            demand_surplus = array_inv - demand_mean_7
            left = demand_surplus[1:, :] + 1 - 1
            left[left < 0] = 0  # 记录FDC中已有的库存是否能够满足未来7天的需求，不够的记为0
            demand_surplus[1:, :][demand_surplus[1:, :] > 0] = 0  # 若FDC中已有的库存能够满足未来7天的需求则记为0

            if t > 1:
                start_time = max(t - 1 - 7, 0)
                end_time = t
                trans_sum = np.zeros((1, 1000))
                for i in range(start_time, end_time):
                    trans_sum = trans_sum + self.decision_record[i - 1][0, :]  # 过去‘8’天中的补货总量
                demand_surplus_7 = trans_sum + np.sum(demand_surplus, axis=0)
            else:
                demand_surplus_7 = np.sum(demand_surplus, axis=0)

            if t <= 4:
                demand_mean_3 = np.sum(array_sku_mean[:, :, t + 6:t + 8], axis=2)  # 前4天考虑补2天的需求量
                demand_mean_3[std_abn[:, 1:]] = demand_mean_3[std_abn[:, 1:]] + std_all1[:, 1:][std_abn[:, 1:]] / 7
                # demand_mean_7=np.sum(array_sku_mean[:,:,t-1:t+6],axis=2)#7天需求总量

            demand_mean_3[1:, :] = demand_mean_3[1:, :] - left
            demand_mean_3[demand_mean_3 < 0] = 0
            demand_whole_3 = np.sum(demand_mean_3, axis=0)  # 6个dc需求之和
            demand_surplus_7[demand_surplus_7 < 0] = 0
            inventory_decision[0, :] = np.rint(demand_whole_3 - np.minimum(demand_whole_3, demand_surplus_7))

        self.decision_record.append(inventory_decision)
        return inventory_decision

    def info_update(self, end_day_inventory, t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv = end_day_inventory

    def some_other_functions():
        pass
