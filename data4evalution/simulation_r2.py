#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
Submission verification code for GOC 2nd round.
Need initial_inventory.csv and sku_cost.csv in the same directory. 
Run as: python3 simulation_r2.py
'''

import pandas as pd
import numpy as np

class DailyEvaluate:
    def __init__(self, sku_list, dc_list, period_length, 
            initial_inventory, sku_limit, capacity_limit, 
            abandon_rate, sku_cost, extra_shipping_cost_per_unit, 
            replenishment_leadtime, fixed_replenish_cost_per_order, 
            true_demand):

        self.sku_list = sku_list
        self.dc_list = dc_list
        self.period_length = period_length

        '''transform initial_inventory into numpy array, shape (6,1000) '''
        initial_inventory.sort_values(['dc_id','item_sku_id'], ascending = [True, True], inplace =True)
        self.inv = initial_inventory.stock_quantity.values.reshape((len(dc_list),len(sku_list)))

        '''inventory_replenishment, numpy array, shape (period_length, 1000) '''
        self.intransit_replenishment = np.zeros((period_length, len(sku_list))).astype(int)

        '''inventory transshipment, numpy array, shape (5, 1000) '''
        self.intransit_transshipment = np.zeros((len(dc_list) -1 , len(sku_list))).astype(int)

        self.sku_limit = sku_limit
        self.capacity_limit = capacity_limit
        self.abandon_rate = abandon_rate

        sku_cost.sort_values('item_sku_id', ascending = True, inplace = True)
        self.sku_stockout_cost = sku_cost.stockout_cost.values
        self.sku_holding_cost = sku_cost.holding_cost.values

        self.extra_shipping_cost_per_unit = extra_shipping_cost_per_unit
        self.fixed_replenish_cost_per_order = fixed_replenish_cost_per_order
        self.replenishment_leadtime = replenishment_leadtime

        self.true_demand = true_demand
        self.t = 0

        self.ordering_cost = 0
        self.shortage_cost = 0
        self.holding_cost = 0
        self.extra_shipping_cost = 0 


    def daily_update(self, inventory_decision, t):
        '''
        Daily demand fulfillment and inventory status update.
        input values:
            self, inventory_decsion(replenishment and transshipment), date 
        return values:
            daily_update status, True/False
        '''

        # constraint check
        try:
            # decision format: shape (number of DCs, number of SKUs)
            if not inventory_decision.shape == (len(self.dc_list), len(self.sku_list)):
                raise Exception('invalid decision format')

            # inventory decision should be nonnegative integers
            if not np.all(inventory_decision >= 0):
                raise Exception('negative replenish_quantity or transship_quantity')

            if not np.all(np.ceil(inventory_decision) == np.floor(inventory_decision)):
                raise Exception('integrity constraint violation')

            inventory_decision = inventory_decision.astype(int)
            new_replenishment = inventory_decision[0]
            new_transshipment = inventory_decision[1:]

            # constraint on transship sku variety for each fdc
            if not np.all(np.count_nonzero(new_transshipment, axis = 1) <= self.sku_limit):
                raise Exception('transship sku limits violation')

            # constraint on transship capacity for each fdc
            if not np.all(new_transshipment.sum(axis = 1) <= self.capacity_limit):
                raise Exception('transship capacity limits violation')

            # update rdc on hand inventory, and intransit transship inventory according to new_transshipment
            self.inv[0] -= new_transshipment.sum(axis =0)
            # the rdc inventory should be nonnegative after transshipment
            if not np.all(self.inv[0] >= 0):
                raise Exception('transshipment should not be more than RDC inventory')
            self.intransit_transshipment = new_transshipment

        except Exception as e:
            # invalid decision
            return {'status':False, 'constraints_violation':e}

        # demand realization, record leftover inventory and corresponding costs    
        # fdc local inventory only fulfills local demand
        _fdc_sales = np.minimum(self.true_demand[t][1:], self.inv[1:])

        # spillover demand is firstly fulfilled by intransit transshipment to that FDC
        _spill_over = np.floor(((self.true_demand[t][1:] - _fdc_sales).T * (1 - self.abandon_rate)).T)
        _transship_sales = np.minimum(_spill_over, self.intransit_transshipment) 
        _spill_over -= _transship_sales

        # rdc inventory fulfills rdc demand and fdc spillover demand 
        _rdc_tot_demand = self.true_demand[t][0] + _spill_over.sum(axis=0)
        _rdc_sales = np.minimum(_rdc_tot_demand, self.inv[0])

        # deduct inventory that has been consumed
        self.inv[1:] -= _fdc_sales.astype(int)
        self.intransit_transshipment -= _transship_sales.astype(int)
        self.inv[0] -= _rdc_sales.astype(int)

        # calculate inventory cost
        if t < self.period_length - self.replenishment_leadtime:
            self.ordering_cost += np.count_nonzero(new_replenishment) * self.fixed_replenish_cost_per_order
            # update in-transit replenishment inventory
            self.intransit_replenishment[t+self.replenishment_leadtime-1] = new_replenishment

        self.holding_cost += (self.inv * self.sku_holding_cost).sum() + (self.intransit_transshipment * self.sku_holding_cost).sum()
        _lost_sale = self.true_demand[t].sum(axis=0) - (_fdc_sales.sum(axis=0) + _transship_sales.sum(axis = 0) + _rdc_sales)
        self.shortage_cost += (_lost_sale * self.sku_stockout_cost).sum()
        self.extra_shipping_cost += self.extra_shipping_cost_per_unit * np.maximum(_rdc_sales - self.true_demand[t][0], np.zeros(len(self.sku_list)).astype(int) ).sum()

        # update on hand inventory, receive intransit replenishment and transshipment
        self.inv[0] += self.intransit_replenishment[t]
        self.inv[1:] += self.intransit_transshipment
        
        return {'status':True}


if __name__ == '__main__': 
    # contestant's submission
    from submission import UserPolicy

    # SKU types
    sku_list  = [i for i in range(1, 1001)]
    
    # DC_ID, the RDC ID is 0, the FDC IDs are from 1 to 5
    dc_list  = list(range(6))

    # simulation period length, 31 in Test Set A and 61 in Test Set B
    period_length = 61

    # ordering for each sku replenishment order  
    fixed_replenish_cost_per_order = 0.1 
    
    replenishment_leadtime = 7

    # initial on hand inventory
    initial_inventory = pd.read_csv('initial_inventory.csv')

    # maximum number of unique products that can be allocated to FDC 
    sku_limit = np.asarray([300, 300, 300, 300, 300])

    # maximum total number of units that can be allocated to FDC
    capacity_limit = np.asarray([4000, 5000, 6000, 2000, 1000])

    # ratio of customers that abandon purchase due to local FDC stock out
    abandon_rate =np.asarray([9./100, 1./100, 7./100, 8./100, 10./100])

    # rdc extra_shipping cost
    extra_shipping_cost_per_unit = 0.01

    # per unit shortage cost and daily holding cost 
    sku_cost = pd.read_csv('sku_cost.csv')
    
    '''FAKE DATA, RANDOMLY GENERATED'''
    # true_demand = np.random.randint(100, size = (period_length, len(dc_list), len(sku_list)))
    '''REAL DEMAND'''
    df_demand = pd.read_csv('sku_sales_answer_filled.csv')
    df_demand.sort_values(['date','dc_id','item_sku_id'], ascending =[True,True,True], inplace = True)
    true_demand = df_demand.quantity.values.reshape((period_length,len(dc_list), len(sku_list)))
    # for test set A period_length = 31, for test set B period_length = 61
    period_length = 61
    true_demand = true_demand[:period_length]

    '''instance of simulation'''
    instance = DailyEvaluate(sku_list, dc_list, period_length, 
        initial_inventory, sku_limit, capacity_limit, 
        abandon_rate,  sku_cost, extra_shipping_cost_per_unit, 
        replenishment_leadtime, fixed_replenish_cost_per_order, 
        true_demand)

    '''instance of inventory policy'''
    some_policy = UserPolicy(initial_inventory.copy(deep=True), sku_cost.copy(deep=True))

    '''run simulation'''
    for t in range(1, period_length+1):
        inventory_decision = some_policy.daily_decision(t)
        update_return = instance.daily_update(inventory_decision, t-1)
        if update_return['status']:
            if t <period_length:
                # inventory at the end of the day
                # tday_inventory = pd.DataFrame([[s, d, instance.inv[d][s-1] ] for d in dc_list for s in sku_list ], \
                #                  columns = ['item_sku_id','dc_id','stock_quantity'])
                tday_inventory = pd.DataFrame({'item_sku_id': np.repeat([sku_list], len(dc_list), axis =0).flatten(), \
                         'dc_id': np.repeat(dc_list, len(sku_list)), \
                         'stock_quantity': instance.inv.flatten()})
                some_policy.info_update(tday_inventory,t)
        else:
            print('constraint violation in period ', t, '\n', update_return['constraints_violation']) 
            break

    '''
    Final score of the policy, lower is better
    '''
    print(instance.ordering_cost, instance.shortage_cost, instance.holding_cost, instance.extra_shipping_cost)
    score = instance.ordering_cost + instance.shortage_cost + instance.holding_cost + instance.extra_shipping_cost
    print('score',score)



