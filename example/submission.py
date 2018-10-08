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
        self.inv = [initial_inventory]
        self.costs = sku_cost
        self.extra_shipping_cost_per_unit = 0.01
        self.fixed_replenish_cost_per_order = 0.1
        self.sku_limit = np.asarray([300, 300, 300, 300, 300])
        self.capacity_limit = np.asarray([4000, 5000, 6000, 2000, 1000])
        self.abandon_rate =np.asarray([9./100, 1./100, 7./100, 8./100, 10./100])
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
        inventory_decision = np.zeros((6, 1000)).astype(int)
        self.decision_record.append(inventory_decision)
        return inventory_decision

    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv.append(end_day_inventory)

    def some_other_functions():
        pass
