#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
Sample submission for 2nd round competition.
Author:feifei
Date:2018/8/22

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
        self.abandon_rate =np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])
        self.decision_record = []
        
        self.predict=[]
        self.predict_skufirst=[]
        self.hold_costs=[]
        self.shortage_cost=[]
        self.sku_priority=[]
        self.sku_sum=[]
        self.pre=[]
        self.on_way=np.zeros((1000))
        self.arrive_days=np.zeros((1000))
        self.meansquared=np.zeros((6,1000))
        self.meansquared_sum=np.zeros((1000))
        self.lower_adddays=np.zeros((1000))
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
        self.inv[t-1].sort_values(['dc_id','item_sku_id'], ascending = [True, True], inplace =True)
        stock = self.inv[t-1].stock_quantity.values.reshape((6,1000))
        rdc_total=stock[0]+self.on_way
        if(t==1):           
            self.init()
        inventory_decision = np.zeros((6, 1000)).astype(int)
        vlimit=np.zeros((5))
        climit=np.zeros((5))        
        dc0_8days=self.getNDaysNeed(t-1,10)
        dc_priority=[5,1,4,3,2]
        for i in self.sku_priority:
            sku=self.predict_skufirst[i]
            holdCost=self.hold_costs[0,i]
            if(dc0_8days[i]>rdc_total[i]):
                inventory_decision[0,i]=self.getRDCOrderSize(i,t+6,holdCost)
                self.on_way[i]=inventory_decision[0,i]
                self.arrive_days[i]=7
            for j in dc_priority:                
                if( vlimit[j-1]<self.sku_limit[j-1] and climit[j-1]<self.capacity_limit[j-1] ):
                    if self.getDcNDaysNeed(t-1,2,j,i)>stock[j,i]:                       
                        need_detail=round(self.getOrderSize(j,sku,t,climit[j-1],holdCost))
                        inventory_decision[j,i]=need_detail
                        sum_tran=0
                        for l in range(1,6):
                            sum_tran=sum_tran+inventory_decision[l,i]
                        sum_tran=sum_tran-need_detail
                        if((sum_tran+need_detail)>stock[0,i]):
                            if (stock[0,i]-sum_tran)>0:
                                inventory_decision[j,i]=stock[0,i]-sum_tran
                            else:
                                inventory_decision[j,i]=0   
                        vlimit[j-1]=vlimit[j-1]+1
                        climit[j-1]=climit[j-1]+inventory_decision[j,i]
        for i in range(1000):
            if self.arrive_days[i]>0:
                self.arrive_days[i]-=1
            if self.arrive_days[i]==0:
                self.on_way[i]=0
        self.decision_record.append(inventory_decision)
        return inventory_decision

    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv.append(end_day_inventory)
    def init(self):
        predict_data=pd.read_csv('y_new.csv')
        for j in range(1000):
            dc=[]
            for i in range(6):
                days=[]
                for t in range(61):
                    item=predict_data.iloc[t+j*61+i*61000,0]
                    days.append(item)
                dc.append(days)
                dcs=np.array(dc)
            self.predict_skufirst.append(dcs)
        for t in range(61):
            day=[]
            for i in range(6):
                dc=[]
                for j in range(1000):
                    item=predict_data.iloc[t+j*61+i*61000,0]
                    dc.append(item)
                day.append(dc)
            self.predict.append(day)
        self.costs.sort_values('item_sku_id', ascending = True, inplace =True)
        self.hold_costs = self.costs.holding_cost.values.reshape((1,1000))
        self.shortage_cost=self.costs.stockout_cost.values.reshape((1,1000))
        cost_times=self.shortage_cost[0]/self.hold_costs[0]
        for i in range(1000):
            if cost_times[i]>=4000:
                self.lower_adddays[i]=5
            elif cost_times[i]>=3200 and cost_times[i]<4000:
                self.lower_adddays[i]=4
            elif cost_times[i]>=2400 and cost_times[i]<3200:
                self.lower_adddays[i]=3
            elif cost_times[i]>=1600 and cost_times[i]<2400:
                self.lower_adddays[i]=2
            elif cost_times[i]>=1000 and cost_times[i]<1600:
                self.lower_adddays[i]=1
            else:
                self.lower_adddays[i]=0
        self.costs.sort_values('stockout_cost', ascending = False, inplace =True)
        self.sku_priority=(self.costs.item_sku_id.values.reshape((1,1000))-np.ones((1,1000)))[0].astype(int)
        for i in range(1000):
            s=self.predict_skufirst[i].sum(axis=0)
            self.sku_sum.append(s)
        self.pre=np.asarray(self.predict)
        
    def getNDaysNeed(self,start,n):
        #get RDC(cantain FDC) n days needs for confirming RDC order point
        needs=np.array(np.zeros((6,1000)))
        end=start+n+self.lower_adddays
        end[end>61]=61       
        for i in range(1000):
            a=self.predict_skufirst[i]
            e=int(end[i])
            for j in range(6):
                self.predict_skufirst[i]
                needs[j,i]=a[j,start:e].sum()
        totalNeeds=needs.sum(axis=0)
        return totalNeeds
    
    def getDcNDaysNeed(self,start,n,dc_id,sku_id):
        #get FDC n days needs for confirming FDC order point
        needs=0
        end=start+n
        if end>61:
            end=61
        for i in range(start,end):
            needs=needs+self.pre[i][dc_id][sku_id]
        return needs
    
    def getOrderSize(self,dc_id,sku,begin_date,dc_total,hold_cost):
        #get dc_id's(FDC) transshipment
        days=0
        sku_hold_cost=0
        order_size=0
        while sku_hold_cost < 0.1:
            if (begin_date+days)>=61:
                break
            needs_per_day=sku[dc_id][begin_date+days]
            sku_hold_cost=sku_hold_cost+needs_per_day*hold_cost*days
            order_size=order_size+needs_per_day
            days=days+1
        if ((order_size+dc_total)>self.capacity_limit[dc_id-1]):
            if (self.capacity_limit[dc_id-1]-dc_total)>0:
                order_size=self.capacity_limit[dc_id-1]-dc_total
            else:
                order_size=0
        return order_size
    
    def getRDCOrderSize(self,sku_id,begin_date,hold_cost):
        #get RDC transshipment
        days=0
        sku_hold_cost=0
        order_size=0
        while sku_hold_cost < 0.1:
            if (begin_date+days)>=61:
                break
            needs_per_day=self.sku_sum[sku_id][begin_date+days]
            sku_hold_cost=sku_hold_cost+needs_per_day*hold_cost*days
            order_size=order_size+needs_per_day
            days=days+1
        return order_size
