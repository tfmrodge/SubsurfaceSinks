# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:54:29 2022
Calibrate the Pine/8th System
@author: trodge01
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
#This is the name of the python module containing the Bioretention Blues submodel.
from BioretentionBlues import BCBlues
from HelperFuncs import df_sliced_index
import pdb
import time
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
import hydroeval
#Testing the model
#Load parameterization files
pdb.set_trace()
#numc = ['water', 'subsoil', 'air', 'pond'] #
codetime = time.time()
#For the vancouver tree trench, no ponding zone. 
#numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
#locsumm = pd.read_excel('inputfiles/QuebecSt_TreeTrench.xlsx',index_col = 0)
locsumm = pd.read_excel('inputfiles/Pine8th_BC.xlsx',index_col = 0)
#Only bromide
chemsumm = pd.read_excel('inputfiles/Br_CHEMSUMM.xlsx',index_col = 0)
#Only rhodamine
#chemsumm = pd.read_excel('inputfiles/CHEMSUMM_rhodamine.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/params_Pine8th.xlsx',index_col = 0)
pp = None
#testing the model
timeseries = pd.read_excel('inputfiles/timeseries_Pine8th.xlsx')
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_short.xlsx')
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_3hr.xlsx')
#Run only for the first event
timeseries = timeseries[timeseries.time<=6]
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_simstorm.xlsx')
#Import a flows if you want it
#flowpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/flowtest.pkl'
#flow_time = pd.read_pickle(flowpath)
#Instantiate the model
bc =  BCBlues(locsumm,chemsumm,params,timeseries,numc)
pklpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/'
timedfname = 'mod_timeseries.pkl'
#How much should we modify the time-step. Multiply the index by this number. 
indfactor = 3#'Load' #3#3
if indfactor == 'Load':
    timeseries = pd.read_pickle(pklpath+timedfname)
else:
    try: 
        int(indfactor) == indfactor
        timeseries = bc.modify_timestep(timeseries,indfactor)
    except TypeError:
        pass
#Now, run the calibration
cal_flows = True
#For flows:
if cal_flows == True:
    paramnames = ['Kf','Kn','native_depth']
    param0s = [0.20297723,0.1247891,0.1541343]
    cal = bc.calibrate_flows(timeseries,paramnames,param0s)
else:
    #paramnames = ['AF_soil']
    #param0s = [55.0416069]
    #bnds = ((1e-6,100.),)
    #paramnames = ['alpha']
    #param0s = [0.63210325]
    #bnds = ((1e-6,100.),)
    tol = 1e-6
    #paramnames = ['alpha','thetam','wmim']
    #param0s = [0.63210325, 0.12847097, 0.01205933]
    #bnds = ((1e-6,10.),(1e-6,1.),(1e-6,20.))
    
    paramnames = ['Kf','Kn','native_depth','alpha']
    param0s = [0.20330968, 0.12166552, 0.15501755,0.63210325]
    bnds = ((1e-6,1.),(1e-6,1.),(1e-6,1.),(1e-6,10.))
    #paramnames = ['AF_soil','wmim']
    #param0s = [71.55216748, 0.01205933]#, 0.01205933]
    #bnds = ((1e-6,100),(0.001,100.))#,(0.0,100))((1e-6,100),(0.001,1.0))#,(0.0,100))
    #,(0.0,10000))
    #objective = {'recovery':0.1371}    
    objective = None
    #cal = calibrate_tracer(timeseries,paramnames,param0s,flows=flow_time)
    cal = bc.calibrate_tracer(timeseries,paramnames,param0s,bounds = bnds,
                              tolerance = tol,flows=None,objective=objective)
    
'''
Results - 20221208
    
Bromide - ['alpha','thetam','wmim']
0.16698067838138309 [0.63210325, 0.12847097, 0.01205933]

Rhodamine - AF_soil
0.4520882813731859 [71.55216748]

Results - 20221027    
Flows -  ['Kf','Kn','native_depth']
    obj - 0.07090380394757101, params = [0.20330968, 0.12166552, 0.15501755]

'''