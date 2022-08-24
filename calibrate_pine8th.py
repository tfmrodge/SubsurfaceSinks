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
#chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/params_Pine8th.xlsx',index_col = 0)
pp = None
#testing the model
timeseries = pd.read_excel('inputfiles/timeseries_Pine8th.xlsx')
#Run only for the first event
timeseries = timeseries[timeseries.time<=6]
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_simstorm.xlsx')
#Import a flows if you want it
flowpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/flowtest.pkl'
flow_time = pd.read_pickle(flowpath)
#Instantiate the model
bc =  BCBlues(locsumm,chemsumm,params,timeseries,numc)
#Now, run the calibration
cal_flows = False
#For flows:
if cal_flows == True:
    paramnames = ['Kf','Kn','native_depth']
    param0s = [0.2,0.1,0.1]
    cal = bc.calibrate_flows(timeseries,paramnames,param0s)
else:
    #paramnames = ['Kf','Kn','native_depth']
    paramnames = ['alpha','thetam']
    #param0s = [0.69,0.2]
    param0s = [0.144]
    bnds = ((0.0,50),(0.0,1.0))#,(0.0,10000))
    #cal = calibrate_tracer(timeseries,paramnames,param0s,flows=flow_time)
    cal = bc.calibrate_tracer(timeseries,paramnames,param0s,bounds = bnds,flows=None)