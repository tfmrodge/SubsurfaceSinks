# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:19:29 2024

@author: trodge01
"""
#Load packages
#Standard packages
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
#This is the name of the python module containing the Bioretention Blues submodel.
from BioretentionBlues import BCBlues
import pdb
pdb.set_trace()
#Load parameterization files
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
#chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
locsumm = pd.read_excel('inputfiles/Pine8th/Pine8th_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/Pine8th/EngDesign_CHEMSUMM2.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/Pine8th/params_Pine8th.xlsx',index_col = 0)
timeseries = pd.read_excel('inputfiles/timeseries_test.xlsx')
Cin = 1000 #ng/L
for compound in chemsumm.index:
    minname = compound+'_Min'
    timeseries.loc[:,minname] = timeseries.Qin*Cin*1/60 
#The tracertest_Kortright_extended file was used in Rodgers et al. (2021), _test is a smaller toy dataset for 
#testing the model
#timeseries = pd.read_excel('inputfiles/timeseries_tracertest_Kortright_extended.xlsx')
#Instantiate the model. In this case we will ball the model object "bioretention_cell"
bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
results = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)
results.head()