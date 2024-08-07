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
<<<<<<< Updated upstream
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
=======
#Import the res file, load the others
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
#locsumm = pd.read_excel('inputfiles/QuebecSt_TreeTrench.xlsx',index_col = 0)
locsumm = pd.read_excel('inputfiles/Pine8th/Pine8th_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/Pine8th/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_wateryear.xlsx')
#chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/Pine8th/params_Pine8th.xlsx',index_col = 0)
pp = None
inpath = 'D:/OneDrive - UBC/Postdoc/Completed Projects/6PPD_BC Papers/Modeling/Pickles/'
#inpath = 'C:/Users/trodge01/Documents/BigPickles/'
#res = pd.read_pickle(inpath+'outputspiketest1.pkl')
#res = pd.read_pickle(inpath+'wateryear_fvalve_Foc_Asys_Hp.pkl')
#res = pd.read_pickle(inpath+'wateryear_.pkl')
#res = pd.read_pickle(inpath+'20230202_CorrectedFinal.pkl')
res = pd.read_pickle(inpath+'Pine8spiketest_20240807.pkl')
#res = pd.read_pickle(fpath+'/pickles/outputtest.pkl')
#.head()
bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
#timeseries = timeseries.loc[timeseries.time<240,:]
indfactor = 1#3#3#'Load' #3#3
if indfactor == 'Load':
    timeseries = pd.read_pickle(pklpath+timedfname)
else:
    try: 
        int(indfactor) == indfactor
        timeseries = bc.modify_timestep(timeseries,indfactor)
    except TypeError:
        pass
#timeseries = timeseries[timeseries.time<=6]
bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
mbal = bc.mass_balance(res,numc)
mbal_cum = bc.mass_balance_cumulative(numc, mass_balance = mbal,normalized=True)
mbal_cum.head()
mass_flux = bc.mass_flux(res,numc) 
Couts = bc.conc_out(numc,timeseries,chemsumm,res)
Couts.loc[:,'6PPDQ_Coutest'] = Couts.loc[:,'6PPDQ_Coutest'] *1e6 #ng/L
Couts.head()
measdat = pd.read_excel('/inputfiles/Pine8th/Tracer_test_measurements.xlsx')
Couts.head()
>>>>>>> Stashed changes
