# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:16:57 2021

@author: Tim Rodgers
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
from hydroeval import kge
import hydroeval
#Testing the model
#Load parameterization files
#pdb.set_trace()
#numc = ['water', 'subsoil', 'air', 'pond'] #
codetime = time.time()
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air', 'pond']
locsumm = pd.read_excel('inputfiles/Kortright_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0) #Kortright_voltestCHEMSUMM
#chemsumm = pd.read_excel('inputfiles/Kortright_voltestCHEMSUMM.xlsx',index_col = 0) 
#chemsumm = pd.read_excel('inputfiles/PPD_CHEMSUMM.xlsx',index_col = 0) 
params = pd.read_excel('inputfiles/params_BC.xlsx',index_col = 0)
pp = None
#params.loc['numc_disc','val'] = 2
#The tracertest_Kortright_extended file was used in Rodgers et al. (2021), _test is a smaller toy dataset for 
#testing the model
#timeseries = pd.read_excel('inputfiles/timeseries_tracertest_synthetic.xlsx')
#timeseries = pd.read_excel('inputfiles/timeseries_tracertest_Kortright_7hr.xlsx')
timeseries = pd.read_excel('inputfiles/timeseries_tracertest_Kortright_extended.xlsx')
#timeseries = pd.read_excel('inputfiles/timeseries_tracertest_Kortright_7hr_smallin.xlsx')
#timeseries = pd.read_excel('inputfiles/timeseries_synthetic.xlsx')
#Instantiate the model. In this case we will ball the model object "bioretention_cell"
bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
#pdb.set_trace()
flow_time = bc.flow_time(locsumm,params,['water','subsoil'],timeseries)
#mask = timeseries.time>=0
#minslice = np.min(np.where(mask))
#maxslice = np.max(np.where(mask))#minslice + 5 #
#flow_time = df_sliced_index(flow_time.loc[(slice(minslice,maxslice),slice(None)),:])
#flowpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/BiorentionBlues_Modeling/Kortright_OPEs/Pickles/flowtest.pkl'
#flow_time = pd.read_pickle(flowpath)
input_calcs = bc.input_calc(locsumm,chemsumm,params,pp,numc,timeseries,flow_time=flow_time)
#inpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/BiorentionBlues_Modeling/Kortright_OPEs/Pickles/inputtest.pkl'
#input_calcs = pd.read_pickle(inpath)
res = bc.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs=input_calcs)


mass_flux = bc.mass_flux(res,numc) #Run to get mass flux
mbal = bc.mass_balance(res,numc,mass_flux)
Couts = bc.conc_out(numc,timeseries,chemsumm,res,mass_flux)
recovery = mass_flux.N_effluent.groupby(level=0).sum()/mass_flux.N_influent.groupby(level=0).sum()
#Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
KGE = {}
pltnames = []
pltnames.append('time')
#Calculate performance and define what to plot
#for chem in chemsumm.index:
#    KGE[chem] = hydroeval.evaluator(kge, np.array(Couts.loc[:,chem+'_Coutest']),\
#                      np.array(Couts.loc[:,chem+'_Coutmeas']))
#    pltnames.append(chem+'_Coutest')
#    pltnames.append(chem+'_Coutmeas')


#res = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)
codetime = time.time() - codetime
outpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/BiorentionBlues_Modeling/Kortright_OPEs/Pickles/outputvoltest.pkl'
#flow_time.to_pickle(outpath)
res.to_pickle(outpath)
