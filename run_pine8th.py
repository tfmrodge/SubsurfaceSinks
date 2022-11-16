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
#chemsumm = pd.read_excel('inputfiles/PPD_CHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
#chemsumm.loc['Rhodamine','pKa'] = 999
#chemsumm.loc['Rhodamine','chemcharge'] = 0
#chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/params_Pine8th.xlsx',index_col = 0)
pp = None
#testing the model
timeseries = pd.read_excel('inputfiles/timeseries_Pine8th.xlsx')
timeseries = timeseries[timeseries.time<=6]
timeseries.loc[:,['pHwater']] = timeseries.pHwater + 4
timeseries.loc[:,['pHsubsoil']] = timeseries.pHwater
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_simstorm.xlsx')
#Check KGE
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_3hr.xlsx')
#Changing parameters for testing
'''
paramnames = ['alpha','thetam']
paramvals = [0.55519243, 0.34502549]
for ind, paramname in enumerate(paramnames):
    params.loc[paramname,'val'] = paramvals[ind]
'''
#timeseries = pd.read_excel('inputfiles/timeseries_2014.xlsx')
#Instantiate the model. In this case we will ball the model object "bioretention_cell"
bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
#pdb.set_trace()
calcflow = True#True#False# True# True#
#flowpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/flowtest.pkl'
flowpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/2014_flows.pkl'
if calcflow is True:
    flow_time = bc.flow_time(locsumm,params,['water','subsoil'],timeseries)
    mask = timeseries.time>=0
    minslice = np.min(np.where(mask))
    maxslice = np.max(np.where(mask))#minslice + 5 #
    flow_time = df_sliced_index(flow_time.loc[(slice(minslice,maxslice),slice(None)),:])
    flow_time.to_pickle(flowpath)
    #bc.plot_flows(flow_time,Qmeas = flow_time.loc[(slice(None),'pond'),'Q_in'],compartments=['drain','water'],yvar='Q_todrain')
    #Plot whole event
    bc.plot_flows(flow_time,Qmeas = timeseries.Qout_meas,compartments=['drain','water'],yvar='Q_todrain')
    #Plot spike event
    bc.plot_flows(flow_time.loc[flow_time.time<6],Qmeas = timeseries.loc[timeseries.time<6,'Qout_meas'],
                  compartments=['drain','water'],yvar='Q_todrain')
    #Plot latter event
    #bc.plot_flows(flow_time.loc[flow_time.time>140],Qmeas = timeseries.loc[timeseries.time>140,'Qout_meas'],
    #              compartments=['drain','water'],yvar='Q_todrain')
    #% infiltrated - actual was ~78%
    inf_pct = 1 - (flow_time.loc[(slice(None),'drain'),'Q_todrain'].sum()/flow_time.loc[(slice(None),'pond'),'Q_in'].sum())
    #timeseries.loc[:,'Q_drainout'] = np.array(flow_time.loc[(slice(None),'drain'),'Q_out'])
    KGE = hydroeval.evaluator(kge, np.array(flow_time.loc[(slice(None),'drain'),'Q_todrain']),\
                          np.array(timeseries.loc[timeseries.time>=0,'Qout_meas']))
    #flow_time.to_pickle(flowpath)
elif calcflow is None:
    pass
else:
    flow_time = pd.read_pickle(flowpath)

codetime = time.time() - codetime

#'''
#Input calculations
#inpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/inputtest.pkl'   
inpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/2014_inputs.pkl'   
calcinp = True#False#
if calcinp is True:
    input_calcs = bc.input_calc(locsumm,chemsumm,params,pp,numc,timeseries,flow_time=flow_time)
    input_calcs.to_pickle(inpath)
else:
    input_calcs = pd.read_pickle(inpath)
#
#input_calcs = pd.read_pickle(inpath)
#'''
runall = False#None#None#'Load'#'Load'#
if runall is True:
    res = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)
elif runall == 'Load':
    outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/outputtest.pkl'
    res = pd.read_pickle(outpath)
elif runall == None:
    pass
else:
    res = bc.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs=input_calcs)
#res = bc.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs=input_calcs)
#res = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)
mass_flux = bc.mass_flux(res,numc) #Run to get mass flux
mbal = bc.mass_balance(res,numc,mass_flux)
Couts = bc.conc_out(numc,timeseries,chemsumm,res,mass_flux)
#bc.plot_Couts(res,Couts,multfactor=1000)
#bc.plot_Couts(res,Couts.loc[Couts.time<6],multfactor=1e6)
recovery = mass_flux.N_effluent.groupby(level=0).sum()/mass_flux.N_influent.groupby(level=0).sum()
KGE = {}
for ind,chem in enumerate(chemsumm.index):
    try:
        KGE[chem] = (hydroeval.evaluator(kge, np.array(Couts.loc[:,chem+'_Coutest']),\
                          np.array(Couts.loc[:,chem+'_Coutmeas'])))
    except KeyError:
        pass
codetime = time.time() - codetime
plotfig = False
if plotfig == True:
    #Calculate the masses
    mbal_cum = bc.mass_balance_cumulative(numc, mass_balance = mbal,normalized=True)
    compound = '6PPDQ'#use same name as in chemsumm
    #Set time (hrs), any time after end gives the end.
    time = 8760#1000#6#1000#6#1000
    fig,ax = bc.BC_fig(numc,mass_balance=mbal_cum,time = time,compound=compound,figheight=6,fontsize=7,dpi=300)
#bc.plot_flows(flow_time,Qmeas = timeseries.Qout_meas,compartments=['drain','water'],yvar='Q_out')
#outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/2014_results.pkl'
#outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/6PPDQ_simstorm.pkl'
outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/6PPDQ_spiketest.pkl'
#res.to_pickle(outpath)
#'''