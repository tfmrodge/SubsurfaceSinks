# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:52:52 2021

@author: Tim Rodgers
"""

import time
import pandas as pd
import numpy as np
from Loma_Loadings import LomaLoadings
from HelperFuncs import df_sliced_index
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pdb
import math
import hydroeval #For the efficiency
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
from hydroeval import *
#plt.style.use("ggplot")
params = pd.read_excel('params_1d.xlsx',index_col = 0) 
#params = pd.read_excel('params_OroLoma_CellF.xlsx',index_col = 0)
#params = pd.read_excel('params_OroLomaTracertest.xlsx',index_col = 0)
#Cell F and G
locsumm = pd.read_excel('Oro_Loma_CellF.xlsx',index_col = 0) 
#locsumm = pd.read_excel('Oro_Loma_CellG.xlsx',index_col = 0) 
#15 cm tracertest conducted with the Oro Loma system Mar. 2019
#locsumm = pd.read_excel('Oro_Loma_Brtracertest.xlsx',index_col = 0) 
#chemsumm = pd.read_excel('Kortright_OPECHEMSUMM.xlsx',index_col = 0)

#Specific Groups
chemsumm = pd.read_excel('Oro_ALLCHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Oro_FAVCHEMSUMM.xlsx',index_col = 0)

#timeseries = pd.read_excel('timeseries_OroLomaTracertest.xlsx')
timeseries = pd.read_excel('timeseries_OroLoma_Dec_CellF.xlsx')
#timeseries = pd.read_excel('timeseries_OroLoma_Spinup_CellF.xlsx')
#Truncate timeseries if you want to run fewer
pdb.set_trace()
totalt = len(timeseries.index) #100
if totalt <= len(timeseries):
    timeseries = timeseries[0:totalt+1]
else:
    while math.ceil(totalt/len(timeseries)) > 2.0:
        timeseries = timeseries.append(timeseries)
    totalt = totalt - len(timeseries)
    timeseries = timeseries.append(timeseries[0:totalt])
    timeseries.loc[:,'time'] = np.arange(1,len(timeseries)+1,timeseries.time.iloc[1]-timeseries.time.iloc[0])
    timeseries.index = range(len(timeseries))
    
pp = None
numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
test = LomaLoadings(locsumm,chemsumm,params,numc) 

res = pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_input_calcs_extended.pkl')
#Load steady-state concentrations as the initial conditions.
#laststep = pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_steady.pkl')
#laststep = pd.read_pickle('D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_input_calcs_spinup.pkl')
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_spinup20210602.pkl'
laststep = pd.read_pickle(outpath)

#Then, run it.
start = time.time()
res = test.run_it(locsumm,chemsumm,params,pp,numc,timeseries,res,last_step=laststep)
codetime = (time.time()-start)
#res = test.run_it(locsumm,chemsumm,params,pp,numc,timeseries,res)
mass_flux = test.mass_flux(res,numc) #Run to get mass flux
mbal = test.mass_balance(res,numc,mass_flux)
Couts = test.conc_out(numc,timeseries,chemsumm,res,mass_flux)
recovery = mass_flux.N_effluent.groupby(level=0).sum()/mass_flux.N_influent.groupby(level=0).sum()
#Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
KGE = {}
pltnames = []
pltnames.append('time')
#Calculate performance and define what to plot
for chem in chemsumm.index:
    #If measured values provided
    try:
        KGE[chem] = hydroeval.evaluator(kge, np.array(Couts.loc[:,chem+'_Coutest']),\
                          np.array(Couts.loc[:,chem+'_Coutmeas']))
        pltnames.append(chem+'_Coutmeas')
    except KeyError:    
        pass
    pltnames.append(chem+'_Coutest')
    

#plot it    
pltdata = Couts[pltnames]
pltdata = pltdata.melt('time',var_name = 'Test_vs_est',value_name = 'Cout (mg/L)')
ylim = [0, 50]
ylabel = 'Cout (mg/L)'
xlabel = 'Time'
#pltdata = res_time #All times at once
fig = plt.figure(figsize=(14,8))
ax = sns.lineplot(x = pltdata.time, y = 'Cout (mg/L)', hue = 'Test_vs_est',data = pltdata)

#ax.set_ylim(ylim)
ax.set_ylabel(ylabel, fontsize=20)
ax.set_xlabel(xlabel, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
#Save it
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_outs_extended.pkl'
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/tracer_outs_630max.pkl'
res.to_pickle(outpath)