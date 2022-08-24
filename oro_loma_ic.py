# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:45:52 2021

@author: Tim Rodgers
"""

import time
import pandas as pd
import numpy as np
from Loma_Loadings import LomaLoadings
import pdb
import math
#plt.style.use("ggplot")
#from data_1d import makeic

params = pd.read_excel('params_1d.xlsx',index_col = 0) 
#params = pd.read_excel('params_OroLoma_CellF.xlsx',index_col = 0)
#params = pd.read_excel('params_OroLomaTracertest.xlsx',index_col = 0)
#Cell F and G
locsumm = pd.read_excel('Oro_Loma_CellF.xlsx',index_col = 0) 
#locsumm = pd.read_excel('Oro_Loma_CellG.xlsx',index_col = 0) 
#15 cm tracertest conducted with the Oro Loma system Mar. 2019
#locsumm = pd.read_excel('Oro_Loma_Brtracertest.xlsx',index_col = 0) 


#Specific Groups
#chemsumm = pd.read_excel('Kortright_BRCHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('Oro_ALLCHEMSUMMtest.xlsx',index_col = 0)
#chemsumm = pd.read_excel('Oro_FAVCHEMSUMM.xlsx',index_col = 0)

#timeseries = pd.read_excel('timeseries_OroLomaTracertest.xlsx')
#timeseries = pd.read_excel('timeseries_OroLoma_Dec_CellF.xlsx')
timeseries = pd.read_excel('timeseries_OroLoma_Spinup_CellF.xlsx')
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
#numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
test = LomaLoadings(locsumm,chemsumm,params,numc) 

#pdb.set_trace()
timeseries = timeseries.loc[timeseries.time>0,:]

start = time.time()


#res = test.make_system(res_time,params,numc)
res_t = test.input_calc(locsumm,chemsumm,params,pp,numc,timeseries)
#mf = test.mass_flux(res_time,numc)
#res_t, res_time = test.run_it(locsumm,chemsumm,params,1,pp,timeseries)


codetime = (time.time()-start)
#For the input calcs
#outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_input_calcs_extended.pkl'
outpath ='D:/OneDrive - University of Toronto/University/_Active Projects/Bioretention Blues Model/Model/Pickles/oro_input_calcs_extended.pkl'
res_t.to_pickle(outpath)