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
import pdb
#Testing the model
#Load parameterization files
pdb.set_trace()
numc = ['water', 'subsoil', 'air', 'pond'] #
#numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air', 'pond']
locsumm = pd.read_excel('inputfiles/Kortright_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/params_BC.xlsx',index_col = 0)
#params.loc['numc_disc','val'] = 2
#The tracertest_Kortright_extended file was used in Rodgers et al. (2021), _test is a smaller toy dataset for 
#testing the model
#timeseries = pd.read_excel('timeseries_tracertest_Kortright_extended.xlsx')
timeseries = pd.read_excel('inputfiles/timeseries_test.xlsx')
#Instantiate the model. In this case we will ball the model object "bioretention_cell"
bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
#pdb.set_trace()
results = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)