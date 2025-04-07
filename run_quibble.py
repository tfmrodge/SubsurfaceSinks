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
from Stormpond import StormPond
from inputfiles.QuibblePond.quibble_dimcalcs import quibble_dimcalc_tables
from HelperFuncs import df_sliced_index
import pdb
import time
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
import hydroeval
import ast
#import ast
#Testing the model
pdb.set_trace()
#First, import measured dimensions
#D:\Github\SubsurfaceSinks\inputfiles\QuibblePond\Quibble_Pond.xlsx"
qbdims = pd.read_excel('inputfiles/QuibblePond/Quibble_Pond.xlsx',sheet_name='PONDSUMM')
params = pd.read_excel('inputfiles/QuibblePond/params_Quibble.xlsx',index_col = 0)
#Next, define breakpoint depths with dimdict. Code interpolates between these depths
# #Important Depths #m
# designdepth = 0.5 
# bermtop = 0.9 #73.4-72.5
# hwl = 1.8 #74.3-72.5 #high water line
# overflow = 2 #74.5-72.5
# minW = 0.610 #Set to match influent pipe diameter (600mm pipe = 610 ID)
dimdict = ast.literal_eval(params.val.dimdict)
qbdims = quibble_dimcalc_tables(qbdims,dimdict)
#Import rest of the initialization files
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air'] 
locsumm = pd.read_excel('inputfiles/QuibblePond/Quibble_Pond.xlsx',sheet_name='LOCSUMM',index_col=0)
chemsumm = pd.read_excel('inputfiles/QuibblePond/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
pp = None
timeseries = pd.read_excel('inputfiles/QuibblePond/timeseries_qbtest.xlsx')
#"D:\Github\SubsurfaceSinks\inputfiles\QuibblePond\6PPDQ_CHEMSUMM.xlsx"
#Initialize the model
qbl =  StormPond(locsumm,chemsumm,params,timeseries,numc)
#Define dX
dx = params.val.dx
qbsys=qbl.make_system(locsumm,params,numc,timeseries,qbdims,dx=dx)
#numc = ['water', 'subsoil', 'air', 'pond'] #
# codetime = time.time()
# pklpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/'
# #For the vancouver tree trench, no ponding zone. 
# #numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
# numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
# #locsumm = pd.read_excel('inputfiles/QuebecSt_TreeTrench.xlsx',index_col = 0)
# locsumm = pd.read_excel('inputfiles/Pine8th/Pine8th_BC.xlsx',index_col = 0)
# #chemsumm = pd.read_excel('inputfiles/Pine8th/EngDesign_CHEMSUMM.xlsx',index_col = 0)
# chemsumm = pd.read_excel('inputfiles/Pine8th/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
# #Change to episuite version
# #chemsumm.loc['6PPDQ','LogKocW'] = 3.928
# #chemsumm.loc['Rhodamine','chemcharge'] = 0
# #chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
# params = pd.read_excel('inputfiles/Pine8th/params_Pine8th.xlsx',index_col = 0)
# #params.loc['f_apo','val'] = 0
# pp = None
# #testing the model
# timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_Pine8th.xlsx')
