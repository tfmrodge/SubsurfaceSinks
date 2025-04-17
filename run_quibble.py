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
from HelperFuncs import df_sliced_index, culvert_flow_est
import pdb
import time
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
import hydroeval
import ast
from scipy import optimize
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
#timeseries = pd.read_excel('inputfiles/QuibblePond/timeseries_qb_20241018.xlsx')
#Logger offset from channel/pipe bottom
timeseries.loc[:,'outlevel_m'] = timeseries.outlevel_m +(69.215-51.2)/100#Measured stage vs water surface 11:45 2025-03-20
timeseries.loc[:,'inlevel_m'] = timeseries.outlevel_m +0.03 #Assumed value
#For Qin, tailwater depth = outlevel +0.05 (higher by 0.05m at head than tail)
timeseries.loc[:,'Qin'] = 3600*culvert_flow_est(
        timeseries.inlevel_m+0.05, #m, series or array, from channel bottom
        timeseries.outlevel_m, #m, series or array, from channel bottom
        params.val.D_culvert_in, #m, culvert diameter (assumes circular)
        params.val.L_culvert_in,#m, culvert length
        head_offset=0., #m, measured from channel bottom 
        tail_offset=0., #m, measured from channel bottom
        n_manning=params.val.culvert_n)
calc_Qout = False
if calc_Qout==True:
    #"D:\Github\SubsurfaceSinks\inputfiles\QuibblePond\6PPDQ_CHEMSUMM.xlsx"
    def calibrate_QinQout(tailratio):
        timeseries.loc[:,'Qin'] = culvert_flow_est(
                timeseries.inlevel_m, #m, series or array, from channel bottom
                timeseries.outlevel_m-0.05, #m, series or array, from channel bottom
                params.val.D_culvert_in, #m, culvert diameter (assumes circular)
                params.val.L_culvert_in,#m, culvert length
                head_offset=0., #m, measured from channel bottom 
                tail_offset=0., #m, measured from channel bottom
                n_manning=params.val.culvert_n)
        #For Qout, assume same ratio of tailwater depth as average across event
        #tailratio = 0.5#(timeseries.outlevel_m/timeseries.inlevel_m).mean()
        timeseries.loc[:,'Qout'] = culvert_flow_est(
                timeseries.outlevel_m, #m, series or array, from channel bottom
                timeseries.outlevel_m*tailratio, #m, series or array, from channel bottom
                params.val.D_culvert_out, #m, culvert diameter (assumes circular)
                params.val.L_culvert_out,#m, culvert length
                head_offset=0., #m, measured from channel bottom 
                tail_offset=0., #m, measured from channel bottom
                n_manning=params.val.culvert_n)
        minimizer = abs(timeseries.Qin.sum()-timeseries.Qout.sum())
        return minimizer
        #Assume across event that Qin=Qout. If outlet stage balances should be true
    testtr=0.5
    tailratio = optimize.newton(calibrate_QinQout,testtr,tol=1e-5)
    
    #For Qout, assume same ratio of tailwater depth as average across event
    #tailratio = 0.5#(timeseries.outlevel_m/timeseries.inlevel_m).mean()
    timeseries.loc[:,'Qout'] = 3600*culvert_flow_est(
            timeseries.outlevel_m, #m, series or array, from channel bottom
            timeseries.outlevel_m*tailratio, #m, series or array, from channel bottom
            params.val.D_culvert_out, #m, culvert diameter (assumes circular)
            params.val.L_culvert_out,#m, culvert length
            head_offset=0., #m, measured from channel bottom 
            tail_offset=0., #m, measured from channel bottom
            n_manning=params.val.culvert_n)

#Testing - reduce Qin 
#timeseries.loc[:,'Qin'] = timeseries.Qin/3600*6
locsumm.loc['water','Depth'] = timeseries.outlevel_m[0]
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
