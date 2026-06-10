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
from scipy.optimize import minimize_scalar
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
#timeseries = pd.read_excel('inputfiles/QuibblePond/timeseries_qbtest.xlsx')
timeseries = pd.read_excel('inputfiles/QuibblePond/timeseries_qb_20241018.xlsx')
#timeseries = pd.read_excel('inputfiles/QuibblePond/timeseries_qb_20250319.xlsx')
#Logger offset from channel/pipe bottom
timeseries.loc[:,'inlevel_m'] = timeseries.inlevel_m +0.04 #From Surrey measurement May 13 2025
timeseries.loc[:,'outlevel_m'] = timeseries.outlevel_m +(69.215-51.2)/100#Measured stage vs water surface 11:45 2025-03-20
#For Qin (m3/hr), tailwater depth = outlevel + 0.05 (higher by 0.05m at head than tail)
timeseries.loc[:,'Qin'] = 3600*culvert_flow_est(
        timeseries.inlevel_m, #m, series or array, from channel bottom
        timeseries.outlevel_m -0.05, #m, series or array, from channel bottom
        params.val.D_culvert_in, #m, culvert diameter (assumes circular)
        params.val.L_culvert_in,#m, culvert length
        head_offset=0., #m, measured from channel bottom 
        tail_offset=0., #m, measured from channel bottom
        n_manning=params.val.culvert_n)
calc_Qout = True
if calc_Qout==True:
    #"D:\Github\SubsurfaceSinks\inputfiles\QuibblePond\6PPDQ_CHEMSUMM.xlsx"
    def calibrate_QinQout(tailratio):
        if tailratio > 1:
            minimizer = 1e99
            return minimizer
        timeseries.loc[:,'Qin'] = 3600*culvert_flow_est(
                timeseries.inlevel_m, #m, series or array, from channel bottom
                timeseries.outlevel_m-0.05, #m, series or array, from channel bottom.
                params.val.D_culvert_in, #m, culvert diameter (assumes circular)
                params.val.L_culvert_in,#m, culvert length
                head_offset=0., #m, measured from channel bottom 
                tail_offset=0., #m, measured from channel bottom
                n_manning=params.val.culvert_n)
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
        minimizer = abs(timeseries.Qin.sum()-timeseries.Qout.sum())
        return minimizer
        #Assume across event that Qin=Qout. If outlet stage balances should be true
    testtr=0.5
    tailratio_results = minimize_scalar(
        lambda x: calibrate_QinQout(x)**2,
        bounds=(0.0, 1.0),
        method="bounded")
    tailratio = tailratio_results.x
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
#Calculate loadings
timeseries['dt'] = timeseries['time'].diff()
timeseries['Min_6PPDQ'] = timeseries['6PPDQ_Cin']*timeseries['Qin']*timeseries['dt']
timeseries['Min_6PPD'] = timeseries['6PPD_Cin']*timeseries['Qin']*timeseries['dt']
timeseries['Mout_6PPDQ'] = timeseries['6PPDQ_Coutmeas']*timeseries['Qout']*timeseries['dt']
timeseries['Mout_6PPD'] = timeseries['6PPD_Coutmeas']*timeseries['Qout']*timeseries['dt']
removal_6PPDQ = (timeseries['Min_6PPDQ'].sum()-timeseries['Mout_6PPDQ'].sum())/timeseries['Min_6PPDQ'].sum()
removal_6PPD = (timeseries['Min_6PPD'].sum()-timeseries['Mout_6PPD'].sum())/timeseries['Min_6PPD'].sum()
#event = timeseries[timeseries.time>0].copy()
#event_6PPD_Min 
outpath = r'D:\OneDrive - UBC\Postdoc\Active Projects\SurreyPonds\Data'
#outname = outpath + '20241018_event_5mininterp_w_flows.csv'
outname = outpath + '20250319_event_5mininterp_w_flows.csv'
#timeseries.to_csv(outname)
timeseries = timeseries[timeseries.time>=0].copy().sort_values('time').reset_index(drop=True)
# =========================
# Plotting: 3-panel figure
# =========================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# =========================
# PANEL 1: Concentration + Flow
# =========================
ax1 = axes[0]
ax1r = ax1.twinx()

# Ensure flow axis sits behind
ax1.set_zorder(2)
ax1.patch.set_visible(False)

# Flow (BACKGROUND)
ax1r.plot(timeseries['time'], timeseries['Qin']/3600,
          color='black', alpha=0.6, lw=1.5, label='Qin', zorder=0)
ax1r.plot(timeseries['time'], timeseries['Qout']/3600,
          color='gray', alpha=0.6, lw=1.5, label='Qout', zorder=0)

# Concentrations (FOREGROUND)
ax1.plot(timeseries['time'], timeseries['6PPD_Cin'],
         color='tab:blue', label='6PPD Cin', zorder=3)
ax1.plot(timeseries['time'], timeseries['6PPD_Coutmeas'],
         '--', color='tab:blue', label='6PPD Cout', zorder=3)

ax1.plot(timeseries['time'], timeseries['6PPDQ_Cin'],
         color='tab:green', label='6PPDQ Cin', zorder=4)
ax1.plot(timeseries['time'], timeseries['6PPDQ_Coutmeas'],
         '--', color='tab:green', label='6PPDQ Cout', zorder=4)

ax1.set_ylabel('Concentration')
ax1r.set_ylabel('Flow (m³/s)')

ax1.legend(loc='upper left')
ax1r.legend(loc='upper right')
ax1.set_title('Concentration (In/Out) and Flow')


# =========================
# PANEL 2: Mass Flux + Flow
# =========================
ax2 = axes[1]
ax2r = ax2.twinx()

# Ensure correct layering
ax2.set_zorder(2)
ax2.patch.set_visible(False)

# Flow (BACKGROUND)
ax2r.plot(timeseries['time'], timeseries['Qin']/3600,
          color='black', alpha=0.6, lw=1.5, label='Qin', zorder=0)
ax2r.plot(timeseries['time'], timeseries['Qout']/3600,
          color='gray', alpha=0.6, lw=1.5, label='Qout', zorder=0)

# Mass flux (FOREGROUND)
ax2.plot(timeseries['time'], timeseries['Min_6PPD'],
         color='tab:blue', label='Min 6PPD', zorder=3)
ax2.plot(timeseries['time'], timeseries['Mout_6PPD'],
         '--', color='tab:blue', label='Mout 6PPD', zorder=3)

ax2.plot(timeseries['time'], timeseries['Min_6PPDQ'],
         color='tab:green', label='Min 6PPDQ', zorder=4)
ax2.plot(timeseries['time'], timeseries['Mout_6PPDQ'],
         '--', color='tab:green', label='Mout 6PPDQ', zorder=4)

ax2.set_ylabel('Mass per timestep')
ax2r.set_ylabel('Flow (m³/s)')

ax2.legend(loc='upper left')
ax2r.legend(loc='upper right')
ax2.set_title('Mass Flux (Min/Mout) and Flow')


# =========================
# PANEL 3: Cumulative Mass + Cumulative Flow
# =========================
ax3 = axes[2]
ax3r = ax3.twinx()

# ---- Compute cumulative quantities ----
timeseries['CumMin_6PPD'] = timeseries['Min_6PPD'].cumsum()
timeseries['CumMout_6PPD'] = timeseries['Mout_6PPD'].cumsum()

timeseries['CumMin_6PPDQ'] = timeseries['Min_6PPDQ'].cumsum()
timeseries['CumMout_6PPDQ'] = timeseries['Mout_6PPDQ'].cumsum()

timeseries['CumQin'] = (timeseries['Qin'] * timeseries['dt']).cumsum()
timeseries['CumQout'] = (timeseries['Qout'] * timeseries['dt']).cumsum()

# Ensure layering
ax3.set_zorder(2)
ax3.patch.set_visible(False)

# Cumulative flow (BACKGROUND)
ax3r.plot(timeseries['time'], timeseries['CumQin'],
          color='black', alpha=0.6, lw=1.5, label='Cum Qin', zorder=0)
ax3r.plot(timeseries['time'], timeseries['CumQout'],
          color='gray', alpha=0.6, lw=1.5, label='Cum Qout', zorder=0)

# Cumulative mass (FOREGROUND)
ax3.plot(timeseries['time'], timeseries['CumMin_6PPD'],
         color='tab:blue', label='Cum Min 6PPD', zorder=3)
ax3.plot(timeseries['time'], timeseries['CumMout_6PPD'],
         '--', color='tab:blue', label='Cum Mout 6PPD', zorder=3)

ax3.plot(timeseries['time'], timeseries['CumMin_6PPDQ'],
         color='tab:green', label='Cum Min 6PPDQ', zorder=4)
ax3.plot(timeseries['time'], timeseries['CumMout_6PPDQ'],
         '--', color='tab:green', label='Cum Mout 6PPDQ', zorder=4)

ax3.set_ylabel('Cumulative Mass')
ax3r.set_ylabel('Cumulative Flow (m³)')
ax3.set_xlabel('Time')

ax3.legend(loc='upper left')
ax3r.legend(loc='upper right')
ax3.set_title('Cumulative Mass and Flow')


# =========================
plt.tight_layout()
plt.show()



#Testing - reduce Qin 
#timeseries.loc[:,'Qin'] = timeseries.Qin/3600*6
# locsumm.loc['water','Depth'] = timeseries.outlevel_m[0]
# #Initialize the model
# qbl =  StormPond(locsumm,chemsumm,params,timeseries,numc)
# #Define dX
# dx = params.val.dx
# qbsys=qbl.make_system(locsumm,params,numc,timeseries,qbdims,dx=dx)
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
