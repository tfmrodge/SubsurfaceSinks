# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:47:59 2023

@author: trodge01
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
sns.set_style('ticks')
import matplotlib.pyplot as plt
#This is the name of the python module containing the Bioretention Blues submodel.
from BioretentionBlues import BCBlues
import os
import itertools
import pdb
#Plot system performance on IDF curves. This plots the actual values 
#Inputs
inpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDFouts/'
#outpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDF_results.pkl'
#outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/IDF_nodrain.pkl'
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
locsumm = pd.read_excel('inputfiles/Pine8th/Pine8th_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/Pine8th/EngDesign_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/Pine8th/params_Pine8th.xlsx',index_col = 0)
timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_IDFstorms.xlsx')
#Load data
defname = 'IDF_EngDesign.pkl'
#defname = 'IDF_.pkl'
defdf = pd.read_pickle(inpath+defname)
#scenarios = ['fvalve', 'Foc', 'Kinf', 'Dsys', 'Asys', 'Hp']
scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                         'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}
#combos = ((0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0),(0,1,0,0,0,0,0,0),(0,0,1,0,0,0,0,0),(0,0,0,1,0,0,0,0),(0,0,0,0,1,0,0,0),
#          (0,0,0,0,0,1,0,0),(0,0,0,0,0,0,1,0),(0,0,0,0,0,0,0,1))
#combos = ((1,0,0,0,0,0,0),)#(0,1,0,0,0,0),(0,0,1,0,0,0),(0,0,0,1,0,0),(0,0,0,0,1,0),(0,0,0,0,0,1),
#          (1,1,0,0,1,1),(1,1,0,0,0,1))
combos = ((0,0,0,0,0,0,0,0,0),)
#for scenario in scenarios:
#combos = list(itertools.product([0,1],repeat=7))
#combos = combos[:38]
#combos = ((0,0,0,0,0,0,1),(0,0,1,0,0,0,0))
#combos =((0,1,0,0,0,0,0),)
for combo in combos:
    pdb.set_trace()
    scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                         'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}
    for ind, param in enumerate(scenario_dict):
        #pdb.set_trace()
        scenario_dict[param] = bool(combo[ind])
    filtered = [k for k,v in scenario_dict.items() if v == True]
    testname = 'IDF_EngDesign'+'_'.join(filtered)+'.pkl'
    #testname = 'IDF_EngDesign_lowKn'+'_'.join(filtered)+'.pkl'
    #testname = 'IDF_'+'_'.join(filtered)+'.pkl'
    #testname = 'IDF_defaults'
    try:
        pltdf = pd.read_pickle(inpath+testname)
    except FileNotFoundError:
        continue
    #Define the x and y axes
    #pdb.set_trace()
    xticks = [0.25,0.5,1,3, 6,12,24]
    xticks = [np.log10(xticks),xticks]
    yticks = [5,10,25,50,100]
    yticks = [np.log10(yticks),yticks]
    #Define the variables to plot
    for compound in pltdf.index.unique(): #['PFOA']:#
        pltdata = pltdf.loc[compound,:].copy(deep=True)
        defdata = defdf.loc[compound,:].copy(deep=True)
        pltvars=['pct_stormsewer','LogD','LogI']
        #pltvars=['pct_advected','LogD','LogI']
        pltvars_delta = pltvars.copy()
        delname = pltvars[0] +'_delta'
        pltvars_delta[0] = delname
        pltdata.loc[:,delname] = defdata.loc[:,pltvars[0]]# - pltdata.loc[:,pltvars[0]]
        #pdb.set_trace()
        #pltvars=['RQ_av','LogD','LogI']
        #Determine the average risk quotients, sum as a % of the base-case, av as actual value
        bcRQsum = defdata.RQ_sum.sum()
        RQs = [pltdata.MEC_ngl.mean(),pltdata.MEC_ngl.min(),pltdata.MEC_ngl.max(),
               pltdata.pct_advected.mean(),pltdata.pct_advected.min(),pltdata.pct_advected.max()]
        #pdb.set_trace()
        #Define other parameters
        #Limit of interpolation - values outside of these limits will be set to these. Use "none" for no limits
        interplims = [0.,1.]
        #interplims = [0.,3.5]
        vlims = [0.0,1.0]#[0.15,3.5]#
        #pdb.set_trace()
        #define the colormap - default is brown-blue
        cmap = None
            #cmap = sns.light_palette("seagreen", as_cmap=True)
        #cmap = sns.cubehelix_palette(start=.75, rot=-.5,light=0.85, as_cmap=True)
        #cmap = sns.cubehelix_palette(n_colors = 7,start=1.40, rot=-0.9,gamma = 0.3, hue = 0.9, dark=0.1, light=.95,as_cmap=True,reverse=True)
        bc = BCBlues(locsumm,chemsumm,params,timeseries,numc) 
        fig,ax = bc.plot_idfs(pltdata,pltvars=pltvars_delta,cmap=cmap,vlims=vlims,interplims=interplims,
                              xticks=xticks,yticks=yticks,figsize=(6,4))
        ax.set_xlabel('Event Duration (hrs)')
        ax.set_ylabel('Intensity (mm/hr)')
        #figname = 'IDF_delta_'+str(scenario)
        figname = compound+'_IDF_EngDesign'+'_'.join(filtered)
        ax.set_title(figname)
        #Annotate the risk quotients
        ax.annotate('MEC='+f'{RQs[0]:.2f}',xy= (np.log10(0.2),np.log10(5)),color = 'k')
        ax.annotate('('+f'{RQs[1]:.2f}',xy= (np.log10(0.60),np.log10(5)),color = 'k')
        ax.annotate('-'+f'{RQs[2]:.2f}',xy= (np.log10(0.95),np.log10(5)),color = 'k')
        ax.annotate('% Advected='+f'{RQs[3]:.0%}',xy= (np.log10(0.2),np.log10(4)),color = 'k')   
        ax.annotate('('+f'{RQs[4]:.0%}',xy= (np.log10(0.85),np.log10(4)),color = 'k')   
        ax.annotate('-'+f'{RQs[5]:.0%}',xy= (np.log10(1.3),np.log10(4)),color = 'k')   
        figpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Manuscript/Figs/Pythonfigs/'
        #
        fig.savefig(figpath+figname+'.pdf',format='pdf')
        #xl = ax.get_xlim()
    #fig.savefig()


'''
import pandas as pd
import numpy as np
import os
import seaborn as sns
sns.set_style('ticks')
import matplotlib.pyplot as plt
#This is the name of the python module containing the Bioretention Blues submodel.
from BioretentionBlues import BCBlues
import os
import itertools
import pdb
#Plot system performance on IDF curves. This plots the change from bnase case values. 
#Inputs
inpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDFouts/'
#outpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDF_results.pkl'
#outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/IDF_nodrain.pkl'
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
locsumm = pd.read_excel('inputfiles/Pine8th/Pine8th_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/Pine8th/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/Pine8th/params_Pine8th.xlsx',index_col = 0)
timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_IDFstorms.xlsx')
#Load data
defname = 'IDF_EngDesign.pkl'
#defname = 'IDF_.pkl'
defdf = pd.read_pickle(inpath+defname)
#scenarios = ['fvalve', 'Foc', 'Kinf', 'Dsys', 'Asys', 'Hp']
scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 'Asys':False, 'Hp':False,'hpipe':False}#
combos = ((1,0,0,0,0,0,0),(0,1,0,0,0,0,0),(0,0,1,0,0,0,0),(0,0,0,1,0,0,0),(0,0,0,0,1,0,0),
          (0,0,0,0,0,1,0),(0,0,0,0,0,0,1),(1,1,0,1,1,1,0),(1,1,1,0,0,1,0))
#combos = ((1,0,0,0,0,0),(0,1,0,0,0,0),(0,0,1,0,0,0),(0,0,0,1,0,0),(0,0,0,0,1,0),(0,0,0,0,0,1),
#          (1,1,0,0,1,1),(1,1,0,0,0,1))
#for scenario in scenarios:
#combos = list(itertools.product([0,1],repeat=7))
#combos = combos[:38]
#combos = ((0,0,0,0,0,0,1),(0,0,1,0,0,0,0))
#combos =((1,1,0,1,1,1,0),)
for combo in combos:
    #pdb.set_trace()
    scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 'Asys':False, 'Hp':False,'hpipe':False}
    for ind, param in enumerate(scenario_dict):
        #pdb.set_trace()
        scenario_dict[param] = bool(combo[ind])
    filtered = [k for k,v in scenario_dict.items() if v == True]
    testname = 'IDF_EngDesign'+'_'.join(filtered)+'.pkl'
    #testname = 'IDF_'+'_'.join(filtered)+'.pkl'
    #testname = 'IDF_defaults'
    try:
        pltdf = pd.read_pickle(inpath+testname)
    except FileNotFoundError:
        continue
    #Define the x and y axes
    #pdb.set_trace()
    xticks = [0.25,0.5,1,3, 6,12,24]
    xticks = [np.log10(xticks),xticks]
    yticks = [5,10,25,50,100]
    yticks = [np.log10(yticks),yticks]
    #Define the variables to plot
    for compound in pltdf.index.unique():
        pltdata = pltdf.loc[compound,:]
        defdata = defdf.loc[compound,:]
        pltvars=['pct_stormsewer','LogD','LogI']
        pltvars_delta = pltvars.copy()
        delname = pltvars[0] +'_delta'
        pltvars_delta[0] = delname
        pltdata.loc[:,delname] = defdata.loc[:,pltvars[0]] - pltdata.loc[:,pltvars[0]]
        #pdb.set_trace()
        #pltvars=['RQ_av','LogD','LogI']
        #Determine the average risk quotients, sum as a % of the base-case, av as actual value
        bcRQsum = defdata.RQ_sum.sum()
        RQs = [pltdata.RQ_sum.sum()/bcRQsum,pltdata.RQ_av.mean()]
        #Define other parameters
        #Limit of interpolation - values outside of these limits will be set to these. Use "none" for no limits
        hilim = np.round(2*max(pltdata.loc[:,delname]),decimals=2)
        lolim = np.round(2*min(pltdata.loc[:,delname]),decimals=2)
        interplims = [lolim,hilim]
        #interplims = [0.,3.5]
        vlims = [-0.4,0.4]
        #vlims = [lolim,hilim]#[0.15,3.5]#
        #pdb.set_trace()
        #define the colormap - default is brown-blue
        #cmap = None
        #if lolim<0:
        #    cmap = sns.diverging_palette(250, 30, l=40,s=80,center="light", as_cmap=True)
        #else: 
        #    cmap = sns.light_palette('#8f4e27', as_cmap=True)
        cmap = sns.diverging_palette(250, 30, l=40,s=80,center="light", as_cmap=True)
            #cmap = sns.light_palette("seagreen", as_cmap=True)
        #cmap = sns.cubehelix_palette(start=.75, rot=-.5,light=0.85, as_cmap=True)
        #cmap = sns.cubehelix_palette(n_colors = 7,start=1.40, rot=-0.9,gamma = 0.3, hue = 0.9, dark=0.1, light=.95,as_cmap=True,reverse=True)
        bc = BCBlues(locsumm,chemsumm,params,timeseries,numc) 
        fig,ax = bc.plot_idfs(pltdata,pltvars=pltvars_delta,cmap=cmap,vlims=vlims,interplims=interplims,
                              xticks=xticks,yticks=yticks,figsize=(6,4))
        ax.set_xlabel('Event Duration (hrs)')
        ax.set_ylabel('Intensity (mm/hr)')
        #figname = 'IDF_delta_'+str(scenario)
        figname = compound+'_IDF_EngDesign''_'.join(filtered)
        ax.set_title(figname)
        #Annotate the risk quotients
        ax.annotate('ΣRQ='+f'{RQs[0]:.0%}',xy= (np.log10(0.2),np.log10(5)),color = 'k')
        ax.annotate('RQav='+f'{RQs[1]:.2f}',xy= (np.log10(0.2),np.log10(4)),color = 'k')           
        figpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Manuscript/Figs/Pythonfigs/'
        #
        #fig.savefig(figpath+figname+'.pdf',format='pdf')
    #fig.savefig()
'''