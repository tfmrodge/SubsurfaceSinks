# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:12:34 2022

@author: trodge01
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
import itertools
#from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
#import hydroeval
import joblib
from joblib import Parallel, delayed
#Testing the modelres
#Load parameterization files
pdb.set_trace()
#numc = ['water', 'subsoil', 'air', 'pond'] #
tstart = time.time()
#For the vancouver tree trench, no ponding zone. 
#numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
#locsumm = pd.read_excel('inputfiles/QuebecSt_TreeTrench.xlsx',index_col = 0)
locsumm = pd.read_excel('inputfiles/Pine8th_BC.xlsx',index_col = 0)
#chemsumm = pd.read_excel('inputfiles/PPD_CHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/params_Pine8th.xlsx',index_col = 0)

#Design Tests
def design_tests(scenario_dict):
    #Re-initialize
    locsumm = pd.read_excel('inputfiles/Pine8th_BC.xlsx',index_col = 0)
    params = pd.read_excel('inputfiles/params_Pine8th.xlsx',index_col = 0)
    #Change underdrain valve opening (fvalve) (set to 0 for no underdrain flow)
    if scenario_dict['fvalve'] == True:  
        params.loc['fvalveopen','val'] = 0
    else: params.loc['fvalveopen','val'] = None
    
    #Amount of Foc in the soil (Foc)
    if scenario_dict['Foc'] == True:   
        #focfactor = 
        locsumm.loc[['subsoil'],'FrnOC'] = 0.4
    
    #Change infiltration rate in system (Kinf)
    if scenario_dict['Kinf'] == True: 
        Kffactor = 0.5
        params.loc['Kf','val'] = Kffactor*params.val.Kf
    #Changedepth of system (Dsys)
    if scenario_dict['Dsys'] == True:     
        Dfactor = 2
        #params.loc['Kf','val'] = Kffactor*params.val.Kf
        locsumm.loc[['subsoil','rootbody','rootxylem','rootcyl'],'Depth'] =locsumm.Depth.subsoil*Dfactor    
    #Change the area of the system (Asys)
    if scenario_dict['Asys'] == True:   
        #pdb.set_trace
        Afactor = 2.0
        locsumm.loc[['water','topsoil','subsoil','air',
                     'drain','drain_pores','native_soil','native_pores'],'Area'] =21.8*Afactor
        params.loc['BC_Area_curve','val'] = np.array2string(np.array(params.val.BC_Area_curve.split(",")
                                                                     ,dtype='float')*Afactor,separator=",")[1:-1]
        params.loc['BC_Volume_Curve','val'] = np.array2string(np.array(params.val.BC_Volume_Curve.split(",")
                                                                     ,dtype='float')*Afactor,separator=",")[1:-1]    
    #Change ponding height (Hp)
    if scenario_dict['Hp'] == True:
        Weirfactor = 2
        params.loc['Hw','val'] = params.val.Hw*Weirfactor
        Ds = np.array(params.val.BC_Depth_Curve.split(","),dtype='float')
        step = (Ds[1]-Ds[0])
        numsteps = int((params.val.Hw - max(Ds))/step)
        Ds = np.concatenate((Ds,np.linspace(max(Ds)+step,params.val.Hw,numsteps)))
        #Change the areas to match - we will just keep using the top area.
        As = np.array(params.val.BC_Area_curve.split(","),dtype='float')
        As = np.concatenate((As,np.zeros(numsteps)+max(As)))
        Vs = np.array(params.val.BC_Volume_Curve.split(","),dtype='float')
        Vs =  np.concatenate((Vs,((np.zeros(numsteps)+max(As))*step).cumsum()+max(Vs)))
        params.loc['BC_Depth_Curve','val'] = np.array2string(Ds,separator=",")[1:-1]
        params.loc['BC_Area_curve','val'] = np.array2string(As,separator=",")[1:-1]
        params.loc['BC_Volume_Curve','val'] = np.array2string(Vs,separator=",")[1:-1]
    return locsumm, params


pp = None
#Define the influent concentration
Cin = 1000 #ng/L
Cin = Cin*1e-6 #Convert to g/m³
#Set up joblib stuff
n_jobs = (joblib.cpu_count()-2)
#Duration is the length of the storm (e.g. 30minutes, 1 hr). Must match the Excel file e.g. 30min, 1hr, 24hr
#Frequency is the recurrence period (e.g. 100-yr storm)
durations = ['10min','30min', '1hr','2hr', '6hr','12hr','24hr']
dur_dict = {'10min':10/60,'30min':30/60, '1hr':1.0,'2hr':2.0, '6hr':6.0,'12hr':12.0,'24hr':24.0}
intensities = [34.581917698987,67.2409410804227,118.178784076204,137.164085563433,20.9467490187692,38.3517651852434,
               64.1705454107794,73.4522738958964,15.2666149456928,26.9112661085693,43.6526658205085,49.5307503897111,
               11.1267639523064,18.8835178789863,29.6951696613895,33.39985413719,6.73963583884412,10.7704656111228,
               16.1243428601216,17.8858425227741,4.91204745584287,7.55758867353632,10.9687481373651,12.0608819103516,
               3.58004657601657,5.3031269603961,7.46160241968732,8.13296171372551]
frequencies = ['2yr','10yr','100yr','200yr']
#Next, we will define the function that will run the model.
def run_IDFs(locsumm,chemsumm,params,numc,Cin,dur_freq):
    #First, we will define the timeseries based on the duration. Probably better to put this i/o step out of the loop but w/e
    timeseries = pd.read_excel('inputfiles/timeseries_IDFstorms.xlsx',sheet_name=dur_freq[0])
    #Test if no underdrain - has to be after timeseries is imported.
    if params.loc['fvalveopen','val'] != None:
        timeseries.loc[timeseries.time>0,'fvalveopen'] = params.loc['fvalveopen','val']
    #try:     
    #except NameError:
    #    pass
    #Define the Qin and rain rate based on the frequency
    timeseries.loc[:,'Qin'] = timeseries.loc[:,dur_freq[1] + '_Qin']
    timeseries.loc[:,'RainRate'] = timeseries.loc[:,dur_freq[1] + '_RainRate']
    timeseries.loc[:,'6PPDQ_Min'] = timeseries.Qin*Cin*1/60 #m3/hr * g/m3*hrs = g
    bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
    #Next, run the model!
    #flow_time = bc.flow_time(locsumm,params,['water','subsoil'],timeseries)
    model_outs = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)
    mass_flux = bc.mass_flux(model_outs,numc) #Run to get mass flux
    denom = mass_flux.N_influent.groupby(level=0).sum()
    #Finally, we will get the model outputs we want to track.
    res = pd.DataFrame(index = chemsumm.index,columns= ['Duration','Frequency','pct_advected','pct_transformed','pct_sorbed','pct_overflow','pct_stormsewer'])
    res.loc[:,'Duration'] = dur_freq[0]
    res.loc[:,'Frequency'] = dur_freq[1]
    
    res.loc[:,'pct_advected'] = np.array((mass_flux.loc[:,['N_effluent','N_exf',
                               'Nadvpond']].sum(axis=1).groupby(level=0).sum())/denom)
    res.loc[:,'pct_transformed'] = np.array((mass_flux.loc[:,['Nrwater','Nrsubsoil','Nrrootbody','Nrrootxylem',
                               'Nrrootcyl','Nrshoots','Nrair','Nrpond']].sum(axis=1).groupby(level=0).sum())/denom)
    res.loc[:,'pct_sorbed'] = np.array(1-res.loc[:,'pct_advected']-res.loc[:,'pct_transformed'])
    res.loc[:,'pct_overflow'] = np.array((mass_flux.loc[:,['Nadvpond']].sum(axis=1).groupby(level=0).sum())/denom)
    res.loc[:,'pct_stormsewer'] = np.array((mass_flux.loc[:,['N_effluent',
                               'Nadvpond']].sum(axis=1).groupby(level=0).sum())/denom)
    #res.loc[:,'pct_Qover'] = flow_time.loc[(slice(None),'pond'),'Q_todrain'].sum()\
    #    /flow_time.loc[(slice(None),'pond'),'Q_in'].sum()
    #Look at Concentration Reduction
    #Storm Sewer Concentration = (Ceff*Qeff+Cpond*Qover)/(Qeff+Qover)
    numx = len(model_outs.index.levels[2])-1
    Q_stormsewer = np.array(model_outs.loc[(slice(None),slice(None),numx-1),'Qout'])\
        + np.array(model_outs.loc[(slice(None),slice(None),numx),'advpond'])
    res.loc[:,'pct_Qover'] = np.array(model_outs.loc[(slice(None),slice(None),numx),'advpond'].sum())\
                                      /np.array(model_outs.loc[(slice(None),slice(None),numx),'Qin'].sum())
    #Mean Effluent Concentration (ng/L)
    res.loc[:,'MEC_ngl'] = 1e6*((mass_flux.loc[:,['N_effluent','Nadvpond']].sum(axis=1).groupby(level=[0,1]).sum())\
                                    /np.array(Q_stormsewer)).groupby(level=0).mean()*chemsumm.loc[:,'MolMass']
    res.loc[:,'MEC_ngl'] = res.loc[:,'MEC_ngl'].fillna(0)
    #Calculate concentration reduction as the average effluent concentration divided by the 
    res.loc[:,'Conc_red'] = 1- res.MEC_ngl/Cin*1e-6
    #Calculate the time-integrated risk quotient = sum((concentration/reference concentration)*dt)/sum(dt)
    ref_conc = 95/298.400 #nmol/L, Coho Salmon LC50 for 6PPD-Q
    res.loc[:,'RQ_av'] = (1e6*(((mass_flux.loc[:,['N_effluent','Nadvpond']].sum(axis=1)*mass_flux.dt).groupby(level=[0,1]).sum())\
                      /np.array(Q_stormsewer))/ref_conc).groupby(level=0).sum()/mass_flux.dt.groupby(level=[0,1]).mean().groupby(level=0).sum()
    res.loc[:,'RQ_sum'] = (1e6*(((mass_flux.loc[:,['N_effluent','Nadvpond']].sum(axis=1)).groupby(level=[0,1]).sum())\
                      /np.array(Q_stormsewer))/ref_conc).groupby(level=0).sum()  
    #Now, we will calculate the flow rate of a receiving body that would be required to keep the risk low
    #For a "high" risk LOC > 0.5
    #res.loc[:,'QLOC_05'] = 2*1e6*mass_flux.dt.groupby(level=[0,1]).mean().groupby(level=0).sum()/(ref_conc)\
    #    *((((mass_flux.loc[:,['N_effluent','Nadvpond']].sum(axis=1)*mass_flux.dt).groupby(level=[0,1]).sum())).groupby(level=0).sum())
    #ldt = ref_conc*1e-6*mass_flux.dt.groupby(level=[0,1]).mean().groupby(level=0).sum()
    #res.loc[:,'QLOC_05'] = 2/ldt*((mass_flux.loc[:,['N_effluent','Nadvpond']].sum(axis=1)*mass_flux.dt).groupby(level=[0,1]).sum().groupby(level=0).sum()\
    #       -np.array(Q_stormsewer)*ldt[0]).groupby(level=0).sum()
    return res


numtrials = (len(durations)*len(frequencies))
dur_freqs = []#np.zeros((numtrials, 2),dtype=str)
#ind = 0
for duration in durations:
    for freq in frequencies:
        dur_freqs.append([str(duration),str(freq)])#[ind,:] = [str(duration),str(freq)]
        #ind += 1
#Set up loop through scenarios
scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 'Asys':False, 'Hp':False}
#import pdb
#params = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 'Asys':False, 'Hp':False}
#list(itertools.combinations(params,6))
#Set up combinations - all possible
#combos = ((1,1,0,0,1,0),(1,1,0,0,0,0),(0,1,0,1,0,0))
#combos = ((0,0,0,0,0,0),(1,0,0,0,0,0),(0,1,0,0,0,0))
#combos = ((0, 0, 1, 0, 1, 0),)
#all possible
#combos = list(itertools.product([0,1],repeat=6))
combos = ((0,0,0,0,0,0),(1,0,0,0,0,0),(0,1,0,0,0,0),(0,0,0,1,0,0),)
#pdb.set_trace()
for combo in combos:
    scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 'Asys':False, 'Hp':False}
    for ind, param in enumerate(scenario_dict):        
        scenario_dict[param] = bool(combo[ind])
    outpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDFouts/'
    filtered = [k for k,v in scenario_dict.items() if v == True]
    outname = 'IDF_'+'_'.join(filtered)+'.pkl'
    #if outname in os.listdir(outpath):
    #    continue #Skip if already done
#for scenario in scenario_dict:
    #scenario_dict[scenario] = True
    locsumm_test, params_test = design_tests(scenario_dict)
    #pdb.set_trace()
    #chemsumm = chemsumm.loc['6PPDQ']
    #for dur_freq in dur_freqs:
        #dur_freq = dur_freqs[10]
    #    res = run_IDFs(locsumm_test,chemsumm,params_test,numc,Cin,dur_freq)
    res = Parallel(n_jobs=n_jobs)(delayed(run_IDFs)(locsumm_test,chemsumm,params_test,numc,Cin,dur_freq) for dur_freq in dur_freqs)
    #codetime = time.time()-tstart
    res = pd.concat(res)
    #For now, just get rid of bromide. Need to make so that it can run one compound but honestly doesn't add that much time
    res = res.loc['6PPDQ',:] 
    res.loc[:,'dur_num'] = res.loc[:,'Duration'].replace(dur_dict)
    res.loc[:,'LogD'] = np.log10(res.loc[:,'dur_num'])
    res.loc[:,'Intensity'] = intensities
    res.loc[:,'LogI'] = np.log10(res.loc[:,'Intensity'])
    #bc.plot_flows(flow_time,Qmeas = timeseries.Qout_meas,compartments=['drain','water'],yvar='Q_out')
    
    res.to_pickle(outpath+outname)
    #scenario_dict[scenario] = False
#'''