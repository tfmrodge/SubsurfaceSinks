# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:12:34 2022
This script will run the model across intensity-duration-frequency curves (as specified in a timeseries file)
or across a water year (timeseries_wateryear)
It can be used to change the design of the system based on a combination of design features (as explained below)

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
from warnings import simplefilter
simplefilter(action="ignore", category= FutureWarning)
#Testing the modelres
#Load parameterization files 
pdb.set_trace()
#numc = ['water', 'subsoil', 'air', 'pond'] #
tstart = time.time()
#For the vancouver tree trench, no ponding z`one. 
#numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
#locsumm = pd.read_excel('inputfiles/Pine8th/QuebecSt_TreeTrench.xlsx',index_col = 0)
locsumm = pd.read_excel('inputfiles/Pine8th/Pine8th_BC.xlsx',index_col = 0)
#chemsumm = pd.read_excel('inputfiles/Pine8th/PPD_CHEMSUMM.xlsx',index_col = 0)
#chemsumm = pd.read_excel('inputfiles/Pine8th/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/Pine8th/EngDesign_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/Pine8th/params_Pine8th.xlsx',index_col = 0)
#params.loc['Kn','val'] = 3.3e-3 #Median value for silty-clayey soil in S. Ontario, good low-permeability number

#Design Tests
def design_tests(scenario_dict):
    #Re-initialize
    locsumm = pd.read_excel('inputfiles/Pine8th/Pine8th_BC.xlsx',index_col = 0)
    params = pd.read_excel('inputfiles/Pine8th/params_Pine8th.xlsx',index_col = 0)
    params.loc['Kn','val'] = 3.3e-3 #Median value for silty-clayey soil in S. Ontario, good low-permeability number
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
        Kffactor = 2.0
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
        #TR20230616 - Change to weir calculations included changing size of system to match geometry
        #above weir. This code no longer works, and isn't necessary - assumning same height of system
        #with just the weir changing 
        #Ds = np.array(params.val.BC_Depth_Curve.split(","),dtype='float')
        #step = (Ds[1]-Ds[0])
        #numsteps = int((params.val.Hw - max(Ds))/step)
        #Ds = np.concatenate((Ds,np.linspace(max(Ds)+step,params.val.Hw,numsteps)))
        #Change the areas to match - we will just keep using the top area.
        #As = np.array(params.val.BC_Area_curve.split(","),dtype='float')
        #As = np.concatenate((As,np.zeros(numsteps)+max(As)))
        #Vs = np.array(params.val.BC_Volume_Curve.split(","),dtype='float')
        #Vs =  np.concatenate((Vs,((np.zeros(numsteps)+max(As))*step).cumsum()+max(Vs)))
        #params.loc['BC_Depth_Curve','val'] = np.array2string(Ds,separator=",")[1:-1]
        #params.loc['BC_Area_curve','val'] = np.array2string(As,separator=",")[1:-1]
        #params.loc['BC_Volume_Curve','val'] = np.array2string(Vs,separator=",")[1:-1]
    
    #Raise underdrain invert to halfway up - not accounting for pipe diameter
    if scenario_dict['hpipe'] == True:  
        params.loc['hpipe','val'] = 0.1 #10cm, half of drain depth
    #Add amendment
    if scenario_dict['amend'] == True:  
        params.loc['famendment','val'] = 0.03
        #10cm, half of drain depth
    #else: params.loc['fvalveopen','val'] = None
    
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
    #timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_IDFstorms.xlsx',sheet_name=dur_freq[0])
    timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_IDFstorms_old.xlsx',sheet_name=dur_freq[0])
    #Test if no underdrain - has to be after timeseries is imported.
    if params.loc['fvalveopen','val'] != None:
        timeseries.loc[timeseries.time>0,'fvalveopen'] = params.loc['fvalveopen','val']
    #try:     
    #except NameError:
    #    pass
    #Define the Qin and rain rate based on the frequency
    timeseries.loc[:,'Qin'] = timeseries.loc[:,dur_freq[1] + '_Qin']
    timeseries.loc[:,'RainRate'] = timeseries.loc[:,dur_freq[1] + '_RainRate']
    #timeseries.loc[:,'Max_ET'] = params.val.max_ET
    #timeseries.loc[timeseries.RainRate>0,'Max_ET'] = 0.
    for compound in chemsumm.index:
        minname = compound+'_Min'
        timeseries.loc[:,minname] = timeseries.Qin*Cin*1/60 #m3/hr * g/m3*hrs = g
    bc = BCBlues(locsumm,chemsumm,params,timeseries,numc)
    #Next, run the model!
    flow_time = bc.flow_time(locsumm,params,['water','subsoil'],timeseries)
    #Check if the system has drained. If not, re-run with more time
    lastt = flow_time.index.max()[0]
    #Check if the system has drained. If not, we will extend to make sure that it has. 
    ramp = np.array([1/60,1/60]) # np.array([0.25,10/60,5/60,2/60,2/60,1/60])
    rampstep = len(ramp)-1
    while flow_time.loc[(lastt,'drain_pores'),'Depth'] > 2*params.val.hpipe:
        flow_time = flow_time[flow_time.time>=0]
        #Calculate the drain rate in m/hr
        drainrate = max(1e-6,(flow_time.loc[(lastt-60,'drain_pores'),'Depth'] - flow_time.loc[(lastt,'drain_pores'),'Depth']))/ramp[rampstep]
        #Max double
        n_extend = min(flow_time.index[-1][0]-flow_time.index[0][0],
                        3*int(np.ceil((flow_time.loc[(lastt,'drain_pores'),'Depth']/drainrate)/ramp[rampstep]))) #hr
        #Extend
        timeseries = timeseries.append(timeseries.iloc[0:n_extend]).reset_index(drop=True)
        timeseries.loc[lastt+1:,'time'] = ramp[rampstep]
        timeseries.loc[lastt+1:,'time'] = timeseries.loc[lastt+1:,'time'].cumsum()+timeseries.loc[lastt,'time']
        #Just copy the last cell over
        timeseries.iloc[lastt+1:,1:] = timeseries.iloc[lastt,1:]
        lastt = timeseries.index.max()
        flow_time = bc.flow_time(locsumm,params,['water','subsoil'],timeseries)
        #Next time, go a little longert between steps
        rampstep = max(0,rampstep-1)
    mask = timeseries.time>=0
    minslice = np.min(np.where(mask))
    maxslice = np.max(np.where(mask))#minslice + 5 #
    flow_time = df_sliced_index(flow_time.loc[(slice(minslice,maxslice),slice(None)),:])
    draintimes = bc.draintimes(timeseries,flow_time)
    #Error checking - output flowtime
    #outpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDFouts/'
    #outname = 'flowtime_EngDesign_lowKn'+'_'.join(dur_freq)+'.pkl' 
    #flow_time.to_pickle(outpath+outname)
    model_outs = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None,flow_time=flow_time)
    #model_outs = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)
    #model_outs = bc.run_it(locsumm,chemsumm,timeseries,numc,params,pp=None,flow_time = flow_time)
    mass_flux = bc.mass_flux(model_outs,numc) #Run to get mass flux
    #mbal = bc.mass_balance(model_outs,numc,mass_flux=mass_flux,normalized=True)
    denom = mass_flux.N_influent.groupby(level=0).sum()
    #Finally, we will get the model outputs we want to track.
    res = pd.DataFrame(index = chemsumm.index,columns= ['Duration','Frequency','pct_advected','pct_transformed','pct_sorbed','pct_overflow','pct_stormsewer'])
    #lastt = max(model_outs.index.levels[1])
    res.loc[:,'Duration'] = dur_freq[0]
    res.loc[:,'Frequency'] = dur_freq[1]
    
    res.loc[:,'pct_advected'] = ((mass_flux.loc[:,['N_effluent','N_exf',
                               'Nadvpond']].sum(axis=1).groupby(level=0).sum())/denom)
    res.loc[:,'pct_transformed'] = ((mass_flux.loc[:,['Nrwater','Nrsubsoil','Nrrootbody','Nrrootxylem',
                               'Nrrootcyl','Nrshoots','Nrair','Nrpond']].sum(axis=1).groupby(level=0).sum())/denom)
    res.loc[:,'pct_vol'] = ((mass_flux.loc[:,['Nadvair']].sum(axis=1).groupby(level=0).sum())/denom)
    #res.loc[:,'pct_sorbed'] = (1-res.loc[:,'pct_advected']-res.loc[:,'pct_transformed'])
    res.loc[:,'pct_sorbed'] = model_outs.loc[(slice(None),lastt,slice(None)),'M2_t1'].groupby(level=0).sum()\
                            /(mass_flux.dt*mass_flux.N_influent).groupby(level=0).sum() 
    res.loc[:,'pct_overflow'] = ((mass_flux.loc[:,['Nadvpond']].sum(axis=1).groupby(level=0).sum())/denom)
    res.loc[:,'pct_stormsewer'] = ((mass_flux.loc[:,['N_effluent',
                               'Nadvpond']].sum(axis=1).groupby(level=0).sum())/denom)
    
    #flow_time = flow_time.loc[(slice(None),'water'),:]
    #flow_time.loc[(slice(None),'water'),'Q_in'].sum()
    #res.loc[:,'pct_Qover'] = flow_time.loc[(slice(None),'pond'),'Q_todrain'].sum()\
    #    /flow_time.loc[(slice(None),'pond'),'Q_in'].sum()
    #Look at Concentration Reduction
    #Storm Sewer Concentration = (Ceff*Qeff+Cpond*Qover)/(Qeff+Qover)
    numx = len(model_outs.index.levels[2])-1
    #dt = model_outs.dt
    Q_stormsewer = np.array(model_outs.loc[(slice(None),slice(None),numx-1),'Qout'])\
        + np.array(model_outs.loc[(slice(None),slice(None),numx),'advpond'])
    res.loc[:,'pct_Qover'] = np.array(model_outs.loc[(slice(None),slice(None),numx),'advpond'].sum())\
                                      /np.array(model_outs.loc[(slice(None),slice(None),numx),'Qin'].sum())
    res.loc[:,'pct_Qexf'] = np.array(model_outs.loc[(slice(None),slice(None),slice(None)),'Qwaterexf'].sum())\
                                    /np.array(model_outs.loc[(slice(None),slice(None),numx),'Qin'].sum())
    res.loc[:,'pct_Qet'] = np.array(model_outs.loc[(slice(None),slice(None),slice(0,numx-1)),'Qet'].sum())\
                                    /np.array(model_outs.loc[(slice(None),slice(None),numx),'Qin'].sum())
    res.loc[:,'pct_Qdrain'] = np.array(model_outs.loc[(slice(None),slice(None),numx-1),'Qout'].sum())\
                                    /np.array(model_outs.loc[(slice(None),slice(None),numx),'Qin'].sum())
    res.loc[:,'draintime'] = np.max(draintimes)
    #dV = model_outs.loc[('6PPDQ',546,slice(0,numx-1)),'Vwater'].sum() - model_outs.loc[('6PPDQ',246,slice(0,numx-1)),'Vwater'].sum()
    # watbal = (dt*model_outs.loc[('6PPDQ',slice(None),numx),'Qin']).sum() - np.array((dt*model_outs.loc[('6PPDQ',slice(None),numx),'advpond']).sum()) \
    #     -np.array((dt*model_outs.loc[('6PPDQ',slice(None),slice(None)),'Qwaterexf']).sum()) \
    #         - np.array((dt*model_outs.loc[('6PPDQ',slice(None),slice(0,numx-1)),'Qet']).sum()) \
    #                    -np.array(dt*model_outs.loc[('6PPDQ',slice(None),numx-1),'Qout'].sum())
    
    #Mean Effluent Concentration (ng/L)
    #res.loc[:,'MEC_ngl'] = 1e6*((mass_flux.loc[:,['N_effluent','Nadvpond']].sum(axis=1).groupby(level=[0,1]).sum())\
    #                                /np.array(Q_stormsewer)).groupby(level=0).mean()*chemsumm.loc[:,'MolMass']
    res.loc[:,'MEC_ngl'] = 1e6*((mass_flux.loc[:,['N_effluent','Nadvpond']].sum(axis=1).groupby(level=[0]).sum())\
                                    /np.array(Q_stormsewer.sum()))*chemsumm.loc[:,'MolMass']
    res.loc[:,'MEC_ngl'] = res.loc[:,'MEC_ngl'].fillna(0)
    #Calculate concentration reduction as the average effluent concentration divided by the 
    res.loc[:,'Conc_red'] = 1- res.MEC_ngl/Cin*1e-6
    #Calculate the time-integrated risk quotient = sum((concentration/reference concentration)*dt)/sum(dt)
    ref_conc = Cin#95/298.400 #nmol/L, Coho Salmon LC50 for 6PPD-Q
    res.loc[:,'RQ_av'] = res.loc[:,'MEC_ngl']/Cin*1e-6
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

def run_wateryears(locsumm,chemsumm,params,numc,timeseries,combo):
    #First, we will define the timeseries based on the duration. Probably better to put this i/o step out of the loop but w/e
    #timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_IDFstorms.xlsx',sheet_name=dur_freq[0])
    #Test if no underdrain - has to be after timeseries is imported.
    timeseries_test = timeseries.copy()
    scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                     'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}
    Cin = 1000
    #timeseries_test.
    for compound in chemsumm.index:
        minname = compound+'_Min'
        timeseries_test.loc[:,minname] = timeseries_test.Qin*Cin*1/60 #m3/hr * g/m3*hrs = g
        timeseries_test.loc[:,minname].fillna(0.,inplace=True)
    for ind, param in enumerate(scenario_dict):        
        scenario_dict[param] = bool(combo[ind])
    locsumm_test, params_test = design_tests(scenario_dict)
    if params_test.loc['fvalveopen','val'] != None:
        timeseries_test.loc[timeseries.time>0,'fvalveopen'] = params_test.loc['fvalveopen','val']
    else:
        timeseries_test.loc[timeseries.time>0,'fvalveopen'] = 0.8
        
    #try:     
    #except NameError:
    #    pass
    #Define the Qin and rain rate based on the frequency
    #timeseries.loc[:,'6PPDQ_Min'] = timeseries.Qin*Cin*1/60 #m3/hr * g/m3*hrs = g
    bc = BCBlues(locsumm_test,chemsumm,params_test,timeseries_test,numc)
    #Next, run the model!
    #flow_time = bc.flow_time(locsumm,params,['water','subsoil'],timeseries)
    model_outs = bc.run_BC(locsumm_test,chemsumm,timeseries_test,numc,params_test,pp=None)
    outpath = 'D:/Users/trodge01/Documents/BigPickles/'
    filtered = [k for k,v in scenario_dict.items() if v == True]
    outname = 'wateryear_'+'_'.join(filtered)+'.pkl'
    model_outs.to_pickle(outpath+outname)
    return model_outs



#import pdb
#params = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 'Asys':False, 'Hp':False}
#list(itertools.combinations(params,6))
#Set up combinations - all possible
#combos = ((1,1,0,0,1,0),(1,1,0,0,0,0),(0,1,0,1,0,0))
#combos = ((0,0,0,0,0,0),(1,0,0,0,0,0),(0,1,0,0,0,0))
#combos = ((0, 0, 1, 0, 1, 0),)
#all possible
combos = list(itertools.product([0,1],repeat=8))
#combos = ((0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0),(0,1,0,0,0,0,0,0),(0,0,1,0,0,0,0,0),(0,0,0,1,0,0,0,0),(0,0,0,0,1,0,0,0),
#          (0,0,0,0,0,1,0,0),(0,0,0,0,0,0,1,0),(0,0,0,0,0,0,0,1))
#          #(1,1,0,0,1,1),(1,1,0,0,0,1))
#combos = ((0,0,0,0,0,0,0,0),)
#pdb.set_trace()
runwateryear = False
if runwateryear == True:
    n_jobs = (joblib.cpu_count()-2)#len(combos)
    timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_wateryear.xlsx')
    for combo in combos:
        run_wateryears(locsumm,chemsumm,params,numc,timeseries,combo)
    #Parallel(n_jobs=n_jobs)(delayed(run_wateryears)(locsumm,chemsumm,params,numc,timeseries,combo) for combo in combos)
else:
    numtrials = (len(durations)*len(frequencies))
    dur_freqs = []#np.zeros((numtrials, 2),dtype=str)
    #ind = 0
    for duration in durations:
        for freq in frequencies:
            dur_freqs.append([str(duration),str(freq)])#[ind,:] = [str(duration),str(freq)]
            #ind += 1
    #Set up loop through scenarios
    #scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
    #                 'Asys':False, 'Hp':False, 'hpipe':False}
    for combo in combos:
        scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                         'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}
        for ind, param in enumerate(scenario_dict):        
            scenario_dict[param] = bool(combo[ind])
            
        outpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDFouts/'
        filtered = [k for k,v in scenario_dict.items() if v == True]
        #outname = 'IDF_EngDesign'+'_'.join(filtered)+'.pkl'
        outname = 'IDF_EngDesign_lowKn'+'_'.join(filtered)+'.pkl'
        #outname = 'IDF_'+'_'.join(filtered)+'.pkl'
        #outname = 'IDF_lowKn'+'_'.join(filtered)+'.pkl'
        if outname in os.listdir(outpath):
            continue #Skip if already done
        if (scenario_dict['fvalve'] == True) & (scenario_dict['hpipe'] ==True):
            continue #Mutually exclusive
        #if (scenario_dict['amend'] != True):
        #    continue
    #for scenario in scenario_dict:
        #scenario_dict[scenario] = True
        locsumm_test, params_test = design_tests(scenario_dict)
        #pdb.set_trace()
        #chemsumm = chemsumm.loc['6PPDQ'] 
        #for dur_freq in dur_freqs: #Failed larger than 12hrs? Unclear why - flow changes didn't seem to work
        # dur_freq = dur_freqs[2]#dur_freqs[23]
        # res = run_IDFs(locsumm_test,chemsumm,params_test,numc,Cin,dur_freq)
        #"""
        res = Parallel(n_jobs=n_jobs)(delayed(run_IDFs)(locsumm_test,chemsumm,params_test,numc,Cin,dur_freq) for dur_freq in dur_freqs)
        #codetime = time.time()-tstart
        res = pd.concat(res)
        #For now, just get rid of bromide. Need to make so that it can run one compound but honestly doesn't add that much time
        #res = res.loc['6PPDQ',:]
        res.loc[:,'dur_num'] = res.loc[:,'Duration'].replace(dur_dict)
        res.loc[:,'LogD'] = np.log10(res.loc[:,'dur_num'])
        for compound in chemsumm.index:
            res.loc[compound,'Intensity'] = intensities
        res.loc[:,'LogI'] = np.log10(res.loc[:,'Intensity'])
        #bc.plot_flows(flow_time,Qmeas = timeseries.Qout_meas,compartments=['drain','water'],yvar='Q_out')
        
        res.to_pickle(outpath+outname)
        #"""
    #scenario_dict[scenario] = False
#'''