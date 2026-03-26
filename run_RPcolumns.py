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
from HelperFuncs import df_sliced_index, make_input_timeseries
import pdb
import time
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
import hydroeval

#Testing the model
#Load parameterization files
pdb.set_trace()
#numc = ['water', 'subsoil', 'air', 'pond'] #
codetime = time.time()
pklpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/'
#For the vancouver tree trench, no ponding zone. 
#numc = ['water', 'subsoil','topsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air']
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
#locsumm = pd.read_excel('inputfiles/QuebecSt_TreeTrench.xlsx',index_col = 0)
locsumm = pd.read_excel('inputfiles/RPColumns/RPColumn_BC.xlsx',index_col = 0)
#chemsumm = pd.read_excel('inputfiles/Pine8th/EngDesign_CHEMSUMM.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/RPColumns/TrOC_column_CHEMSUMM.xlsx',index_col = 0)
chemsumm = chemsumm.dropna(how='all')
#Change to episuite version
#chemsumm.loc['6PPDQ','LogKocW'] = 3.928
#chemsumm.loc['Rhodamine','chemcharge'] = 0
#chemsumm = pd.read_excel('inputfiles/Kortright_ALL_CHEMSUMM.xlsx',index_col = 0)
params =  pd.read_excel('inputfiles/RPColumns/params_columns.xlsx',index_col = 0)
#params.loc['f_apo','val'] = 0
pp = None
#testing the model
timeseries = pd.read_excel('inputfiles/RPColumns/timeseries_ref.xlsx')
coldata = pd.read_excel('inputfiles/RPColumns/GeneralParameters.xlsx', sheet_name=None)
colnames = ['A','B','C','P','Q','R','X','Y','Z']

def build_soil_column_tables(dict_of_excel_sheets, colnames):
    """
    Stage 1: Extract soil-column parameter tables.
    Each parameter sheet may have different sampling times.
    Output time vector is the union of all times.

    Returns
    -------
    soil_raw : dict of DataFrames
        Keys: soil column names ('A','B',...)
        Each DF contains columns:
           time, T_C, DO_mg_L, EC_uS, pH, FlowRate_ml_min
    """

    wb = dict_of_excel_sheets
    columns = colnames

    # Rename time columns so they all share "time"
    df_T  = wb['Temperature'] .rename(columns={'time point (hrs)': 'time'})
    df_DO = wb['DO']          .rename(columns={'time point (hrs)': 'time'})
    df_EC = wb['EC']          .rename(columns={'time point (hrs)': 'time'})
    df_pH = wb['pH']          .rename(columns={'time point (hrs)': 'time'})
    df_Q  = wb['Flow Rate']   .rename(columns={'time point (hrs)': 'time'})

    soil_raw = {}

    for col in columns:

        # Extract parameter-specific sub-dataframes
        T_sub  = df_T[['time', col]].rename(columns={col: 'T_C'})
        DO_sub = df_DO[['time', col]].rename(columns={col: 'DO_mg_L'})
        EC_sub = df_EC[['time', col]].rename(columns={col: 'EC_uS'})
        pH_sub = df_pH[['time', col]].rename(columns={col: 'pH'})

        fr_col = f"{col}_flow_ml_min"
        Q_sub  = df_Q[['time', fr_col]].rename(columns={fr_col: 'FlowRate_ml_min'})

        # Build a master time vector: union of all times
        all_times = pd.concat([
            T_sub['time'], DO_sub['time'], EC_sub['time'],
            pH_sub['time'], Q_sub['time']
        ]).unique()
        
        
        for df_param in [T_sub, DO_sub, EC_sub, pH_sub, Q_sub]:
            for c in df_param.columns:
                    df_param[c] = pd.to_numeric(df_param[c], errors='coerce')
                


        all_times = np.sort(all_times)

        # Build final dataframe
        df = pd.DataFrame({'time': all_times})

        # Merge each parameter separately (left merge preserves master time)
        df = df.merge(T_sub,  on='time', how='left')
        df = df.merge(DO_sub, on='time', how='left')
        df = df.merge(EC_sub, on='time', how='left')
        df = df.merge(pH_sub, on='time', how='left')
        df = df.merge(Q_sub,  on='time', how='left')

        soil_raw[col] = df

    return soil_raw

def build_model_ready_timeseries(
        soil_raw,
        df_ref,
        dt,
        time_bnds,
        time_col='time'
    ):
    """
    Stage 3: Build fully model-ready timeseries for each soil column.

    Parameters
    ----------
    soil_raw : dict of DataFrames
        Output from Stage 1. Keys are soil column IDs.
    df_ref : DataFrame
        Reference dataframe for constant parameters (RainRate, etc.)
        Must contain at least one row.
    dt : float
        Desired timestep (hours).
    time_bnds : [tmin, tmax]
        Explicit interpolation bounds.
    time_col : str
        Time column name ('time').

    Returns
    -------
    model_ts : dict of DataFrames
        Keys: soil column IDs
        Values: DataFrames with required model columns.
    """

    model_ts = {}

    for col, df_raw in soil_raw.items():

        df = df_raw.copy()

        # --- Convert flow to m^3/hr ---
        df['FlowRate_m3_hr'] = df['FlowRate_ml_min'] * 6e-5

        # --- Prepare columns for model ---
        # Using raw names; will map to model names after interpolation
        interp_cols = [
            'FlowRate_m3_hr',
            'T_C',
            'pH',
            'EC_uS',
            'DO_mg_L'
        ]

        # --- Interpolate onto uniform grid ---
        df_interp = make_input_timeseries(
            df=df,
            dt=dt,
            df_ref=df_ref,
            time_bnds=time_bnds,
            time_col=time_col,
            interp_cols=interp_cols,
            constant_from_ref=True
        )

        # --- Build final model-ready structure ---
        model_df = pd.DataFrame()
        model_df['time'] = df_interp['time']

        # Flow → Qin and Qout_meas
        model_df['Qin']        = df_interp['FlowRate_m3_hr']
        model_df['Qout_meas']  = df_interp['FlowRate_m3_hr']

        # Meteorology + valve state (constant from df_ref)
        model_df['RainRate']   = df_ref.loc[0, 'RainRate']
        model_df['WindSpeed']  = df_ref.loc[0, 'WindSpeed']
        model_df['RH']         = df_ref.loc[0, 'RH']
        model_df['fvalveopen'] = df_ref.loc[0, 'fvalveopen']

        # Temperatures (all equal)
        model_df['Tair']     = df_interp['T_C']
        model_df['Twater']   = df_interp['T_C']
        model_df['Tsubsoil'] = df_interp['T_C']

        # pH
        model_df['pHwater']   = df_interp['pH']
        model_df['pHsubsoil'] = df_interp['pH']

        # Conductivity
        model_df['Condwater'] = df_interp['EC_uS']

        # Dissolved oxygen
        model_df['DO_mgL']    = df_interp['DO_mg_L']

        # Save
        model_ts[col] = model_df

    return model_ts


soil_raw = build_soil_column_tables(coldata,colnames)
model_ts = build_model_ready_timeseries(soil_raw,df_ref=)





# durations = ['10min','30min', '1hr','2hr', '6hr','12hr','24hr']
# dur_dict = {'10min':10/60,'30min':30/60, '1hr':1.0,'2hr':2.0, '6hr':6.0,'12hr':12.0,'24hr':24.0}
# intensities = [34.581917698987,67.2409410804227,118.178784076204,137.164085563433,20.9467490187692,38.3517651852434,
#                64.1705454107794,73.4522738958964,15.2666149456928,26.9112661085693,43.6526658205085,49.5307503897111,
#                11.1267639523064,18.8835178789863,29.6951696613895,33.39985413719,6.73963583884412,10.7704656111228,
#                16.1243428601216,17.8858425227741,4.91204745584287,7.55758867353632,10.9687481373651,12.0608819103516,
#                3.58004657601657,5.3031269603961,7.46160241968732,8.13296171372551]
# frequencies = ['2yr','10yr','100yr','200yr']
# dur_freqs = []#np.zeros((numtrials, 2),dtype=str)
# #ind = 0
# for duration in durations:
#     for freq in frequencies:
#         dur_freqs.append([str(duration),str(freq)])
# dur_freq = dur_freqs[23]
# timeseries = pd.read_excel('inputfiles/Pine8th/timeseries_IDFstorms_old.xlsx',sheet_name=dur_freq[0])
# timeseries.loc[:,'Qin'] = timeseries.loc[:,dur_freq[1] + '_Qin']
# timeseries.loc[:,'RainRate'] = timeseries.loc[:,dur_freq[1] + '_RainRate']


# Cin = 1000 #ng/L
# for compound in chemsumm.index:
#     minname = compound+'_Min'
#     timeseries.loc[:,minname] = timeseries.Qin*Cin*1/60 #m3/hr * g/m3*hrs = g
#Cin = Cin*1e-6 #Convert to g/m³
#timeseries.loc[:,'6PPDQ_Min'] = timeseries.Qin*Cin*1/60 
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_short.xlsx')
#timeseries = pd.read_excel('inputfiles/timeseries_wateryear.xlsx')
#Run only for the first event
#timeseries = timeseries[timeseries.time<=6]
#timeseries = timeseries[timeseries.time<=240]
#timeseries = pd.read_excel('inputfiles/timeseries_Pine8th_simstorm.xlsx')
#Import a flows if you want it
#flowpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/flowtest.pkl'
#flow_time = pd.read_pickle(flowpath)
#Instantiate the model
bc =  BCBlues(locsumm,chemsumm,params,timeseries,numc)
pklpath = 'D:/OneDrive - UBC/Postdoc/Completed Projects/6PPD_BC Papers/Modeling/Pickles/'
timedfname = 'mod_timeseries.pkl'
#How much should we modify the time-step. Multiply the index by this number. 
indfactor = 1#1#3#'Load' #3#3
if indfactor == 'Load':
    timeseries = pd.read_pickle(pklpath+timedfname)
else:
    try: 
        int(indfactor) == indfactor
        timeseries = bc.modify_timestep(timeseries,indfactor)
    except TypeError:
        pass
    
#pdb.set_trace()
calcflow = True #'Load' #True #
flowname = 'RPflowtest.pkl'
if calcflow is True:
    flow_time = bc.flow_time(locsumm,params,['water','subsoil'],timeseries)
    mask = timeseries.time>=0
    minslice = np.min(np.where(mask))
    maxslice = np.max(np.where(mask))#minslice + 5 #
    flow_time = df_sliced_index(flow_time.loc[(slice(minslice,maxslice),slice(None)),:])
    flow_time.to_pickle(pklpath+flowname)
    try:
        bc.plot_flows(flow_time.loc[flow_time.time<6],Qmeas = timeseries.loc[timeseries.time<6,'Qout_meas'],
                      compartments=['drain','water'],yvar='Q_todrain')
        #Plot latter event
        bc.plot_flows(flow_time.loc[flow_time.time>140],Qmeas = timeseries.loc[timeseries.time>140,'Qout_meas'],
                      compartments=['drain','water'],yvar='Q_todrain')
        #% infiltrated - actual was ~78%
        inf_pct = 1 - (flow_time.loc[(slice(None),'drain'),'Q_todrain'].sum()/flow_time.loc[(slice(None),'pond'),'Q_in'].sum())
        #Calculate KGE for the hydrology
        KGE_hydro = hydroeval.evaluator(kge, np.array(flow_time.loc[(slice(None),'drain'),'Q_todrain']),\
                              np.array(timeseries.loc[timeseries.time>=0,'Qout_meas']))
    except KeyError:
        pass
    #flow_time.to_pickle(flowpath)
elif calcflow is None:
    pass
else:
    flow_time = pd.read_pickle(pklpath+flowname)

#codetime = time.time() - codetime

#'''
#Input calculations
#inpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/inputspiketest_30s.pkl'  
inname = 'inputspiketest.pkl'    
#inpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/2014_inputs.pkl'   
calcinp = True#False#
if calcinp is True:
    input_calcs = bc.input_calc(locsumm,chemsumm,params,pp,numc,timeseries,flow_time=flow_time)
    #input_calcs.to_pickle(pklpath+inname)
elif calcinp == None:
    pass
else:
    input_calcs = pd.read_pickle(pklpath+inname)
    
#
#input_calcs = pd.read_pickle(inpath)

runall = False#None#None#'Load'#'Load'#
if runall is True:
    res = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)
elif runall == 'Load':
    outname = 'outputspiketest.pkl'
    res = pd.read_pickle(pklpath+outname)
elif runall == None:
    pass
else:
    res = bc.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs=input_calcs)

if runall != None:
    print(time.time()-codetime)
    outname = 'outputspiketest_EPI.pkl'
    res.to_pickle(pklpath+outname)
    mass_flux = bc.mass_flux(res,numc) #Run to get mass flux
    mbal = bc.mass_balance(res,numc,mass_flux)
    Couts = bc.conc_out(numc,timeseries,chemsumm,res,mass_flux)
    recovery = mass_flux.N_effluent.groupby(level=0).sum()/mass_flux.N_influent.groupby(level=0).sum()
    #bc.plot_Couts(res,Couts.loc[Couts.time<6],multfactor=1e6)
#res = bc.run_it(locsumm,chemsumm,params,pp,numc,timeseries,input_calcs=input_calcs)
#res = bc.run_BC(locsumm,chemsumm,timeseries,numc,params,pp=None)

#bc.plot_Couts(res,Couts,multfactor=1000)


KGE = {}
for ind,chem in enumerate(chemsumm.index):
    try:
        KGE[chem] = (hydroeval.evaluator(kge, np.array(Couts.loc[:,chem+'_Coutest']),\
                          np.array(Couts.loc[:,chem+'_Coutmeas'])))
    except KeyError:
        pass
codetime = time.time() - codetime
plotfig = True
if plotfig == True:
    #Calculate the masses
    mbal_cum = bc.mass_balance_cumulative(numc, mass_balance = mbal,normalized=True)
    compound = 'Rhodamine'#use same name as in chemsumm
    #Set time (hrs), any time after end gives the end.
    time = 8760#1000#6#1000#6#1000
    fig,ax = bc.BC_fig(numc,mass_balance=mbal_cum,time = time,compound=compound,figheight=6,fontsize=7,dpi=300)

#'''