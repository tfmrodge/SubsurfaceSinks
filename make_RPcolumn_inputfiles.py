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
#Functions
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
    df_Br = wb['Tracer Test'] .rename(columns={'time point (hrs)': 'time'})

    soil_raw = {}

    for col in columns:

        # Extract parameter-specific sub-dataframes
        T_sub  = df_T[['time', col]].rename(columns={col: 'T_C'})
        DO_sub = df_DO[['time', col]].rename(columns={col: 'DO_mg_L'})
        EC_sub = df_EC[['time', col]].rename(columns={col: 'EC_uS'})
        pH_sub = df_pH[['time', col]].rename(columns={col: 'pH'})
        Br_sub = df_Br[['time', col]].rename(columns={col: 'C_Br_mg_L'})

        fr_col = f"{col}_flow_ml_min"
        Q_sub  = df_Q[['time', fr_col]].rename(columns={fr_col: 'FlowRate_ml_min'})

        # Build a master time vector: union of all times
        all_times = pd.concat([
            T_sub['time'], DO_sub['time'], EC_sub['time'],
            pH_sub['time'], Q_sub['time'], Br_sub['time']
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
        df = df.merge(Br_sub,  on='time', how='left')

        soil_raw[col] = df

    return soil_raw



def build_model_ready_timeseries(
        soil_raw,
        chem_unified,
        df_ref,
        dt,
        time_bnds,
        time_col='time',
        var_map = None,
    ):
    """
    Stage 3: Build fully model-ready timeseries for each soil column,
             integrating soil parameters + all chemicals.
    """

    model_ts = {}

    for soilcol, df_raw in soil_raw.items():

        df = df_raw.copy()

        # Hydraulics conversion
        df["FlowRate_m3_hr"] = df["FlowRate_ml_min"] * 6e-5
      
        # --- Merge unified chemical table for this soil column ---
        df_chem = chem_unified[soilcol].copy()
        chem_tmin = df_chem['time'].min()
        
        # Round time columns to avoid floating-point mismatches
        df[time_col] = df[time_col].round(3)
        df_chem[time_col] = df_chem[time_col].round(3)
        
        # Build full union time axis
        t_union = np.sort(
            np.unique(
                np.concatenate([
                    df[time_col].dropna().values,
                    df_chem[time_col].dropna().values
                ])
            )
        )
        
        # Reindex soil and chemical dataframes on union time axis
        df = df.set_index(time_col).reindex(t_union).reset_index().rename(columns={"index": time_col})
        df_chem = df_chem.set_index(time_col).reindex(t_union).reset_index().rename(columns={"index": time_col})
        
        # For each chemical column: zero-out values before first time step
        chem_cols = [c for c in df_chem.columns if c.endswith("_Cout") or c.endswith("_Cin")]
        df_chem.loc[df_chem.time < chem_tmin,chem_cols] = 0.
        
        # Now merge back into main df (safe: no duplicate times)
        df = df.merge(df_chem, on=time_col, how="left")


        # Identify chemical columns to interpolate and enforce non-negativity
        chem_cols = [c for c in df.columns if c.endswith("_Cout") or c.endswith("_Cin")]

        interp_cols = ["FlowRate_m3_hr", "T_C", "pH", "EC_uS", "DO_mg_L"] + chem_cols
        nonneg_cols = ["FlowRate_m3_hr", "EC_uS", "DO_mg_L"] + chem_cols

        # Interpolate using Stage-2
        df_interp = make_input_timeseries(
            df=df,
            dt=dt,
            df_ref=df_ref,
            time_bnds=time_bnds,
            time_col=time_col,
            interp_cols=interp_cols,
            constant_from_ref=True,
            nonneg_cols=nonneg_cols
        )

        # ---- Build the model-ready timeseries ----
        model_df = pd.DataFrame()
        model_df["time"] = df_interp["time"]

        # Core physical & met data via var_map
        for outcol, spec in var_map.items():
            if spec["source"] == "interp":
                model_df[outcol] = df_interp[spec["column"]]
            else:  # reference value
                model_df[outcol] = df_ref.loc[0, spec["column"]]

        # Add chemicals (auto)
        for chemcol in chem_cols:
            model_df[f"{chemcol}_mg_L"] = df_interp[chemcol]

        # store output
        model_ts[soilcol] = model_df

    return model_ts

def clean_chemical_series(
        ser,
        MDL,
        treat_pre_detection_as_zero=True
    ):
    """
    Clean a chemical concentration series with censoring logic.

    Rules:
    - 'N/F' or '<MDL' --> censored values
    - Censored values:
        * Before first detection: 0 if flag is True
        * After first detection: MDL/2
    - Blank values (empty Excel cells) stay NaN
    """

    # Convert to string for consistent processing of N/F
    ser_str = ser.astype(str).str.strip()

    # Identify censored values
    censored_mask = ser_str.isin(["N/F", "<MDL"])

    # Convert everything to numeric, forcing errors to NaN
    ser_num = pd.to_numeric(ser_str, errors="coerce")

    # Apply MDL/2 to censored values
    ser_num.loc[censored_mask] = 0.5 * MDL

    if treat_pre_detection_as_zero:
        # Detection is defined as > MDL
        first_detection_idx = ser_num[ser_num > MDL].index.min()

        if pd.notna(first_detection_idx):
            # All censored values before detection become 0
            early_censored = censored_mask & (ser.index < first_detection_idx)
            ser_num.loc[early_censored] = 0.0

    # Enforce non-negative
    ser_num = ser_num.clip(lower=0)

    return ser_num

def load_chemical_sheets(
        wb_dict,
        chemical_sheets,
        colnames,
        MDLdict,
        treat_pre_detection_as_zero=True
    ):
    """
    Build a unified chemical dataframe for each soil column, with:
      - unified time axis
      - MDL handling
      - censored value logic (<MDL => MDL/2, or 0 before first detection)
      - effluent (A,B,C,...) and influent (InA,InB,...) columns
      - name style: Chemical_Cout, Chemical_Cin

    Returns
    -------
    chem_all : dict of DataFrames
        chem_all[col] = dataframe with time + all chemicals for that soil column
    """

    # Initialize empty dicts for each soil column
    chem_all = {col: {} for col in colnames}

    # First pass: create a master time union per soil column
    time_union = {col: [] for col in colnames}

    # --- Helper: clean censored data ---
    def clean_series(ser, MDL):
        ser_str = ser.astype(str).str.strip()

        # Identify censored values
        censored_mask = ser_str.isin(["N/F", "<MDL"])

        # Convert to numeric, coerce blanks to NaN
        ser_num = pd.to_numeric(ser_str, errors="coerce")

        # Apply MDL replacement
        ser_num.loc[censored_mask] = 0.5 * MDL

        # Optional: treat pre-detection (<MDL) as zero until first true detection
        if treat_pre_detection_as_zero:
            first_detect_idx = ser_num[ser_num > MDL].index.min()
            if pd.notna(first_detect_idx):
                early = censored_mask & (ser.index < first_detect_idx)
                ser_num.loc[early] = 0.0

        return ser_num.clip(lower=0)

    # --- Load each chemical sheet ---
    for sheet in chemical_sheets:

        chemical_name = sheet
        MDL = MDLdict[chemical_name]

        df_sheet = wb_dict[sheet].copy()
        df_sheet = df_sheet.rename(columns={"time point (hrs)": "time"})
        df_sheet["time"] = pd.to_numeric(df_sheet["time"], errors="coerce")

        # For each soil column, pull effluent & influent
        for col in colnames:

            # Effluent column
            if col in df_sheet.columns:
                ser = clean_series(df_sheet[col], MDL)
                df_tmp = pd.DataFrame({"time": df_sheet["time"], f"{chemical_name}_Cout": ser})
                chem_all[col][f"{chemical_name}_Cout"] = df_tmp
                time_union[col].extend(df_tmp["time"].dropna().tolist())

            # Influent column
            incol = f"In{col}"
            if incol in df_sheet.columns:
                ser = clean_series(df_sheet[incol], MDL)
                df_tmp = pd.DataFrame({"time": df_sheet["time"], f"{chemical_name}_Cin": ser})
                chem_all[col][f"{chemical_name}_Cin"] = df_tmp
                time_union[col].extend(df_tmp["time"].dropna().tolist())

    
    # --- Build unified chemical DF for each soil column ---
    chem_unified = {}

    for col in colnames:
        # 1. unify and round time
        t_unique = np.sort(pd.Series(time_union[col]).round(3).unique())
        df_col = pd.DataFrame({"time": t_unique})

        # 2. merge all chemical columns on this time axis
        for chem_col, df_tmp in chem_all[col].items():
            df_tmp["time"] = df_tmp["time"].round(3)
            df_col = df_col.merge(df_tmp, on="time", how="left")

        # 3. final deduplication in case any remained
        df_col = df_col.drop_duplicates(subset=["time"]).reset_index(drop=True)

        chem_unified[col] = df_col


    return chem_unified


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
ref_timeseries = pd.read_excel('inputfiles/RPColumns/timeseries_ref.xlsx')
coldata = pd.read_excel('inputfiles/RPColumns/AllMeasurements.xlsx', sheet_name=None)
colnames = ['A','B','C','P','Q','R','X','Y','Z']
chems=['6PPD', '6PPD-Q', 'CBZ', 'BTZ', 'SFX', 'FIP', 'HMMM', 'CAFF']
MDLdict={'6PPD':0, '6PPD-Q':0, 'CBZ':0, 'BTZ':0, 
         'SFX':0, 'FIP':0, 'HMMM':0, 'CAFF':0}
# Unified mapping dictionary
var_map = {
    "Qin":        {"source": "interp", "column": "FlowRate_m3_hr"},
    "Qout_meas":  {"source": "interp", "column": "FlowRate_m3_hr"},

    "Tair":       {"source": "interp", "column": "T_C"},
    "Twater":     {"source": "interp", "column": "T_C"},
    "Tsubsoil":   {"source": "interp", "column": "T_C"},

    "pHwater":    {"source": "interp", "column": "pH"},
    "pHsubsoil":  {"source": "interp", "column": "pH"},

    "Condwater":  {"source": "interp", "column": "EC_uS"},
    "DO_mgL":     {"source": "interp", "column": "DO_mg_L"},
    

    "RainRate":   {"source": "ref", "column": "RainRate"},
    "WindSpeed":  {"source": "ref", "column": "WindSpeed"},
    "RH":         {"source": "ref", "column": "RH"},
    "fvalveopen": {"source": "ref", "column": "fvalveopen"},
}
#Build the parameter dataframes by soil column
soil_raw = build_soil_column_tables(coldata,colnames)
chem_raw = load_chemical_sheets(coldata, chems, colnames,MDLdict,treat_pre_detection_as_zero=True)
#Build the timeseries dataframes
dt = 5/60 #hrs
time_bnds = [soil_raw['A'].time.min(),soil_raw['A'].time.max()]
time_col = 'time'
model_ts = build_model_ready_timeseries(soil_raw,chem_raw,df_ref=ref_timeseries,
            dt=dt,time_bnds=time_bnds,time_col=time_col,var_map = var_map)
#Save model timeseries as an excel worksbook with one sheet per column
#outpth = 'inputfiles/RPCol_InputTimeseries.xlsx'
for soilcol, df in model_ts.items():
    outname = f"inputfiles/RPColumns/InputTimeseries_{soilcol}.csv"
    df.to_csv(outname, index=False)





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
