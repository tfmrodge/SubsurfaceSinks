# -*- coding: utf-8 -*-
"""
Created on 3/25/2026.
Updated from previous script with the help of M365 Copilot

@author: Tim Rodgers
"""
#This is the name of the python module containing the Bioretention Blues submodel.
from BioretentionBlues import BCBlues, calibrate_flow_system
import pdb
from datetime import datetime
import os

from joblib import Parallel, delayed 

#Testing the model
#Load parameterization files
pdb.set_trace()

CONFIG = {
    "colnames": ['A','B','C','P','Q','R','X','Y','Z'],

    "paths": {
        "timeseries_dir": 'inputfiles/RPColumns/',
        "locsumm_xlsx":  'inputfiles/RPColumns/RPColumn_BC.xlsx',
        "chemsumm_xlsx": 'inputfiles/RPColumns/TrOC_column_CHEMSUMM.xlsx',
        "params_xlsx":   'inputfiles/RPColumns/params_columns.xlsx',
        "pickle_dir":    "D:/OneDrive - UBC/Postdoc/Completed Projects/6PPD_BC Papers/Modeling/Pickles/"
    },

    "flags": {
        "modify_timestep": True,
        "calcflow": True, #Run the hydrology calcs
        "calcinp": False, #Run the input calcs
        "runall": False, #Run the model without pre-calculating intermediates 
        "runmodel": False, #run the full model
        "save_intermediate": True,
        "plot_flows": True,
        "plot_mass_balance": False,
        "plot_conc": False,
        "pulse": True, #False, # Set True for tracer test/spike type system

        # output control
        "timestamp_outputs": True
    },

    "timestep": {
        "indfactor": 1
    },
    
    "timeslice": {
            # units assumed to be same as timeseries.time (e.g. hours)
            # Use None to disable slicing on either end
            "tstart": 1487.5,   # 1000
            "tend": 1568.5      # 1575
        },


    "model": {
        "numc": ['water', 'subsoil', 'air', 'pond'],
        "run_chems": ['Bromide'], #['6PPD-Q'], #
        # now a LIST
        "compound_mass_plot": ['Bromide','6PPD', '6PPD-Q', 'CBZ', 'BTZ', 'SFX', 'FIP', 'HMMM', 'CAFF'],
        "mass_plot_time": 8760,
        "multfactor_conc_plot": 1e6
    },

    # optional label added to all outputs
    "run_label": "testing",
    #hydroplot flags
    "hydrology_plot": {
            "enable": True,
    
            # One of:
            # "all" → plot entire time series
            # [(t0, t1), (t2, t3), ...] → plot specific windows
            "windows": [
                "all"  # None means "to end"
            ],
    
            "compartments": ["drain", "water"],
            "yvar": "Q_todrain"
        },

}
cal_flows = True
today = 20260330 #datetime.today().strftime("%Y%m%d")
n_workers = 1 #os.cpu_count() -5
sysname = CONFIG['colnames'][0]
if cal_flows:
    out = calibrate_flow_system(
        CONFIG,
        system_name=sysname,
        paramnames=["Ks"],
        param0s=[0.2],
        bounds=[(1e-5, 1)],
        solver="differential_evolution",
        solver_kwargs={
            "workers": n_workers,
            "updating": "deferred",
            "maxiter": 40,
            "popsize": 12,
            "polish": True,
            "disp": True,
        },
        suffix=f"testing_{today}",
    )

