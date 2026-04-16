# -*- coding: utf-8 -*-
"""
Created on 3/25/2026.
Updated from previous script with the help of M365 Copilot

@author: Tim Rodgers
"""
#This is the name of the python module containing the Bioretention Blues submodel.
from BioretentionBlues import BCBlues, calibrate_flow_system, calibrate_tracer_system
import pdb
from datetime import datetime
import os
import sys
import argparse

from joblib import Parallel, delayed 

#pdb.set_trace()

CONFIG = {
    "colnames": ['A','B','C','P','Q','R','X','Y','Z'],

    "paths": {
        "timeseries_dir": 'inputfiles/RPColumns/30min',
        "locsumm_pth":  'inputfiles/RPColumns/RPColumn_BC.xlsx',
        "chemsumm_pth": 'inputfiles/RPColumns/TrOC_column_CHEMSUMM.xlsx',
        "params_pth":   'inputfiles/RPColumns/params_columns.xlsx',
        "pickle_dir":    "/home/tfmrodge/scratch/RPColumns/Pickles/"
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
            #None
            "tstart": 1300,   # 1000
            "tend": 1800     # 1575
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
#pdb.set_trace()

#Universal config
today = datetime.today().strftime("%Y%m%d")
n_workers = 48 #-1 #75#-1 #-1 #os.cpu_count() -5
man_idx = 1
#Flow cal config
cal_flows =False
TARGET_HYDRO='bias'
#Tracer cal config
cal_tracer = True
TARGET_TRACER='kge'

#Parse arguments
parser = argparse.ArgumentParser(
    description="Calibrate BCBlues flow model for a single RP column."
)

parser.add_argument(
    "system_index",
    type=int,
    nargs="?",          # <-- makes it optional
    default=None,
    help="Column index (1=A, 2=B, 3=C, ...)"
)
args = parser.parse_args()

# 1-based index → 0-based Python index
if args.system_index is None:
    idx = man_idx - 1
    print(f"No argument given; running for index {man_idx}")
else:
    idx = args.system_index - 1


if idx < 0 or idx >= len(CONFIG["colnames"]):
    raise ValueError(
        f"System index {args.system_index} out of range. "
        f"Must be between 1 and {len(CONFIG['colnames'])}."
    )

sysname = CONFIG["colnames"][idx]

print(f"Running flow calibration for system {sysname} (index {args.system_index})")
flow_time = None
if cal_flows:
    out = calibrate_flow_system(
        CONFIG,
        system_name=sysname,
        paramnames=["Kf","Ks"],
        target=TARGET_HYDRO,
        param0s=[0.2,0.2],
        bounds=[(1e-5, 1),(1e-5, 1)],
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
    forward_results = out["forward_results"]
    flow_time = forward_results['flow_time']
    outpth = f"{CONFIG['paths']['pickle_dir']}{today}_RPcalout_{sysname}.csv"
    flow_time.to_csv(outpth)
    outfig = f"{CONFIG['paths']['pickle_dir']}{today}_RPcalout_{sysname}.jpg"
    flow_fig = forward_results['flow_fig']
    flow_fig.savefig(outfig)

if cal_tracer:
    tracer_out = calibrate_tracer_system(
        CONFIG,
        system_name=sysname,

        # ---- tracer parameters to calibrate ----
        paramnames= ["alpha", "Kf","thetam"],   #["alpha", "Kf"],   #
        param0s=[1, 0.2,0.2], #[1, 0.2], #
        bounds= [(1e-4, 50), (1e-4, 10), (1e-4, 9.9999e-1)],#[(1e-4, 50), (1e-4, 10)],#

        # ---- objective ----
        target=TARGET_TRACER,
        objective=None, #Use to set a recovery value e.g. 0.7 proportion recovered

        #Reuse flows if needed.
        flows=flow_time,

        solver="differential_evolution",
        solver_method="L-BFGS-B", #
        solver_kwargs={
            "workers": n_workers,
            "updating": "deferred",
            "maxiter": 50,
            "disp": True,
        },

        prefix="tracer",
        suffix=f"{sysname}_{today}",
        output_dir=CONFIG["paths"]["pickle_dir"],

        save_forward_run=True,
        plot_forward_run=True,
    )

    print(
        f"Tracer calibration complete for system {sysname} "
        f"(objective={tracer_out['result'].fun:.3g})"
    )
    
