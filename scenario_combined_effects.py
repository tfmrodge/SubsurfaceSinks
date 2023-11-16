# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:28:16 2023

@author: trodge01
"""
import pandas as pd
import numpy as np
import pdb
#Calculate first and second order intervention effects for a given scenario. 
#This one may end up being a little tricky - may want to calculate multiple ways, too.
#Goal is to find the contribution to the removal of a scenario that is given by each scenario.
#First way, following the effects above, average with intervention = 0 minus intervention = 1
fpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Manuscript/Figs/Pythonfigs/'    
sdf = pd.read_excel(fpath+'KocScenarioData_upload1.xlsx')
scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                 'Asys':False, 'Dpond':False, 'hpipe':False, 'amend':False}    
chems = ['PFOA','Benzotriazole','6PPDQ','TCEP','Phe','6PPD','BaP']
fastdf = sdf.loc[sdf.Low_kn==0,:]
slowdf = sdf.loc[sdf.Low_kn==1,:]
#OK, now we want to give the combos, then we can calculate for each combo.
#Highway  (low Kn)
#Residential (TCEP, Phe, B[a]P) - [1,1,1,1,1,1,0,1], [0,1,1,1,1,1,1,1] (low Kn)
#Airport (PFOA, benzotriazole) -  [0,1,0,1,1,1,1,1], [0,1,0,1,1,1,1,1] (low Kn)
vignette='Residential'
vdict = {'Highway':[[1,1,1,1,1,1,0,0],[0,1,1,1,1,1,1,0],['6PPDQ', '6PPD']],
        'Residential':[[1,1,1,1,1,1,0,1],[0,1,1,1,1,1,1,1],['TCEP', 'Phenanthrene','Benzo[a]pyrene']],
        'Airport':[[0,1,0,1,1,1,1,1],[0,1,0,1,1,1,1,1],['PFOA','Benzotriazole']]}
combofast = vdict[vignette][0]
adcols = ['6PPDQ_advected','PFOA_advected','Phenanthrene_advected','Benzo[a]pyrene_advected','Benzotriazole_advected',
                              'TCEP_advected','6PPD_advected']
stormcols = ['6PPDQ_storm','PFOA_storm','Phenanthrene_storm','Benzo[a]pyrene_storm','Benzotriazole_storm',
                              'TCEP_storm','6PPD_storm']
Edf = pd.DataFrame(index=['base','scenario','fvalve', 'Foc', 'Kinf', 'Dsys', 
                         'Asys', 'Dpond', 'hpipe', 'amend','fvalve_Foc','fvalve_Kinf',
                         'fvalve_Dsys','fvalve_Asys','fvalve_Dpond','fvalve_amend','Foc_Kinf',
                         'Foc_Dsys','Foc_Asys','Foc_Dpond','Foc_hpipe','Foc_amend','Kinf_Dsys',
                         'Kinf_Asys','Kinf_Dpond','Kinf_hpipe','Kinf_amend','Dsys_Asys','Dsys_Dpond',
                         'Dsys_hpipe','Dsys_amend','Asys_Dpond','Asys_hpipe','Asys_amend','Dpond_hpipe',
                         'Dpond_amend','hpipe_amend'],columns = adcols+stormcols)

#Set the scenario_dict to the scenario
pdb.set_trace()
for ind, param in enumerate(scenario_dict):
    #pdb.set_trace()
    scenario_dict[param] = bool(combofast[ind])
mask = (fastdf.fvalve == scenario_dict['fvalve']) & (fastdf.Foc == scenario_dict['Foc']) \
 & (fastdf.Kinf == scenario_dict['Kinf'])& (fastdf.Dsys == scenario_dict['Dsys'])& (fastdf.Asys == scenario_dict['Asys']) \
 & (fastdf.Dpond == scenario_dict['Dpond'])& (fastdf.hpipe == scenario_dict['hpipe'])& (fastdf.amend == scenario_dict['amend'])
maskb = (fastdf.fvalve == False) & (fastdf.Foc == False) \
 & (fastdf.Kinf == False)& (fastdf.Dsys == False)& (fastdf.Asys == False) \
 & (fastdf.Dpond == False)& (fastdf.hpipe == False)& (fastdf.amend == False)
Edf.loc['base',adcols] = np.array(fastdf[maskb].loc[:,['pct_advected']]).T
Edf.loc['scenario',adcols] = np.array(fastdf[mask].loc[:,['pct_advected']]).T
Edf.loc['base',stormcols] = np.array(fastdf[maskb].loc[:,['pct_stormsewer']]).T
Edf.loc['scenario',stormcols] = np.array(fastdf[mask].loc[:,['pct_stormsewer']]).T
scenarios = [scenario for scenario in scenario_dict.keys() if (scenario_dict[scenario] == True)]
#pdb.set_trace()
for scenario in scenarios:
    #Set the scenario to false
    scenario_dict[scenario] = False
    mask0 = (fastdf.fvalve == scenario_dict['fvalve']) & (fastdf.Foc == scenario_dict['Foc']) \
     & (fastdf.Kinf == scenario_dict['Kinf'])& (fastdf.Dsys == scenario_dict['Dsys'])& (fastdf.Asys == scenario_dict['Asys']) \
     & (fastdf.Dpond == scenario_dict['Dpond'])& (fastdf.hpipe == scenario_dict['hpipe'])& (fastdf.amend == scenario_dict['amend'])
    indname = scenario
    Edf.loc[indname,adcols] = np.array(fastdf[mask0].loc[:,['pct_advected']]).T - np.array(Edf.loc['scenario',adcols])
    Edf.loc[indname,stormcols] = np.array(fastdf[mask0].loc[:,['pct_stormsewer']]).T - np.array(Edf.loc['scenario',stormcols])
    scenario_dict[scenario] = True
    #fastdf[mask0].loc[:,['Compound','pct_advected']]
#Loop again for lazy codewriting
for scenario in scenarios:
    scenario_dict2 = scenario_dict.copy()
    scenario_dict[scenario] = False
    mask0 = (fastdf.fvalve == scenario_dict['fvalve']) & (fastdf.Foc == scenario_dict['Foc']) \
     & (fastdf.Kinf == scenario_dict['Kinf'])& (fastdf.Dsys == scenario_dict['Dsys'])& (fastdf.Asys == scenario_dict['Asys']) \
     & (fastdf.Dpond == scenario_dict['Dpond'])& (fastdf.hpipe == scenario_dict['hpipe'])& (fastdf.amend == scenario_dict['amend'])
    for scenario2 in scenarios:
        if scenario == scenario2:
            pass
        else:
            indname = scenario +'_'+ scenario2
            if indname not in Edf.index:
                pass
            else:
                scenario_dict[scenario2] = False
                indname = scenario +'_'+ scenario2
                mask00 = (fastdf.fvalve == scenario_dict['fvalve']) & (fastdf.Foc == scenario_dict['Foc']) \
                 & (fastdf.Kinf == scenario_dict['Kinf'])& (fastdf.Dsys == scenario_dict['Dsys'])& (fastdf.Asys == scenario_dict['Asys']) \
                 & (fastdf.Dpond == scenario_dict['Dpond'])& (fastdf.hpipe == scenario_dict['hpipe'])& (fastdf.amend == scenario_dict['amend'])
                mask01 = mask0 #Same as scenario_dict
                #We will use scenario dict 2 for this one
                scenario_dict2[scenario2] = False 
                mask10 = (fastdf.fvalve == scenario_dict2['fvalve']) & (fastdf.Foc == scenario_dict2['Foc']) \
                 & (fastdf.Kinf == scenario_dict2['Kinf'])& (fastdf.Dsys == scenario_dict2['Dsys'])& (fastdf.Asys == scenario_dict2['Asys']) \
                 & (fastdf.Dpond == scenario_dict2['Dpond'])& (fastdf.hpipe == scenario_dict2['hpipe'])& (fastdf.amend == scenario_dict2['amend'])
                #mask11 is the full scenario - already have in Edf
                Edf.loc[indname,adcols] = -0.5*(np.array(fastdf[mask00].loc[:,['pct_advected']]).T
                                           - np.array(fastdf[mask10].loc[:,['pct_advected']]).T
                                           - np.array(fastdf[mask01].loc[:,['pct_advected']]).T
                                           + np.array(Edf.loc['scenario',adcols])) #Y11, just put at end for convenience#pdb.set_trace()
                #TR20231115 - redid to match secondary effect calculations from before.
                #Edf.loc[indname,adcols] = np.array(fastdf[mask00].loc[:,['pct_advected']]).T - np.array(Edf.loc['scenario',adcols]) -  np.array(Edf.loc[scenario,adcols]) -  np.array(Edf.loc[scenario2,adcols])
                #Edf.loc[indname,stormcols] = np.array(fastdf[mask00].loc[:,['pct_stormsewer']]).T - np.array(Edf.loc['scenario',stormcols]) - np.array(Edf.loc[scenario,stormcols]) -np.array(Edf.loc[scenario2,stormcols])
                scenario_dict[scenario2] = True
                scenario_dict2[scenario2] = True
    #Set the scenario back to True
    scenario_dict[scenario] = True   
Edf