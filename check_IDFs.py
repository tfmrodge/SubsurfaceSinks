# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:09:26 2023

@author: trodge01
"""


import os
import seaborn as sns
#This is the name of the python module containing the Bioretention Blues submodel.
import itertools

#Testing the model results that have been output and what is missing

combos = list(itertools.product([0,1],repeat=8))
#combos = ((0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0),(0,1,0,0,0,0,0,0),(0,0,1,0,0,0,0,0),(0,0,0,1,0,0,0,0),(0,0,0,0,1,0,0,0),
#          (0,0,0,0,0,1,0,0),(0,0,0,0,0,0,1,0),(0,0,0,0,0,0,0,1))
#          #(1,1,0,0,1,1),(1,1,0,0,0,1))
#combos = ((0,0,0,0,0,0,0,0),)
#pdb.set_trace()
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
    elif (scenario_dict['fvalve'] == True) & (scenario_dict['hpipe'] ==True):
        continue #Mutually exclusive
    else:
        print(outname,combo)

