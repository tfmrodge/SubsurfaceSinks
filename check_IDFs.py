# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:12:34 2022
This script will run the model across intensity-duration-frequency curves (as specified in a timeseries file)
or across a water year (timeseries_wateryear)
It can be used to change the design of the system based on a combination of design features (as explained below)

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
sum=0
for combo in combos:
    scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                        'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}
    for ind, param in enumerate(scenario_dict):        
        scenario_dict[param] = bool(combo[ind])
        
    outpath = '/home/tfmrodge/scratch/bcdesign/pickles/IDFouts/'
    filtered = [k for k,v in scenario_dict.items() if v == True]
    #outname = 'IDF_EngDesign'+'_'.join(filtered)+'.pkl'
    outname = 'IDF_EngDesign_lowKn'+'_'.join(filtered)+'.pkl'
    #outname = 'IDF_'+'_'.join(filtered)+'.pkl'
    #outname = 'IDF_lowKn'+'_'.join(filtered)+'.pkl'
    if outname in os.listdir(outpath):
        sum+=1 #Skip if already done
    elif (scenario_dict['fvalve'] == True) & (scenario_dict['hpipe'] ==True):
        continue #Mutually exclusive
    else:
        print(outname,combo)
print('Done '+str(sum) +' combinations')

