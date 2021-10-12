# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:36:27 2018

@author: Tim Rodgers
"""
import numpy as np
import pandas as pd
import math
import pdb

def ppLFER(L,S,A,B,V,l,s,a,b,v,c):
    """polyparameter linear free energy relationship (ppLFER) in the 1 equation form from Goss (2005)
    Upper case letters represent Abraham's solute descriptors (compund specific)
    while the lower case letters represent the system parameters.
    """
    res = L*l+S*s+A*a+B*b+V*v+c
    return res

def vant_conv(dU,T2,k1,T1 = 298.15,):
    """Van't Hoff equation conversion of partition coefficients (Kij) from T1 to T2 (K)
    The default value for T1 is 298.15K. Activation energy should be in J. The 
    result (res) will be K2 at T2
    """
    R = 8.314 #J/mol/K
    res =  k1 * np.exp((dU / R) * (1 / T1 - 1 / T2))
    return res
    
def arr_conv(Ea,T2,k1,T1 = 298.15,):
    """Arrhenius equation conversion of rate reaction constants (k) from T1 to T2 (K)
    The default value for T1 is 298.15K. Activation energy should be in J. The 
    result (res) will be k2 at T2. This will work on vectors as well as scalars
    Units of k can be anything, as it is multiplied by a unitless quantity
    """
    R = 8.314 #J/mol/K
    res =  k1 * np.exp((Ea / R) * (1 / T1 - 1 / T2))
    return res

def make_ppLFER(pp):
    """Check if ppLFER system parameters are in the pp file, if not generate
    them. The order is l,s,a,b,v, c.
    """
    #Aerosol-air ppLFER system parameters Arp (2008) 
    if 'logKqa' not in pp.columns:
        pp['logKqa'] = [0.63,1.38,3.21,0.42,0.98,-7.24]
    #Organic Carbon - Water ppLFER system parameters Bronner & Goss (2011) (L/kg)
    if 'logKocW' not in pp.columns:
        pp['logKocW'] = [0.54,-0.98,-0.42,-3.34,1.2,0.02]
    #Storage lipid - water Geisler, Endo, & Goss 2012 [L(water)/L(lipid)]
    if 'logKslW' not in pp.columns:
        pp['logKslW'] = [0.58,-1.62,-1.93,-4.15,1.99,0.55]
    #Air - water Goss (2005)
    if 'logKaw' not in pp.columns:
        pp['logKaw'] = [-0.48,-2.07,-3.367,-4.87,2.55,0.59]
    #Ionic Kow
    #if 'logKowi' no in pp.columns:
        
    #dU Storage lipid - Water Geisler et al. 2012 (kJ/mol)
    if 'dUslW' not in pp.columns:
        pp['dUslW'] = [10.51,-49.29,-16.36,70.39,-66.19,38.95]
    #dU Octanol water Ulrich et al. (2017) (J/mol)
    if 'dUow' not in pp.columns:
        pp['dUow'] = [8.26,-5.31,20.1,-34.27,-18.88,-1.75]
    #dU Octanol air Mintz et al. (2008) (kJ/mol)
    if 'dUoa' not in pp.columns:
        pp['dUoa'] = [53.66,-6.04,53.66,9.19,-1.57,6.67]
    #dU Water-Air Mintz et al. (2008) (kJ/mol)
    if 'dUaw' not in pp.columns:
        pp['dUaw'] = [-8.26,0.73,-33.56,-43.46,-17.31,-8.41]
        
    return pp

#When slicing, reduce an index to match the display rather than keeping the
#entire index
def df_sliced_index(df):
    new_index = []
    rows = []
    for ind, row in df.iterrows():
        new_index.append(ind)
        rows.append(row)
    return pd.DataFrame(data=rows, index=pd.MultiIndex.from_tuples(new_index))

#Find nearest value, used to find the index of a specific time
#Source: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array,value):
    #pdb.set_trace()
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]