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
def kinvisc(T):
    #Calculate kinematic viscosity of water
    # Temperature in °C
    temps = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                      55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    # Kinematic viscosity of water in m²/s (from CRC Handbook or IAPWS data)
    kin_viscs = np.array([
        1.787e-6, 1.519e-6, 1.308e-6, 1.139e-6, 1.004e-6,
        0.902e-6, 0.801e-6, 0.729e-6, 0.658e-6, 0.607e-6,
        0.553e-6, 0.514e-6, 0.475e-6, 0.445e-6, 0.415e-6,
        0.392e-6, 0.368e-6, 0.350e-6, 0.331e-6, 0.312e-6,
        0.294e-6])
    nu = np.interp(T, temps, kin_viscs) #Assume constant T
    return nu

#Using the equation of Ferguson and Church (2004)
#following the implementation shown at https://stormwaterbook.safl.umn.edu/other-resources/appendices/importance-particle-size-distribution-performance-sedimentation
def particle_settling(psd,T,R=2.5,C1=18,C2=1.0):
    #psd should be a 2D array where x is particle size (m), y  mass%
    #v = gRd**2/(18*nu+(0.75*C*g*R*d**3)**1/2)
    g = 9.80665#m/s2
    nu = kinvisc(T)
    #Then, calculate particle settling velocity for each particle size
    psd_df = pd.DataFrame(psd,columns=['d','masspct'])
    psd_df.loc[:,'vs']=(g*R*psd_df.d**2)/(C1*nu+(0.75*C2*g*R*psd_df.d**3)**(1/2))
    return psd_df  

#Orifice/weir flow
def culvert_flow_est(
        headwater_depth, #m, series or array, from channel bottom
        tailwater_depth, #m, series or array, from channel bottom
        diameter, #m, culvert diameter (assumes circular)
        L_culvert,#m, culvert length
        head_offset=0., #m, measured from channel bottom 
        tail_offset=0., #m, measured from channel bottom
        n_manning=0.011 #
        ):
    g = 9.80665 #m/s2
    #Initialize res dataframe
    res = pd.DataFrame(np.array([headwater_depth,tailwater_depth]).T,columns=['h_h','h_t'])
    #Adjust heights for offsets of culvert inverts at head and tail
    res.loc[:,'hh_adj'] = res.h_h-head_offset
    res.loc[:,'ht_adj'] = res.h_t-tail_offset
    
    #Calculate flow area and wetted perimeter per https://support.tygron.com/wiki/Culvert_formula_(Water_Overlay)
    res.loc[(res.hh_adj<diameter/2),'h']=res.hh_adj
    res.loc[(res.hh_adj>=diameter/2),'h']=diameter-res.hh_adj
    res.loc[(res.hh_adj>=diameter),'h']=0.0
    res.loc[:,'theta']= 2 * np.arccos((diameter/2-res.h)/(diameter/2))
    res.loc[(res.hh_adj<diameter/2),'area'] = 1/2*(((diameter/2)**2*(res.theta-np.sin(res.theta))))
    res.loc[(res.hh_adj>=diameter/2),'area'] = np.pi*1/4*diameter**2-1/2*(((diameter/2)**2*(res.theta-np.sin(res.theta))))
    res.loc[(res.hh_adj<diameter/2),'Pwet'] = (diameter/2)*res.theta
    res.loc[(res.hh_adj>=diameter/2),'Pwet'] =2*np.pi*(diameter/2)-(diameter/2)*res.theta
    res.loc[:,'R_h'] =res.area/res.Pwet #Hydraulic radius
    #Determine flow regime and calculate discharged. Simplified from https://il.water.usgs.gov/proj/feq/fequtl98.i2h/4_7aupdate.html
    #Calculate culvert loss correction factor
    res.loc[:,'U'] = np.sqrt(1 / (1 + (2 * g * n_manning**2 * L_culvert) / res.R_h**(4/3)))
    res.loc[:,'flow_direction'] = np.sign(res.hh_adj-res.ht_adj) #Flow direction
    res.loc[:,'Q']=0.0 #m3/s. if depth < weir bottom (no flow)
    #They calculate flow with the orifice equation anyway, so no transition. Could
    #adjust U in the future to approach typical orifice values of e.g. 0.6
    res.loc[(res.hh_adj>0.0),'Q']=res.area*res.U*np.sqrt(2 * g * abs(res.hh_adj-res.ht_adj)) * res.flow_direction
    
    # #Estimate weir coefficient using sharp-crested circular weir data
    # #from Staus 1936 via Irzooki 2014 DOI 10.1007/s13369-014-1360-8s
    # #Ratio of head to diameter
    # H0_Ds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
    #                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    # Cds = np.array([0.750, 0.650, 0.623, 0.610, 0.604, 0.600, 0.597, 0.595, 0.594, 0.593,
    #                       0.593, 0.594, 0.595, 0.596, 0.597, 0.599, 0.600, 0.602, 0.604, 0.606])
    # res.loc[:,'H_D']=res.hh_adj/diameter #head over diameter
    # res.loc[:,'Cd'] = np.interp(res.H_D, H0_Ds, Cds)
    # #Free weir flow - outlet is free-fall, inlet is not submerged
    # res.loc[(res.hh_adj<diameter) & (res.ht_adj<=0),'Q'] = res.Cd*diameter*res.hh_adj**1.5
    # #Submerged weir flow
    # res.loc[(res.hh_adj<diameter) & (res.ht_adj>0),'Q'] = res.Cd*diameter*res.hh_adj**1.5
    # #Free Orifice flow
    # res.loc[(res.hh_adj>=diameter) & (res.ht_adj<=diameter),'Q'] = Co*area*np.sqrt(2*g*res.hh_adj)
    # #Submerged Orifice flow
    # res.loc[(res.hh_adj>=diameter) & (res.ht_adj>diameter),'Q'] = Co*area*np.sqrt(2*g*(res.hh_adj-res.ht_adj))
    return res.Q