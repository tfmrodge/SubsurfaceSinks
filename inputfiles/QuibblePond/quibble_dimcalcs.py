# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:07:59 2025

@author: trodge01
"""

#Quibble dimensions calculations. This series of functions will produce a table
#that can be interpolated by the StormPond make_system function

import pandas as pd

import pdb
import numpy as np

def qbl_minWD(W_meas,designdepth,minW):
    #Calculate the width at 0 depth
    W0 = W_meas-6*(designdepth)#6 = 4:1 outer bank + 2:1 inner bank
    #Assume that once it reaches min width channel is straight-cut.
    W0[W0<minW] = minW
    D_minw = designdepth - (W_meas-minW)/6
    D_minw[W0!=minW] = 0.
    return D_minw

def qbl_W(W_meas,D,designdepth,bermtop,overflow,minW):
    #Calculate the top width of the channel based on the depth D.
    #W_meas is the measured channel width at the regular pond depth
    #W is calculate based on design channel slopes
    #Outside berm = 4:1, inside = 2:1.
    #Inside berm only goes up to 'bermtop', which is 0.5m wide, so allocated 
    #0.25 to each side. Outside berm goes to overflow.
    #Need to adjust for minimum depth - No slope only 1m across straight there
    D_minw = qbl_minWD(W_meas,designdepth,minW)
    Dapp = (D - D_minw) #Dapp is depth that the 
    Dapp[Dapp<0] = 0
    #Next, calculate bottom width
    W0 = W_meas-6*(designdepth)
    #Set minimum channel width
    W0[W0<minW] = minW
    #Total width can then be calculated from W0 and channel slope.
    if D <= bermtop:
        W = W0+6*(Dapp)
    elif D <= overflow:
        W = W0+2*(bermtop-D_minw)+4*Dapp+0.25 #Inside slope only goes to bermtop
    else:
        return 'Overflowing'
    return W

def qbl_TA(W_meas,L,D,designdepth,bermtop,overflow,minW):
    #Calculate the Top Area (TA). L is the length of each segment (dX)
    TA = qbl_W(W_meas,D,designdepth,bermtop,overflow,minW)*L
    return TA

def qbl_CA(W_meas,D,designdepth,bermtop,overflow,minW):
    #Calculate the wetted Channel Area (CA) per unit length
    #Channel area is the bottom area + the wetted side areas.
    #First calculate bottom area = W[0]*L+2*minwD*L
    D_minw = qbl_minWD(W_meas,designdepth,minW)
    BA = (2*D_minw+qbl_W(W_meas,0.0,designdepth,bermtop,overflow,minW))
    #For subsequent calculations
    Dapp = (D - D_minw)
    Dapp[Dapp<0] = 0
    #Next, hypotenuse of outside bank=sqrt(17)*D. Inside = sqrt(5)*D
    if D <= bermtop:
        CA = BA + (np.sqrt(17)+np.sqrt(5))*Dapp
    elif D <= overflow:
        CA = BA + (0.25+(bermtop-D_minw)*np.sqrt(5)+(np.sqrt(17))*Dapp)
    else:
        return 'Over 100-yr Event Line'
    return CA

def qbl_XA(W_meas,L,D,designdepth,bermtop,overflow,minW):
    #Calculate the Cross-Section Area (XA). L is the length of each segment (dX)
    #Split into bottom/middle rectangle plus outside and inside triangles
    #Triangle is 0.5bh
    #First calculate bottom/middle rectangle
    MRA = qbl_W(W_meas,0.0,designdepth,bermtop,overflow,minW)*D
    #Rebase depth  to apparent depth for other calcs 
    D_minw = qbl_minWD(W_meas,designdepth,minW)
    Dapp = D - D_minw
    Dapp[Dapp<0] = 0
    #Inside Bank - right bank triangle width + berm top
    #IB Width = width at depth - outside bank width - bottom width
    IBW = qbl_W(W_meas,D,designdepth,bermtop,overflow,minW)-4*Dapp\
        - qbl_W(W_meas,0.0,designdepth,bermtop,overflow,minW)
    IBA = Dapp**2
    if D > bermtop:
        #triangle at bank to bermtop + rectangle above
        IBA = (bermtop-D_minw)**2 #0.5*(2D)*D
        #IBA += (IBW - qbl_W(W_meas,bermtop,designdepth,bermtop,overflow,minW))*D
        IBA += IBW*(D-bermtop)
    #Outside bank area - always the same
    #0.5*4Dapp*Dapp = 2*Dapp**2
    OBA = 2*(Dapp**2)
    XA = MRA + IBA + OBA
    return XA

#Testing
# qbdims.loc[:,'W_bottom'] = qbl_W(qbdims.W,0.0,designdepth,bermtop,overflow,minW)
# qbdims.loc[:,'W_design'] = qbl_W(qbdims.W,designdepth,designdepth,bermtop,overflow,minW)
# qbdims.loc[:,'W_berm'] = qbl_W(qbdims.W,bermtop,designdepth,bermtop,overflow,minW)
# qbdims.loc[:,'TA_berm'] = qbl_TA(qbdims.W,qbdims.dX,bermtop,designdepth,bermtop,overflow,minW)
# qbdims.loc[:,'CA_berm'] = qbl_CA(qbdims.W,qbdims.dX,bermtop,designdepth,bermtop,overflow,minW)
# qbdims.loc[:,'XA_berm'] = qbl_XA(qbdims.W,qbdims.dX,bermtop,designdepth,bermtop,overflow,minW)
#Now lets generate a series of interpolation tables for further processing
#pdb.set_trace()
#dX = 0.5 
def quibble_dimcalc_tables(qbdf,dimdict=None):
    #Define breakpoints necessary for the system. These are the 
    #measured cross-sections. Control volume calcs will assume linear between them
    #Four depths necessary for this system:
    #bottom = 0.0. berm - 0.9, berm+epsilon = 0.9001, overflow =2.0
    if dimdict is None:
        dimdict = {'bottom':0.0,'berm':0.9,'berm_eps':0.9001,'overflow':2.0,'minW':0.61,
               'designdepth':0.5}
    for dim in list(dimdict)[:4]:#Manual, need to change if more depths wanted
        qbdf.loc[:,'W_'+dim] = qbl_W(qbdf.W,dimdict[dim],dimdict['designdepth'],
                                     dimdict['berm'],dimdict['overflow'],dimdict['minW'])
        #qbdf.loc[:,'TA_'+dim] = qbl_TA(qbdf.W,qbdf.dX,dimdict[dim],designdepth,bermtop,overflow,minW)
        qbdf.loc[:,'CA_'+dim] = qbl_CA(qbdf.W,dimdict[dim],dimdict['designdepth'],
                                       dimdict['berm'],dimdict['overflow'],dimdict['minW'])
        qbdf.loc[:,'XA_'+dim] = qbl_XA(qbdf.W,qbdf.dX,dimdict[dim],dimdict['designdepth'],
                                       dimdict['berm'],dimdict['overflow'],dimdict['minW'])
    #Calculate D_minw and XA_Dminw
    qbdf.loc[:,'D_minw'] = qbl_minWD(qbdf.W,dimdict['designdepth'],dimdict['minW'])
    qbdf.loc[:,'XA_Dminw'] = qbdf.loc[:,'W_bottom']*qbdf.loc[:,'D_minw']
    return qbdf
    #qbdims.loc[:,'W_bottom':].sort_ascending() 
    # #Calculate widths
    # qbdims.loc[:,'W_bottom'] = qbl_W(qbdims.W,0.0,designdepth,bermtop,overflow,minW)
    # qbdims.loc[:,'W_berm'] = qbl_W(qbdims.W,bermtop,designdepth,bermtop,overflow,minW)
    # qbdims.loc[:,'W_berm_eps'] = qbl_W(qbdims.W,bermtop+0.001,designdepth,bermtop,overflow,minW) #
    # qbdims.loc[:,'W_overflow'] = qbl_W(qbdims.W,overflow,designdepth,bermtop,overflow,minW) #
    # #Calculate Water Surface Area
    # qbdims.loc[:,'W_bottom'] = qbl_W(qbdims.W,0.0,designdepth,bermtop,overflow,minW)
    # qbdims.loc[:,'W_berm'] = qbl_W(qbdims.W,bermtop,designdepth,bermtop,overflow,minW)
    # qbdims.loc[:,'W_berm_eps'] = qbl_W(qbdims.W,bermtop+0.001,designdepth,bermtop,overflow,minW) #
    # qbdims.loc[:,'W_overflow'] = qbl_W(qbdims.W,overflow,designdepth,bermtop,overflow,minW) #
#qbdf = quibble_dimcalc_tables(qbdims)



