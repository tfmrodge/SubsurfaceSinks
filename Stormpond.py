# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 13:33:09 2025

@author: trodge01
"""

from FugModel import FugModel #Import the parent FugModel class
from Subsurface_Sinks import SubsurfaceSinks
from HelperFuncs import df_sliced_index #Import helper functions
from scipy import optimize
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('ticks')
import matplotlib.pyplot as plt
import pdb #Turn on for error checking
from warnings import simplefilter
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
import hydroeval
from scipy.optimize import minimize
import psutil
import ast
from scipy.interpolate import LinearNDInterpolator
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category= RuntimeWarning)
simplefilter(action="ignore", category= FutureWarning)

class StormPond(SubsurfaceSinks):
    """Stormwater management pond implementation of the Subsurface_Sinks model.
    This model represents a horizontally flowing stormwater ponmd.
    It is intended to be solved as a Level V 1D ADRE, across space and time, 
    although it can be modified to make a Level III or Level IV multimedia model
        
    Attributes:
    ----------
            
            locsumm (df): physical properties of the systmem
            chemsumm (df): physical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            timeseries (df): Timeseries values providing the time-dependent inputs
            to the model.
            num_compartments (int): (optional) number of non-equilibirum 
            compartments
            name (str): (optional) name of the stormpond model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,locsumm,chemsumm,params,timeseries,num_compartments = 9,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc) 
    
    def make_system(self,locsumm,params,numc,timeseries,ponddims,dx = None,flow_time=None):
        """ 2025-03-25
        This function will build the dimensions of the 1D system based on the "locsumm" input file.
        and the 'ponddims' and 'dimmdict' input files which gives dimensions as
        a function of dX and depth, with the depth breakpoints specified in dimdict
        If you want to specify more things you can can just skip this and input a dataframe directly
        
        Since the hydrology and the contaminant transport are calculated independently,
        we can call them independently to save time & memory
        Either input a raw summary of the BC dimensions, and we will run the hydrology,
        or input the timeseries of BC flows etc. as a locsumm with the flows
        If the input timeseries has the same index as locsumm, it assumes that
        the hydrology has been pre-calculated
        Attributes:
        ----------
                
                locsumm (df): physical properties of the systmem
                params (df): Other parameters of the model
                numc (list): list of compartment names (str)
                timeseries (df): Timeseries values providing the time-dependent inputs
                to the model.
                ponddims (df): Dataframe with pond dimensions along X
                name (str): (optional) name of the stormpond model 
                pplfer_system (df): (optional) input ppLFERs to use in the model
        """
        #pdb.set_trace()
        #Check if flow_time exists. flow_time defines the flow routing for this
        #system, so defines compartment dimensions which vary in time
        if flow_time is not None:
            res = flow_time
        else:
            res = self.flow_time(locsumm,params,['water'],timeseries,ponddims,dx)
            
        
        return res
    
    def flow_time(self,locsumm,params,numc,timeseries,ponddims,dx=None):
        """2025-03-25
        This code will determine the dimensions of the reach/pond through time
        Flow routing can be specified as simple (assume all one bucket) or
        other, not yet defined
        
        Attributes:
        -----------        
        locsumm (df) - gives the dimensions of the system at the initial conditions
        params (df) - parameters of the system
        numc (str) - list of compartments
        timeseries (df) - needs to contain inflow in m³/h, rainrate in mm/h
        ponddims (df): Dataframe with pond dimensions along X
        """
        #dt = timeseries.time[1]-timeseries.time[0] #Set this so it works
        #Define the length dimension
        L = ponddims.X.max()
        params.loc['L','val'] = L
        #dx = 0.1
        if dx == None:
            dx = params.val.dx
        #Set up control volumes. X is defined in the centre of each cell.
        #Xs = np.arange(0.0+dx/2.0,L,dx)
        #Xs = np.append(np.arange(0.0+dx/2.0,L,dx),ponddims.X.max())
        res = pd.DataFrame(np.append(np.arange(0.0+dx/2.0,L,dx),ponddims.X.max()),columns = ['x'])
        res.loc[:,'dx'] = res.x - res.x.shift(1)
        res.loc[0,'dx']=dx
        # #Now we will add a final cell for the end of system pipe
        # res.loc[len(res.index),'x'] = L + params.val.Lpipe/2
        # res.loc[:,'dx'] = res.x - res.x.shift(1)
        # res.loc[len(res.index)-1,'dx'] = params.val.Lpipe
        # res.loc[0,'dx']=dx
        # #Add a final x for those that aren't discretized
        # if len(numc) > 2: #Don't worry about this for the 2-compartment version
        #     res.loc[len(res.index),'x'] = 999#May need to change this for longer systems!
        #
        #
        #res = ponddims.copy(deep=True)
        ntimes = len(timeseries['time'])
        #Set up output dataframe by adding time 
        #Index level 0 = time, level 1 = chems, level 2 = cell number
        res_t = dict.fromkeys(timeseries.index,[]) 
        #pdb.set_trace()
        #Set up our interpolation functions
        #Dimdict contains the breakpoints depths for interpolation
        dimdict = ast.literal_eval(params.val.dimdict)
        #Find the keys that matter
        matching_keys = [key for key in dimdict if any(key in s for s in ponddims.columns)]
        #Set up the interpolation as a function of x and depth
        #First set up grid coordinates (x,depth) for interpolation
        depths = [dimdict[key] for key in matching_keys]
        X,Y = np.meshgrid(ponddims.X,depths)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        #Add D_minw
        newpts = np.column_stack([ponddims.X, ponddims.D_minw])
        pts = np.vstack([pts, newpts])
        #Need to interpolate width (W), channal area (CA), cross-section area (XA)
        interps = {} #Make a dict with the interpolation functions for each dim
        for i in ['W','CA','XA']:
            cols = [i+'_'+s for s in matching_keys]
            #add D_minw = same value as bottom except XA
            if i == 'XA':
               cols.append('XA_Dminw')
            else:
                cols.append(i+'_bottom')
            Z = np.array(ponddims.loc[:,cols]).ravel(order='F')
            interps[i] = LinearNDInterpolator(pts, Z)
        #X will always be at the points of res.x
        for t in range(ntimes):
            if t == 0:
                dt = timeseries.time[1]-timeseries.time[0]
                #Initialize the system. Assume uniform depth
                #Interpolate top area, channel area, and cross-sectional area
                #Dapp = locsumm.Depth.water - ponddims.loc[:,'D_minW']#Apparent depth due 
                res.loc[:,'Depth'] = locsumm.Depth.water
                res.loc[:,'toparea'] = interps['W'](res.x,locsumm.Depth.water)*res.dx
                res.loc[:,'channelarea'] = interps['CA'](res.x,locsumm.Depth.water)*res.dx
                res.loc[:,'crossarea'] = interps['XA'](res.x,locsumm.Depth.water)#*res.dx
                res.loc[:,'Vwater'] = res.crossarea*res.dx
                #res.loc[:,'W'] = interps['W'](res.x,locsumm.Depth.water)
                #res.loc[:,'Vwater'] = locsumm.Depth.Water*
            else:                
                dt = timeseries.time[t]-timeseries.time[t-1] #For adjusting step size
            #First, update params. Updates:
            #Qin, Qout, RainRate, WindSpeed, 
            #Next, update locsumm
            rainrate = timeseries.RainRate[t] #mm/h
            inflow = timeseries.Qin[t]*3600 #m³/hr
            #Update optional parameters based on timeseries
            try:
                params.loc['Emax','val'] = timeseries.loc[t,'Max_ET']
            except KeyError:
                pass
            # if (t == 255) or (t==2016):#216:#Break at specific timing 216 = first inflow
            # #if timeseries.time[t] == 0.0:
            #     #pdb.set_trace()
            #     yomama = 'great'
            if params.val.flow_routing == 'simple':
                oldV = res.Vwater.sum()
                outflow = timeseries.Qout[t]*3600#m³/hr
                Vr = res.toparea.sum()*rainrate/1E3*dt #m³
                Vin = inflow*dt #m³
                Vout = outflow*dt #m³
                def pond_evap(A,v,RH,Tair,Twat): #https://www.engineeringtoolbox.com/evaporation-water-surface-d_690.html                
                    #Evaporation coefficient
                    theta = (25 + 19*v) #v =windspeed m/s  
                    #Saturated water pressure at T - Buck equation kPa
                    P_sat = 0.61121*np.exp((18.678-Twat/234.5)*(Twat/(257.14+Twat)))
                    #Saturated humidity ratio
                    xs = 0.622*P_sat/(100-P_sat) #kg water/kg air
                    P_rh = RH/100*(0.61121*np.exp((18.678-Tair/234.5)*(Tair/(257.14+Tair))))
                    x = 0.622*P_rh/(100-P_sat) #kg water/kg air
                    gs = theta*A*(xs-x)/1000#m³/hr (assuming 1000kg/m³)
                    return gs
                #Define function to minimize and find new depth based on new volume
                def pond_depth(pond_depth):
                    test_toparea = interps['W'](res.x,pond_depth)*res.dx
                    test_channelarea = interps['CA'](res.x,pond_depth)*res.dx
                    test_XA = interps['XA'](res.x,pond_depth)#*res.dx
                    nomV = res.crossarea*res.dx
                    Vevap = pond_evap(test_toparea,timeseries.WindSpeed[t],timeseries.RH[t],
                                      timeseries.Tair[t],timeseries.Twater[t])
                    testV = oldV+inflow*dt-outflow*dt
                    minimizer = abs(nomV - testV)
                    return minimizer
                testdepth = 0.5
                res.loc[:,'Depth'] = optimize.newton(pond_depth,testdepth,tol=1e-5)
                #res = self.simple_route(res,inflow,outflow,rainrate,dt,params,ponddims)
            else:
                print('Need to implement routing')
            res_t[t] = res.copy(deep=True)
            #Add the 'time' to the resulting dataframe, update some parameters
            res_t[t].loc[:,'time'] = timeseries.loc[t,'time']
            res_t[t].loc[:,'RH'] = timeseries.loc[t,'RH']
            res_t[t].loc[:,'RainRate'] = timeseries.loc[t,'RainRate']
            for j in numc:
                Tj,pHj = 'T'+str(j),'pH'+str(j)
                res_t[t].loc[j,'Temp'] = timeseries.loc[t,Tj] #compartment temperature
                res_t[t].loc[j,'pH'] = timeseries.loc[t,pHj] #Compartment pH
            #For now, assume that inflow only to the water compartm     
    
        res_time = pd.concat(res_t)
        #The structure of this is set up such that V(t) = V(t-1)+sum(Qt). What this means in practice is that the flows
        #are acting on the previous step's V. For the contaminant transport, we need to have the flows acting on the volumes
        #in the same time step. To accomplish this, we will shift it up one and add zeros in the first timestep.
        res_time.loc[:,'V'] = res_time.loc[:,'V'].groupby(level=1).shift(1)
        #Initial condition was zero volume in all cells. 
        res_time.loc[(min(res_time.index.levels[0]),slice(None)),'V'] = 0
        #Now we will also calculate advection from the air compartment - side area of air control volume * windspeed*3600 (windspeed in m/s)
        res_time.loc[(slice(None),'air'),'Q_out'] = np.array(np.sqrt(locsumm.Area.air)*timeseries.loc[:,'WindSpeed'])*\
            res_time.loc[(slice(None),'air'),'Depth']*3600
        
        return res_time
    
    def simple_route(res,inflow,outflow,rainrate,dt,params,ponddims):
        """2025-03-25
        Calculate reach dimensions for a single timestep.
        This code assumes a single bucket for the entire reach.
        These calculations do not depend on the contaminant transport calculations.
        
        Attributes:
        -----------        
        res (df) - discretized control volumes
        inflow (float) - flow into system m³/s
        outflow (float) - flow out of system m³/s
        rainrate (float) - rain onto system mm/h
        params (df) - parameters of the system
        numc (str) - list of compartments
        timeseries (df) - needs to contain inflow in m³/h, rainrate in mm/h
        ponddims (df): Dataframe with pond dimensions along X
        """
        #
        Vin = inflow*dt
        Vout = inflow*dt
        
        
        
        
        
        
        
        
        
        
        
        
        
        