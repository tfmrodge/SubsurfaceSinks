# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:52:52 2019

@author: Tim Rodgers
"""

from FugModel import FugModel #Import the parent FugModel class
from Subsurface_Sinks import SubsurfaceSinks
from HelperFuncs import df_sliced_index #Import helper functions
from scipy import optimize
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pdb #Turn on for error checking
from warnings import simplefilter
#simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
#simplefilter(action="ignore", category= RuntimeWarning)


class BCBlues(SubsurfaceSinks):
    """Bioretention cell implementation of the Subsurface_Sinks model.
    This model represents a vertically flowing stormwater infiltration low impact 
    development (LID) technology. It is intended to be solved as a Level V 
    1D ADRE, across space and time, although it can be modified to make a Level III or
    Level IV multimedia model
        
    Attributes:
    ----------
            
            locsumm (df): physical properties of the systmem
            chemsumm (df): physical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            timeseries (df): Timeseries values providing the time-dependent inputs
            to the model.
            num_compartments (int): (optional) number of non-equilibirum 
            compartments
            name (str): (optional) name of the BC model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,locsumm,chemsumm,params,timeseries,num_compartments = 9,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc) 
        
    def make_system(self,locsumm,params,numc,timeseries,dx = None,flow_time = None):
        """
        This function will build the dimensions of the 1D system based on the "locsumm" input file.
        If you want to specify more things you can can just skip this and input a dataframe directly
        
        Since the hydrology and the contaminant transport are calculated independently,
        we can call them independently to save time & memory
        Either input a raw summary of the BC dimensions, and we will run the hydrology,
        or input the timeseries of BC flows etc. as a locsumm with the flows
        If the input timeseries has the same index as locsumm, it assumes that
        the hydrology has been pre-calculated
        """
        #pdb.set_trace()
        #Check if flow_time exists. flow_time is the "timeseries" of this method, so 
        #change the name.
        if flow_time is None:
            timeseries = self.flow_time(locsumm,params,['water','subsoil'],timeseries)
        else: 
            timeseries = flow_time

        
        #Set up our control volumes
        L = locsumm.loc['subsoil','Depth']+locsumm.loc['topsoil','Depth']
        params.loc['L','val'] = L
        #dx = 0.1
        if dx == None:
            dx = params.val.dx
        #Set up results dataframe - for discretized compartments this is the length of the flowing water
        res = pd.DataFrame(np.arange(0.0+dx/2.0,L,dx),columns = ['x'])
        #This is the number of discretized cells
        numx = res.index #count 
        #Now we will add a final one for the drainage layer/end of system
        res.loc[len(res.index),'x'] = locsumm.loc['subsoil','Depth']+locsumm.loc['topsoil','Depth']\
        +locsumm.loc['drain','Depth']/2
        res.loc[:,'dx'] = dx
        res.loc[len(res.index)-1,'dx'] = locsumm.loc['drain','Depth']

        if len(numc) > 2: #Don't worry about this for the 2-compartment version
            res = res.append(pd.DataFrame([999],columns = ['x']), ignore_index=True) #Add a final x for those that aren't discretized

        #Then, we put the times in by copying res across them
        res_t = dict.fromkeys(timeseries.index.levels[0],[]) 
        for t in timeseries.index.levels[0]:
            res_t[t] = res.copy(deep=True)
        res = pd.concat(res_t)
        #Add the 'time' column to the res dataframe
        #pdb.set_trace()
        #Old way (worked in pandas XXXX) - this worked for everywhere that is currently the long dataframe/array formula.
        #res.loc[:,'time'] = timeseries.loc[:,'time'].reindex(res.index,method = 'bfill')
        res.loc[:,'time'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'time']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]        
        #Set up advection between compartments.
        #The non-discretized compartments don't need to worry about flow the same way - define a mask "discretized mask" or dm
        #Currently set up so that the drainage one is not-dm, rest are. 
        res.loc[:,'dm'] = (res.x!=locsumm.loc['subsoil','Depth']+locsumm.loc['topsoil','Depth']\
        +locsumm.loc['drain','Depth']/2)
        res.loc[res.x==999,'dm'] = False
        #numx = res[res.dm].index.levels[1] #count 
        #Now, we are going to define the soil as a single compartment containing the topsoil and filter
        #Assuming topsoil is same depth at all times.
        res.loc[:,'maskts'] = res.x < timeseries.loc[(min(timeseries.index.levels[0]),'topsoil'),'Depth']
        res.loc[:,'maskss'] = (res.maskts ^ res.dm)
        res.loc[res.maskts,'porositywater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'topsoil'),'Porosity']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]
        res.loc[res.maskss,'porositywater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Porosity']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0] #added so that porosity can vary with x
        #Drainage zone
        res.loc[(slice(None),numx[-1]+1),'porositywater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Porosity']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0] #added so that porosity can vary with x
        #Now we define the flow area as the area of the compartment * porosity * mobile fraction water
        res.loc[res.maskts,'Awater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'topsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                        * res.loc[res.maskts,'porositywater']* params.val.thetam #right now not worrying about having different areas
        res.loc[res.maskss,'Awater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                        * res.loc[res.maskss,'porositywater']* params.val.thetam
        #drainage
        res.loc[(slice(None),numx[-1]+1),'Awater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                                    * res.loc[(slice(None),numx[-1]+1),'porositywater']
        #Now we calculate the volume of the soil
        res.loc[res.dm,'Vsubsoil'] = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        - res.loc[res.dm,'Awater'])*res.dx
        res.loc[(slice(None),numx[-1]+1),'Vsubsoil'] =(pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        - res.loc[(slice(None),numx[-1]+1),'Awater'])*res.dx
        res.loc[:,'V2'] = res.loc[:,'Vsubsoil'] #Limit soil sorption to surface
        #Subsoil area - surface area of contact with the water, based on the specific surface area per m³ soil and water
        res.loc[res.dm,'Asubsoil'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                    *res.dm*params.val.AF_soil
        res.loc[(slice(None),numx[-1]+1),'Asubsoil'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]*params.val.AF_soil
        #For the water compartment assume a linear flow gradient from Qin to Qout - so Q can be increasing or decreasing as we go
        res.loc[res.dm,'Qwater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_towater']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                    - (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_towater']),
                                 index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                    - pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
                                 index=res.index.levels[0]).reindex(res.index,level=0)[0])/L*res.x
        Qslope = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_towater']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        - pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        +pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_towater']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0])/L
        res.loc[res.dm,'Qin'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_towater']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]-\
            Qslope*(res.x - res.dx/2)
        #Out of each cell
        res.loc[res.dm,'Qout'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_towater']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]-\
            Qslope*(res.x + res.dx/2)
        #Water out in last cell is flow to the drain, this is also water into the drain. Net change with capillary flow
        res.loc[(slice(None),numx[-1]+1),'Qin'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        -pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_towater']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #We will put the flow going into the ponding zone into the non-discretized cell.
        #pdb.set_trace()
        if 'pond' in numc:    
            res.loc[(slice(None),numx[-1]+2),'Qin'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_in']),
                         index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #For the drainage compartment assume Qin = Qwater (it either exfiltrates or goes out the pipe)
        res.loc[(slice(None),numx[-1]+1),'Qwater'] = res.loc[(slice(None),numx[-1]+1),'Qin']
        #Assume ET flow from filter zone only, assume proportional to water flow
        #To calculate the proportion of ET flow in each cell, divide the total ETflow for the timestep
        #by the average of the inlet and outlet flows, then divide evenly across the x cells (i.e. divide by number of cells)
        #to get the proportion in each cell, and multiply by Qwater
        ETprop = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'QET']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]/(\
                         (pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
                                      index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                          +pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_towater']),
                                       index=res.index.levels[0]).reindex(res.index,level=0)[0])/2)
        ETprop[np.isnan(ETprop)] = 0 #returns nan if there is no flow so replace with 0
        res.loc[res.dm,'Qet'] = ETprop/len(res.index.levels[1]) * res.Qwater
        res.loc[(slice(None),numx[-1]+1),'Qet'] = 0
        #Qet in plants is additive from bottom to top at each time step.
        #pdb.set_trace()
        res.loc[res.dm,'Qetplant'] = res.Qet[::-1].groupby(level=[0]).cumsum()
        res.loc[:,'Qetsubsoil'] = (1-params.val.froot_top)*res.Qet
        res.loc[:,'Qettopsoil'] = (params.val.froot_top)*res.Qet
        #This is exfiltration across the entire filter zone
        exfprop = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_exf']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]/(\
                         (pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
                                      index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                          +pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_towater']),
                                       index=res.index.levels[0]).reindex(res.index,level=0)[0])/2)
        exfprop[np.isnan(exfprop)] = 0
        exfprop[np.isinf(exfprop)] = 0
        res.loc[res.dm,'Qwaterexf'] = exfprop/len(res.index.levels[1]) * res.Qwater #Amount of exfiltration from the system, for unlined systems
        #We will put the drainage zone exfiltration in the final drainage cell
        res.loc[(slice(None),numx[-1]+1),'Qwaterexf'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_exf']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #Pipe flow is the end of the system
        res.loc[(slice(None),numx[-1]+1),'Qout'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_out']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #If we want more capillary rise than drain zone to filter - currently just net flow is recored.
        res.loc[res.dm,'Qcap'] = 0 #
        #Then water volume and velocity
        dt = timeseries.loc[(slice(None),'pond'),'time'] - timeseries.loc[(slice(None),'pond'),'time'].shift(1)
        dt[np.isnan(dt)] = dt.iloc[1]
        
        #Matrix solution to volumes. Changed from running through each t to running through each x, hopefully faster.
        #NOTE THAT THIS USES LOTS OF RAM!! - basically uses another couple of GB.
        #A slower solution would do better RAM-wise
        numt = len(res.index.levels[0])
        mat = np.zeros([max(numx)+1,numt,numt],dtype=np.int8)
        inp = np.zeros([max(numx)+1,numt])
        m_vals = np.arange(0,numt,1)
        b_vals = np.arange(1,numt,1)
        mat[:,m_vals,m_vals] = 1       
        mat[:,b_vals,m_vals[0:numt-1]] = -1
        #Set last one manually
        #mat[:,numt-1,numt-2] = -1
        for x in numx:
            #RHS of the equation are the net ins and outs from the cell.
            inp[x,:] = np.array((res.loc[(slice(None),0),'Qin']+res.loc[(slice(None),0),'Qcap']-res.loc[(slice(None),0),'Qet']\
                       -res.loc[(slice(None),0),'Qwaterexf']-res.loc[(slice(None),0),'Qout']))*np.array(dt)     
        #Then for the first time step just assume that it is uniformly saturated
        inp[:,0] =  timeseries.loc[(min(res.index.levels[0]),'water'),'V']/(len(numx))
        #for x in range(numx):
        matsol = np.linalg.solve(mat,inp)
        for x in numx:
            res.loc[(slice(None),x),'V1'] = matsol[x,:]
        '''
        #OLD CODE - run through each t rather than each x. Significantly slower, but less RAM
        #Left in in case someone has RAM limitations
        for t in timeseries.index.levels[0]: #This step is slow, as it needs to loop through the entire timeseries.
            if t == timeseries.index.levels[0][0]:
                res.loc[(t,numx),'V1'] = timeseries.loc[(slice(None),'water'),'V'].reindex(res.index,method = 'bfill')\
                        /(len(res[res.dm].index.levels[1])) #First time step just assume that it is uniformly saturated
            else:
                #if t == 5133:
                #    cute = 'peanut'
                res.loc[(t,numx),'V1'] = np.array(res.loc[(t-1,numx),'V1']) + (res.loc[(t,numx),'Qin']+res.loc[(t,numx),'Qcap']-res.loc[(t,numx),'Qet']\
                       -res.loc[(t,numx),'Qwaterexf']-res.loc[(t,numx),'Qout'])*np.array(dt[t])                                        
        '''
        #Volume
        res.loc[(slice(None),numx[-1]+1),'V1'] =  pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain_pores'),'V']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        res.loc[:,'Vwater'] = res.loc[:,'V1'] #

        #Velocity
        res.loc[:,'v1'] = res.Qwater/res.Awater #velocity [L/T] at every x
        #Root volumes & area based off of soil volume fraction.
        #pdb.set_trace()
        res.loc[res.dm,'Vroot'] = res.Vsubsoil*params.val.VFroot #Total root volume per m³ ground volume
        res.loc[res.dm,'Aroot'] = res.Asubsoil*params.val.Aroot #Total root area per m² ground area
        #Don't forget your drainage area - assume roots do not go in the drainage zone
        res.loc[(slice(None),numx[-1]+1),'Vroot'] = 0 #Total root volume per m² ground area
        res.loc[(slice(None),numx[-1]+1),'Aroot'] = 0 #Total root area per m² ground area        
        #Now we loop through the compartments to set everything else.
        #Change the drainage zone to part of the dm 
        res.loc[(slice(None),numx[-1]+1),'dm'] = True
        
        for jind, j in enumerate(numc):
            #Area (A), Volume (V), Density (rho), organic fraction (foc), ionic strength (Ij)
            #water fraction (fwat), air fraction (fair), temperature (tempj), pH (phj)
            jind = jind+1 #The compartment number, for the advection term
            Aj, Vj, Vjind, rhoj, focj, Ij = 'A' + str(j), 'V' + str(j), 'V' + str(jind),'rho' + str(j),'foc' + str(j),'I' + str(j)
            fwatj, fairj, tempj, pHj = 'fwat' + str(j), 'fair' + str(j),'temp' + str(j), 'pH' + str(j)
            rhopartj, fpartj, advj = 'rhopart' + str(j),'fpart' + str(j),'adv' + str(j)
            compartment = j
            if compartment not in ['topsoil', 'drain_pores', 'filter']:#These compartments are only used for flows
                if compartment in ['rootbody', 'rootxylem', 'rootcyl']: #roots are discretized
                    mask= res.dm
                    if compartment in ['rootbody']:
                        res.loc[mask,Vj] = (params.val.VFrootbody+params.val.VFapoplast)*res.Vroot #Main body consists of apoplast and cytoplasm
                    else:
                        VFrj = 'VF' + str(j)
                        #Calculated with the volume fraction
                        res.loc[mask,Vj] = params.loc[VFrj,'val']*res.Vroot
                        #Area of roots - this is calculating the surface area of the roots as a fraction of the overall root area.
                        #We can derive this using the lateral surface area of the roots (neglect tips) and the area & volume fractions
                        #We know that V2/V1 = VF12, and we have V1/A1 = X from the params file. So, A2 = VF12^(-1/3)*A1/V1*V2
                        if compartment in ['rootxylem']: #xylem surrounds the central cylinder so need to add the fractions and the volumes
                            res.loc[mask,Aj] = (params.val.VFrootxylem+params.val.VFrootcyl)**(-1/3)*res.Aroot/res.Vroot\
                            *(params.val.VFrootxylem+params.val.VFrootcyl)*res.Vroot
                        else:
                            res.loc[mask,Aj] = params.loc[VFrj,'val']**(-1/3)*res.Aroot/res.Vroot*res.loc[mask,Vj]
                    res.loc[mask,Vjind] = res.loc[mask,Vj] #Needed for the FugModel module
                elif compartment in ['water','subsoil']: #water and subsoil
                    mask = res.dm
                else:#Other compartments aren't discretized
                    mask = res.dm==False
                    res.loc[mask,Vj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'V']),
                                 index=res.index.levels[0]).reindex(res.index,level=0)[0]
                    res.loc[mask,Vjind] = res.loc[mask,Vj] #Needed for FugModel ABC
                    res.loc[mask,Aj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Area']),
                                 index=res.index.levels[0]).reindex(res.index,level=0)[0]
                res.loc[mask,focj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'FrnOC']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] #Fraction organic matter
                res.loc[mask,Ij] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'cond']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]*1.6E-5 #Ionic strength from conductivity #Plants from Trapp (2000) = 0.5
                res.loc[mask,fwatj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'FrnWat']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] #Fraction water
                res.loc[mask,fairj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Frnair']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] #Fraction air
                res.loc[mask,fpartj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'FrnPart']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] #Fraction particles
                res.loc[mask,tempj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Temp']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] + 273.15 #Temperature [K]
                res.loc[mask,pHj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'pH']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] #pH
                res.loc[mask,rhopartj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'PartDensity']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] #Particle density
                res.loc[mask,rhoj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Density']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0] #density for every x [M/L³]
                res.loc[mask,advj] =pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Q_out']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
                if compartment == 'air': #Set air density based on temperature
                    res.loc[mask,rhoj] = 0.029 * 101325 / (params.val.R * res.loc[:,tempj])
                    #res.loc[mask,advj] = np.sqrt(locsumm.loc['air','Area'])*locsumm.loc['air','Depth']     
        res.loc[res.dm,'Arootsubsoil'] = res.Aroot #Area of roots in direct contact with subsoil
        #Vertical facing soil area
        res.loc[res.dm,'AsoilV'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Area']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #For bioretention, we want this interface in the top of the soil and in the non-discretized compartment (if present)
        #pdb.set_trace()
        mask = (res.x == min(res.x)) | (res.x == 999.)
        res.loc[mask,'Asoilair'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'topsoil'),'Area']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0] #Only the "topsoil" portion of the soil interacts with the air
        #Shoot area based off of leaf area index (LAI) 
        if 'shoots' in numc: #skip if no shoots aren't included
            res.loc[res.dm==False,'A_shootair'] = params.val.LAI*res.Asoilair #Total root volume per m² ground area

        #Longitudinal Dispersivity. Calculate using relationship from Schulze-Makuch (2005) 
        #for unconsolidated sediment unless a value of alpha [L] is given
        if 'alpha' not in params.index:
            params.loc['alpha','val'] = 0.2 * L**0.44 #alpha = c(L)^m, c = 0.2 m = 0.44
        res.loc[res.dm,'ldisp'] = params.val.alpha * res.v1 #Resulting Ldisp is in [L²/T]
        #Replace nans with 0s for the next step
        res = res.fillna(0)
        return res
    #'''    
    def bc_dims(self,locsumm,inflow,rainrate,dt,params):
        """
        Calculate BC dimension & compartment information for a given time step.
        Forward calculation of t(n+1) from inputs at t(n)
        The output of this will be a "locsumm" file which can be fed into the rest of the model.
        These calculations do not depend on the contaminant transport calculations.
        
        This module includes particle mass balances, where particles are
        advective transfer media for compounds in the model.
        Water flow modelling based on Randelovic et al (2016)
        
        Attributes:
        -----------
        locsumm (df) -  gives the conditions at t(n). 
        Inflow (float) - m³/s, 
        rainrate (float) - mm/h
        dt (float) - in h
        params (df) - parameters dataframe. Hydrologic parameters are referenced here. 
        """
        
        res = locsumm.copy(deep=True)
        #pdb.set_trace()
        #For first round, calculate volume
        if 'V' not in res.columns:
            res.loc[:,'V'] = res.Area * res.Depth #Volume m³
            #Now we are going to make a filter zone, which consists of the volume weighted
            #averages of he topsoil and the subsoil.
            res.loc['filter',:] = (res.V.subsoil*res.loc['subsoil',:]+res.V.topsoil*\
                   res.loc['topsoil',:])/(res.V.subsoil+res.V.topsoil)
            res.loc['filter','Depth'] = res.Depth.subsoil+res.Depth.topsoil
            res.loc['filter','V'] = res.V.subsoil+res.V.topsoil
            res.Porosity['filter'] = res.Porosity['filter']*params.val.thetam #Effective porosity - the rest is the immobile water fraction
            res.loc['water','V'] = res.V['filter']*res.FrnWat['filter'] #water fraction in subsoil - note this is different than saturation
            res.loc['drain_pores','V'] = res.V.drain*res.FrnWat.drain
            res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
            res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment doesn't exist
            res.loc[np.isinf(res.loc[:,'P']),'P'] = 0
            res.loc['filter','Discrete'] = 1.
        #Define the BC geometry
        pondV = np.array(params.val.BC_Volume_Curve.split(","),dtype='float')
        pondH = np.array(params.val.BC_Depth_Curve.split(","),dtype='float')
        pondA = np.array(params.val.BC_pArea_curve.split(","),dtype='float')

        #filter saturation
        Sss = res.V.water /(res.V['filter'] * (res.Porosity['filter']))
        #ponding Zone
        #Convert rain to flow rate (m³/hr). Direct to ponding zone
        Qr_in = res.Area.air*rainrate/1E3 #m³/hr
        #Infiltration Kf is in m/hr
        #Potential
        Qinf_poss = params.val.Kf * (res.Depth.pond + res.Depth['filter'])\
        /res.Depth['filter']*res.Area['pond']
        #Upstream max flow (from volume)
        Qinf_us = 1/dt*(res.V.pond)+(Qr_in+inflow)
        #Downstream capacity
        #Maximum infiltration to native soils through filter and submerged zones, Kn is hydr. cond. of native soil
        Q_max_inf = params.val.Kn * (res.Area['filter'] + res.P['drain_pores']*params.val.Cs)
        Qinf_ds= 1/dt * ((1-Sss) * res.Porosity.subsoil * res.V.subsoil) +Q_max_inf #Why not Qpipe here?
        #FLow from pond to subsoil zone
        Q26 = max(min(Qinf_poss,Qinf_us,Qinf_ds),0)
        #Overflow from system - everything over top of cell
        if res.V.pond > pondV[-1]:
            Qover = (1/dt)*(res.V.pond-pondV[-1])
            res.loc['pond','V'] += -Qover*dt
            res.loc['pond','Depth'] = np.interp(res.V.pond,pondV,pondH, left = 0, right = pondH[-1])
            res.loc['pond','Area'] = np.interp(res.V.pond,pondV,pondA, left = 0, right = pondA[-1]) 
        else:
            Qover = 0
        #Flow over weir
        if res.Depth.pond > params.val.Hw:
            #Physically possible
            def pond_depth(pond_depth):
                if pond_depth < params.val.Hw:
                    minimizer = 999
                else:
                    Q2_wp = params.val.Cq * params.val.Bw * np.sqrt(2.*9.81*3600.**2.*(pond_depth - params.val.Hw)**3.)
                    Q2_wus =  1/dt * ((pond_depth - params.val.Hw)*res.Area.pond) + (Qr_in+inflow) - Q26-Qover
                    Q2_w = max(min(Q2_wp,Q2_wus),0)
                    dVp = (inflow + Qr_in - Q26 - Q2_w-Qover)*dt
                    oldV = res.loc['pond','V']
                    testV = oldV + dVp
                    oldD = pond_depth
                    testD = np.interp(testV,pondV,pondH, left = 0, right = pondH[-1])
                    minimizer = abs(oldD - testD)
                return minimizer
            try:
                res.loc['pond','Depth'] = optimize.newton(pond_depth,res.Depth.pond,tol=1e-5)
            except RuntimeError:#This indicates no solution converged, caused by draining to weir height 
                res.loc['pond','Depth'] = params.val.Hw
                
            if res.Depth.pond < params.val.Hw:#
                res.loc['pond','Depth'] = params.val.Hw
            Q2_wp = params.val.Cq * params.val.Bw * np.sqrt(2.*9.81*3600.**2.*(res.Depth.pond - params.val.Hw)**3.)
            #Upstream Control
            Q2_wus = 1/dt * ((res.Depth.pond - params.val.Hw)*res.Area.pond) + (Qr_in+inflow) - Q26-Qover
            Q2_w = max(min(Q2_wp,Q2_wus),0)
        else:
            Q2_w = 0
        #Exfiltration to surrounding soil from pond. Assume that infiltration happens preferentially as it
        #flows on the infiltrating part.
        #Maximum possible
        Qpexf_poss = 0 #Always over the system, there would be no drainage out from the bottom of the pond
        #Upstream availability, no need for downstream as it flows out of the system
        Qpexf_us = 1/dt*(res.V.pond) + (Qr_in+inflow-Q26-Q2_w)
        Q2_exf = max(min(Qpexf_poss,Qpexf_us),0) #Actual exfiltration
        #pond Volume from mass balance
        #Change in pond volume dVp at t 
        dVp = (inflow + Qr_in - Q26 - Q2_w-Qover - Q2_exf)*dt
        (inflow + Qr_in - Q26 - Q2_w- Qover)*dt
        if (res.loc['pond','V'] + dVp) < 0:
            #Correct, this will only ever be a very small error.
            dVp = -res.loc['pond','V']
        res.loc['pond','V'] += dVp
        res.loc['pond','Depth'] = np.interp(res.V.pond,pondV,pondH, left = 0, right = pondH[-1])  
        #Area of pond/soil surface m² at t+1
        res.loc['pond','Area'] = np.interp(res.V.pond,pondV,pondA, left = 0, right = pondA[-1])  
        res.loc['air','Area'] = pondA[-1] - res.Area.pond #Just subtract the area that is submerged
    
                
        #Pore water Flow - filter zone
        #Capillary Rise - from drain/submerged zone to subsoil zone.
        if Sss > params.val.Ss and Sss < params.val.Sfc:
            Cr = 4 * params.val.Emax/(2.5*(params.val.Sfc - params.val.Ss)**2)
            Q10_cp = res.Area['filter'] * Cr * (Sss-params.val.Ss)*(params.val.Sfc - Sss)
            #Upstream volume available (in drain layer)
            Q10_cus = (res.V.drain_pores)/dt
            #Space available in pore_filt
            Q10_cds = 1/dt * ((1 - Sss)*res.V['filter']) - Q26
            Q106 = max(min(Q10_cp,Q10_cus,Q10_cds),0)
        else: 
            Q106 = 0
        #Estimated saturation at time step t+1
        S_est = min(1.0,Sss+Q26*dt/(res.V['filter']*res.Porosity['filter']))
        #Infiltration from filter to drainage layer
        Q6_infp = res.Area.subsoil*params.val.Kf*S_est*(res.Depth.pond + res.Depth['filter'])/res.Depth['filter']
        if S_est < params.val.Sh: #No volume available in filter zone if estimated below hygroscopic point
            Q6_inf_us = 0#(Q26+Q106)
        else:#Here we use the saturation value at the last time step, not the actual one. This should keep at hygroscopic point.
            Q6_inf_us = 1/dt*((Sss-params.val.Sh)*res.Porosity['filter']*res.V['filter'])+(Q26+Q106)
        Q610 = max(min(Q6_infp,Q6_inf_us),0)
        #Flow due to evapotranspiration. Some will go our the air, some will be transferred to the plants for cont. transfer?
        if S_est <= params.val.Sh:
            Q6_etp = 0
        elif S_est <= params.val.Sw:
            Q6_etp = res.Area['filter'] * params.val.Ew*(Sss-params.val.Sh)\
            /(params.val.Sw - params.val.Sh)
        elif S_est <= params.val.Ss:
            Q6_etp = res.Area['filter'] * (params.val.Ew +(params.val.Emax - params.val.Ew)\
            *(Sss-params.val.Sw)/(params.val.Ss - params.val.Sw))
        else:
            Q6_etp = res.Area['filter']*params.val.Emax
        #Upstream available
        Q6_etus = 1/dt* ((Sss-params.val.Sh)*res.V['filter']*res.Porosity['filter']) +(Q26+Q106-Q610)
        Q6_et = max(min(Q6_etp,Q6_etus),0)
        
        #topsoil and subsoil pore water - water
        #going to try a single unified water compartment, with velocities that 
        #change depending on the zone. Might mess some things up if drain zone is fast?
        #Change in pore water volume dVf at t
        #pdb.set_trace()
        dVf = (Q26 + Q106 - Q610 - Q6_et)*dt
        res.loc['water','V'] += dVf
        #subsoil Saturation (in the water depth column) at t+1
        Sss = res.V.water /(res.V['filter'] * (res.Porosity['filter'])) 
        
        #Water flow - drain/submerged zone
        #So this is a "pseudo-compartment" which will be incorporated into the 
        #subsoil compartment for the purposes of contaminant transport hopefully.
        #Exfiltration from drain zone to native soil
        Q10_exfp = params.val.Kn * (res.Area.drain + params.val.Cs*\
                   res.P.drain*res.Depth.drain_pores/res.Depth.drain)
        Q10_exfus = 1/dt*((1-res.Porosity.drain)*(res.Depth.drain_pores-params.val.hpipe)*res.Area.drain) + (Q610-Q106)
        Q10_exf = max(min(Q10_exfp,Q10_exfus),0)
        dVdest = (Q610 - Q106 - Q10_exf)*dt #Estimate the height for outflow - assuming free surface at top of bucket
        draindepth_est =  (res.loc['drain_pores','V'] + dVdest) /(res.Area.drain * (res.Porosity.drain))
        #drain through pipe - this is set to just remove everything each time step, probably too fast
        piperem = params.val.hpipe/100 #Keep the level above the top of the pipe so that flow won't go to zero
        if draindepth_est >= (params.val.hpipe+piperem): #Switched to estimated drainage depth 20190912. Added orifice control 20201019
            Q10_us = 1/dt * ((draindepth_est-(params.val.hpipe+piperem))*(1-res.Porosity.drain)\
            *res.Area.drain)
            #Orifice control of outflow. This does not differentiate between water in the pipe and outside of the pipe, but
            #simply restricts the pipe flow based on the orifice opening. Code is from the SWMM manual for a partially
            #submerged side orifice. 
            #First, we determine how far up the pipe the flow has gotten. F= fraction of pipe
            F = min((draindepth_est+piperem)/(params.val.hpipe+params.val.Dpipe),1)#Maximum is 1, then it is just weir equation.
            #Then, we use that to adjust our orifice discharge equation Co = Cd*Ao*sqrt(2gh). Here we use the full area
            #of the orifice, h is the total head in the drainage zone. Remember units are /hr in this code.
            #Total head in drainage zone is the head less the capillary head
            htot = draindepth_est+res.loc['filter','Depth']+res.loc['pond','Depth']
            Co = params.val.Cd*1/4*np.pi*(params.val.Dpipe*params.val.fvalveopen)**2*np.sqrt(2*9.81*3600**2*htot)
            #Then, we take the lower of the upstream control and the downstream (valve) control as the outlet.
            Q10_ds = Co*F**1.5
            Q10_pipe = min(Q10_ds,Q10_us)
        else: Q10_pipe = 0;


        #drain Pore water Volume from mass balance\
        #I think this will have to be a separate compartment.
        #Change in drain pore water volume dVd at t
        dVd = (Q610 - Q106 - Q10_exf - Q10_pipe)*dt
        res.loc['drain_pores','V'] += dVd
        #Height of submerged zone - control variable for flow leaving SZ
        res.loc['drain_pores','Depth'] = res.V.drain_pores /(res.Area.drain * (res.Porosity.drain))
        
        #Put final flows into the res df. Flow rates are given as flow from a compartment (row)
        #to another compartment (column). Flows out of the system have their own
        #columns (eg exfiltration, ET, outflow), as do flows into the system.
        res.loc['pond','Q_towater'] = Q26 #Infiltration to subsoil
        res.loc['pond','Q_out'] = Q2_w +Qover #Weir + overflow
        res.loc['pond','Q_exf'] = Q2_exf #exfiltration from pond
        res.loc['pond','Q_in'] = inflow + Qr_in #From outside system
        res.loc['pond','QET'] = rainrate/1E3 #Needed to calculate wet deposition, put here for ease of use.
        res.loc['water','Q_todrain'] = Q610 #Infiltration to drain layer
        res.loc['water','QET'] = Q6_et #Infiltration to drain layer
        res.loc['water','Q_exf'] = 0 #Assumed zero
        res.loc['drain','Q_towater'] = Q106 #Capillary rise
        res.loc['drain','Q_exf'] = Q10_exf #exfiltration from drain layer
        res.loc['drain','Q_out'] = Q10_pipe #Drainage from system
        #Calculate VFwater based on saturation
        res.loc['subsoil','FrnWat'] = Sss*res.Porosity['filter']
        res.loc['topsoil','FrnWat'] = Sss*res.Porosity['filter']
        res.loc['drain','FrnWat'] = res.V.drain_pores/res.V.drain
        #Calculate VFair for drain and subsoil zones based on saturation
        #pdb.set_trace()
        res.loc['subsoil','Frnair'] = res.Porosity['filter'] - Sss*res.Porosity['filter']
        res.loc['topsoil','Frnair'] = res.Porosity['filter'] - Sss*res.Porosity['filter']
        res.loc['drain','Frnair'] = res.Porosity.drain - res.FrnWat.drain/res.Porosity.drain
        res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
        res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment has no volume
        res.loc[np.isinf(res.loc[:,'P']),'P'] = 0 #If the compartment has no volume
        return res
    #'''
    def bc_dims(self,locsumm,inflow,rainrate,dt,params):
        """
        Calculate BC dimension & compartment information for a given time step.
        Forward calculation of t(n+1) from inputs at t(n)
        The output of this will be a "locsumm" file which can be fed into the rest of the model.
        These calculations do not depend on the contaminant transport calculations.
        
        Water flow modelling based on Randelovic et al (2016)
        
        Attributes:
        -----------
        locsumm (df) -  gives the conditions at t(n). 
        Inflow (float) - m³/s, 
        rainrate (float) - mm/h
        dt (float) - in h
        params (df) - parameters dataframe. Hydrologic parameters are referenced here. 
        """
        
        res = locsumm.copy(deep=True)
        #pdb.set_trace()
        #First, set up functions for the different zones. 3 types of zone - pond, filter, drain
        #Define calculations for a ponding zone.  
        #Pulled calcs to function 20220107
        #compartments represents the compartments that are involved. First one must be 
        #the name of the ponding zone, then the filter layer the pond infiltrates into.
        #For the Kortright BC, this will be compartments = ['pond','filter','water']
        #pdb.set_trace()
        def pondzone(inflow,rainrate,dt,params,res,compartments):
            #First, define pond geometry based on params
            pondV = np.array(params.val.BC_Volume_Curve.split(","),dtype='float')
            pondH = np.array(params.val.BC_Depth_Curve.split(","),dtype='float')
            pondA = np.array(params.val.BC_pArea_curve.split(","),dtype='float')
            #Convert rain to flow rate (m³/hr). Direct to ponding zone
            Qr_in = res.Area.air*rainrate/1E3 #m³/hr
            #Infiltration Kf is in m/hr
            #Potential
            Qinf_poss = params.val.Kf * (res.Depth.pond + res.Depth[compartments[1]])\
            /res.Depth[compartments[1]]*res.Area[compartments[0]]
            #Upstream max flow (from volume)
            Qinf_us = 1/dt*(res.V.pond)+(Qr_in+inflow)
            #Downstream capacity
            #Maximum infiltration to native soils through filter and submerged zones, Kn is hydr. cond. of native soil
            Q_max_inf = params.val.Kn * (res.Area[compartments[1]] + res.P['drain_pores']*params.val.Cs)
            Sss = res.V[compartments[2]] /(res.V[compartments[1]]*(res.Porosity[compartments[1]])) 
            Qinf_ds= 1/dt * ((1-Sss) * res.Porosity.subsoil * res.V.subsoil) +Q_max_inf 
            #FLow from pond to top filter zone
            Q26 = max(min(Qinf_poss,Qinf_us,Qinf_ds),0)
            #Overflow from system - everything over top of cell 
            if res.V.pond > pondV[-1]: #pond is larger than ponding capacity
                Qover = (1/dt)*(res.V.pond-pondV[-1])
                res.loc[compartments[0],'V'] += -Qover*dt
                res.loc[compartments[0],'Depth'] = np.interp(res.V.pond,pondV,pondH, left = 0, right = pondH[-1])
                res.loc[compartments[0],'Area'] = np.interp(res.V.pond,pondV,pondA, left = 0, right = pondA[-1]) 
            else:
                Qover = 0
            #Flow over weir
            if res.Depth.pond > params.val.Hw:
                #Physically possible
                def pond_depth(pond_depth):
                    if pond_depth < params.val.Hw:
                        minimizer = 999
                    else:
                        Q2_wp = params.val.Cq * params.val.Bw * np.sqrt(2.*9.81*3600.**2.*(pond_depth - params.val.Hw)**3.)
                        Q2_wus =  1/dt * ((pond_depth - params.val.Hw)*res.Area.pond) + (Qr_in+inflow) - Q26-Qover
                        Q2_w = max(min(Q2_wp,Q2_wus),0)
                        dVp = (inflow + Qr_in - Q26 - Q2_w-Qover)*dt
                        oldV = res.loc[compartments[0],'V']
                        testV = oldV + dVp
                        oldD = pond_depth
                        testD = np.interp(testV,pondV,pondH, left = 0, right = pondH[-1])
                        minimizer = abs(oldD - testD)
                    return minimizer
                try:
                    res.loc[compartments[0],'Depth'] = optimize.newton(pond_depth,res.Depth.pond,tol=1e-5)
                except RuntimeError:#This indicates no solution converged, caused by draining to weir height 
                    res.loc[compartments[0],'Depth'] = params.val.Hw
                    
                if res.Depth.pond < params.val.Hw:#
                    res.loc[compartments[0],'Depth'] = params.val.Hw
                Q2_wp = params.val.Cq * params.val.Bw * np.sqrt(2.*9.81*3600.**2.*(res.Depth.pond - params.val.Hw)**3.)
                #Upstream Control
                Q2_wus = 1/dt * ((res.Depth.pond - params.val.Hw)*res.Area.pond) + (Qr_in+inflow) - Q26-Qover
                Q2_w = max(min(Q2_wp,Q2_wus),0)
            else:
                Q2_w = 0
            #Exfiltration to surrounding soil from pond. Assume that infiltration happens preferentially as it
            #flows on the infiltrating part.
            #Maximum possible
            Qpexf_poss = 0 #Always over the system, there would be no drainage out from the bottom of the pond
            #Upstream availability, no need for downstream as it flows out of the system
            Qpexf_us = 1/dt*(res.V.pond) + (Qr_in+inflow-Q26-Q2_w)
            Q2_exf = max(min(Qpexf_poss,Qpexf_us),0) #Actual exfiltration
            #pond Volume from mass balance
            #Change in pond volume dVp at t 
            dVp = (inflow + Qr_in - Q26 - Q2_w-Qover - Q2_exf)*dt
            #(inflow + Qr_in - Q26 - Q2_w- Qover)*dt
            if (res.loc[compartments[0],'V'] + dVp) < 0:
                #Correct, this will only ever be a very small error.
                dVp = -res.loc[compartments[0],'V']
            res.loc[compartments[0],'V'] += dVp
            res.loc[compartments[0],'Depth'] = np.interp(res.V.pond,pondV,pondH, left = 0, right = pondH[-1])  
            #Area of pond/soil surface m² at t+1
            res.loc[compartments[0],'Area'] = np.interp(res.V.pond,pondV,pondA, left = 0, right = pondA[-1])  
            res.loc['air','Area'] = pondA[-1] - res.Area.pond #Just subtract the area that is submerged
            #Put necessary values in the output file
            res.loc[compartments[0],'Q_towater'] = Q26 #Infiltration to subsoil
            res.loc[compartments[0],'Q_out'] = Q2_w+Qover #Weir + overflow
            res.loc[compartments[0],'Q_exf'] = Q2_exf #exfiltration from pond
            res.loc[compartments[0],'Q_in'] = inflow + Qr_in #From outside system
            #res.loc[compartments[0],'QET'] = rainrate/1E3 #Needed to calculate wet deposition, put here for ease of use.
            return res,Q26

        #Next, define the calculations for a filter zone.
        #"Qin_f" is the flow into the filter zone - either from an external source,
        #from a ponding zone, or from another filter zone.
        #The 'compartment' variable will tell the model what compartments to update in the code
        #Format is [current compartment, current compartment water, compartment below water, compartment above (if present)]
        #e.g. ['filter','water','drain_pores','pond'] for the BC in rodgers et al (2021)
        #or ['topsoil','topsoil_pores','water'] for the soil above the influent in the Quebec st. tree trench.
        #['subsoil','water','drain_pores','topsoil'] for lower zone in quebec st. 
        def filterzone(Qin_f,dt,params,res,compartments):                    
            #First, define filter saturation
            Sss = res.V[compartments[1]] /(res.V[compartments[0]]*(res.Porosity[compartments[0]]))                
            #Pore water Flow - filter zone
            #Capillary Rise - from zone below to this zone
            if Sss > params.val.Ss and Sss < params.val.Sfc:
                Cr = 4 * params.val.Emax/(2.5*(params.val.Sfc - params.val.Ss)**2)
                Q10_cp = res.Area[compartments[0]] * Cr * (Sss-params.val.Ss)*(params.val.Sfc - Sss)
                #Upstream volume available (in lower layer)
                Q10_cus = (res.V[compartments[2]])/dt
                #Space available in pore_filt
                Q10_cds = 1/dt * ((1 - Sss)*res.V[compartments[0]]) - Qin_f
                Qcap = max(min(Q10_cp,Q10_cus,Q10_cds),0)
            else: 
                Qcap = 0
            #Estimated saturation at time step t+1
            S_est = min(1.0,Sss+Qin_f*dt/(res.V[compartments[0]]*res.Porosity[compartments[0]]))
            #Potential infiltration from filter to layer below
            #If there is a layer above, it should be compartments[3]
            try:
                Q6_infp = res.Area[compartments[0]]*params.val.Kf*S_est*(res.Depth[compartments[3]]*res.FrnWat[compartments[3]]
                                                                         + res.Depth[compartments[0]])/res.Depth[compartments[0]]
            except IndexError:
                Q6_infp = res.Area[compartments[0]]*params.val.Kf*S_est                
            if S_est < params.val.Sh: #No volume available in filter zone if estimated below hygroscopic point
                Q6_inf_us = 0#(Qin_f+Qcap)
            else:#Here we use the saturation value at the last time step, not the actual one. This should keep at hygroscopic point.
                Q6_inf_us = 1/dt*((Sss-params.val.Sh)*res.Porosity[compartments[0]]*res.V[compartments[0]])+(Qin_f+Qcap)
            Qinf_f = max(min(Q6_infp,Q6_inf_us),0)
            #Flow due to evapotranspiration.Transferred to the plants for cont. transfer
            if S_est <= params.val.Sh:
                Q6_etp = 0
            elif S_est <= params.val.Sw:
                Q6_etp = res.Area[compartments[0]] * params.val.Ew*(Sss-params.val.Sh)\
                /(params.val.Sw - params.val.Sh)
            elif S_est <= params.val.Ss:
                Q6_etp = res.Area[compartments[0]] * (params.val.Ew +(params.val.Emax - params.val.Ew)\
                *(Sss-params.val.Sw)/(params.val.Ss - params.val.Sw))
            else:
                Q6_etp = res.Area[compartments[0]]*params.val.Emax
            #Upstream available
            Q6_etus = 1/dt* ((Sss-params.val.Sh)*res.V[compartments[0]]*res.Porosity[compartments[0]]) +(Qin_f+Qcap-Qinf_f)
            Qet_f = max(min(Q6_etp,Q6_etus),0)
            
            #topsoil and subsoil pore water - water
            #going to try a single unified water compartment, with velocities that 
            #change depending on the zone. Might mess some things up if drain zone is fast?
            #Change in pore water volume dVf at t
            #pdb.set_trace()
            dVf = (Qin_f + Qcap - Qinf_f - Qet_f)*dt
            res.loc[compartments[1],'V'] += dVf
            #subsoil Saturation (in the water depth column) at t+1
            Sss = res.V[compartments[1]] /(res.V[compartments[0]] * (res.Porosity[compartments[0]])) 
            Q_exf = 0 #Assumed zero, can change if needed. Currently all exfiltration is at the bottom. 
            #Update the res dataframe           
            res.loc[compartments[1],'QET'] = Qet_f #
            res.loc[compartments[1],'Q_exf'] = Q_exf
            res.loc[compartments[0],'FrnWat'] = Sss*res.Porosity[compartments[0]]
            res.loc[compartments[0],'Frnair'] = res.Porosity[compartments[0]] - Sss*res.Porosity[compartments[0]]
            res.loc[compartments[1],'Q_out'] = Qet_f+Qinf_f+Q_exf
            res.loc[compartments[1],'Q_in'] = Qin_f + Qcap
            return res,Qinf_f,Qcap
        
        
        
        #Water flow - drain/submerged zone
        #compartments that are used. For Kortright ['drain','drain_pores','filter','pond']
        #for tree trench ['drain','drain_pores','subsoil','topsoil']
        def drainzone(Qin_d,Qcap,dt,params,res,compartments):
            #Exfiltration from drain zone to native soil
            Q10_exfp = params.val.Kn * (res.Area.drain + params.val.Cs*\
                       res.P.drain*res.Depth.drain_pores/res.Depth.drain)
            Q10_exfus = 1/dt*((1-res.Porosity.drain)*(res.Depth.drain_pores-params.val.hpipe)*res.Area.drain) + (Qin_d-Qcap)
            Q10_exf = max(min(Q10_exfp,Q10_exfus),0)
            dVdest = (Qin_d - Qcap - Q10_exf)*dt #Estimate the height for outflow - assuming free surface at top of bucket
            draindepth_est =  (res.loc[compartments[1],'V'] + dVdest) /(res.Area.drain * (res.Porosity.drain))
            piperem = params.val.hpipe/100 #Keep the level above the top of the pipe so that flow won't go to zero
            if draindepth_est >= (params.val.hpipe+piperem): #Switched to estimated drainage depth 20190912. Added orifice control 20201019
                Q10_us = 1/dt * ((draindepth_est-(params.val.hpipe+piperem))*(1-res.Porosity.drain)\
                *res.Area.drain)
                #Orifice control of outflow. This does not differentiate between water in the pipe and outside of the pipe, but
                #simply restricts the pipe flow based on the orifice opening. Code is from the SWMM manual for a partially
                #submerged side orifice. 
                #First, we determine how far up the pipe the flow has gotten. F= fraction of pipe
                F = min((draindepth_est+piperem)/(params.val.hpipe+params.val.Dpipe),1)#Maximum is 1, then it is just weir equation.
                #Then, we use that to adjust our orifice discharge equation Co = Cd*Ao*sqrt(2gh). Here we use the full area
                #of the orifice, h is the total head in the drainage zone. Remember units are /hr in this code.
                #Total head in drainage zone is the head less the capillary head
                htot = draindepth_est
                for layer in compartments[2:]:#For simplicity assume height of water column is depth * FrnWat 
                    htot += res.loc[layer,'Depth']*res.loc[layer,'FrnWat']
                #htot = draindepth_est+res.loc['filter','Depth']+res.loc['pond','Depth']
                Co = params.val.Cd*1/4*np.pi*(params.val.Dpipe*params.val.fvalveopen)**2*np.sqrt(2*9.81*3600**2*htot)
                #Then, we take the lower of the upstream control and the downstream (valve) control as the outlet.
                Q10_ds = Co*F**1.5
                Q10_pipe = min(Q10_ds,Q10_us)
            else: Q10_pipe = 0;   
            #drain Pore water Volume from mass balance\
            #Change in drain pore water volume dVd at t
            dVd = (Qin_d - Qcap - Q10_exf - Q10_pipe)*dt
            res.loc[compartments[1],'V'] += dVd
            #Height of submerged zone - control variable for flow leaving SZ
            res.loc[compartments[1],'Depth'] = res.V.drain_pores /(res.Area.drain * (res.Porosity.drain))
            #Update the res dataframe
            res.loc['water','Q_todrain'] = Qin_d #Infiltration to drain layer
            res.loc[compartments[0],'Q_towater'] = Qcap #Capillary rise
            res.loc[compartments[0],'Q_exf'] = Q10_exf #exfiltration from drain layer
            res.loc[compartments[0],'Q_todrain'] = Q10_pipe #Drainage from system - put in "Q_todrain" 20220110, used to be Q_out
            res.loc[compartments[0],'Q_out'] = Q10_pipe+Q10_exf+Qcap
            res.loc[compartments[0],'FrnWat'] = res.V[compartments[1]]/res.V[compartments[0]]
            res.loc[compartments[0],'Frnair'] = res.Porosity[compartments[0]] - res.FrnWat[compartments[0]]/res.Porosity[compartments[0]]
            res.loc[compartments[0],'Q_in'] = Qin_d
            res.loc[compartments[0],'QET'] = rainrate/1E3 #Putting rainrate here for access later. 20220111
            return res
        
        #Put final flows into the res df. Flow rates are given as flow from a compartment (row)
        #to another compartment (column). Flows out of the system have their own
        #columns (eg exfiltration, ET, outflow), as do flows into the system.
        #After we define the functions, put the pieces together according to 
        #the hydrology_structure parameter. This defines what compartments are used
        #in the system. Currently not well-utilized, but could allow for discretizing
        #the flow calculations with a bit of work.
        #Right now this is just hardcoded for a bioretention system (with pond)
        #or a tree-trench (no pond)
        #hydro_struc = params.val.hydrology_structure
        try:
            if (params.val.hydrology_structure == 'Tree_Trench') | (params.val.hydrology_structure == 'TT'):
               hydro_struc = [['topsoil','topsoil_pores'],['subsoil','water'],['drain','drain_pores']]
            else:
               hydro_struc = [['pond'],['filter','water'],['drain','drain_pores']]
        except AttributeError:#Default is a bioretention system if no parameters are given
           hydro_struc = [['pond'],['filter','water'],['drain','drain_pores']]   
        if 'pond' in hydro_struc[0]:
            #For first round, calculate volume
            if 'V' not in res.columns:
                res.loc[:,'V'] = res.Area * res.Depth #Volume m³
                #Now we are going to make a filter zone, which consists of the volume weighted
                #averages of he topsoil and the subsoil.
                res.loc['filter',:] = (res.V.subsoil*res.loc['subsoil',:]+res.V.topsoil*\
                       res.loc['topsoil',:])/(res.V.subsoil+res.V.topsoil)
                res.loc['filter','Depth'] = res.Depth.subsoil+res.Depth.topsoil
                res.loc['filter','V'] = res.V.subsoil+res.V.topsoil
                res.Porosity['filter'] = res.Porosity['filter']*params.val.thetam #Effective porosity - the rest is the immobile water fraction
                res.loc['water','V'] = res.V['filter']*res.FrnWat['filter'] #water fraction in subsoil - note this is different than saturation
                res.loc['drain_pores','V'] = res.V.drain*res.FrnWat.drain
                res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
                res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment doesn't exist
                res.loc[np.isinf(res.loc[:,'P']),'P'] = 0
                res.loc['filter','Discrete'] = 1.
            #Then, do pond calculations
            res,Q26 = pondzone(inflow,rainrate,dt,params,res,['pond','filter','water'])
            #Pond flows to filter zone. Qin_f = Q26
            res,Qinf_f,Qcap = filterzone(Q26,dt,params,res,['filter','water','drain_pores','pond'])
            #Finally, drainage zone. Qin_d = Qinf_f
            res = drainzone(Qinf_f,Qcap,dt,params,res,['drain','drain_pores','filter','pond'])
            #Need to set some values for subsoil and topsoil manually
            res.loc['subsoil','FrnWat'] = res.loc['filter','FrnWat']
            res.loc['topsoil','FrnWat'] = res.loc['filter','FrnWat']
            res.loc['subsoil','Frnair'] = res.loc['filter','Frnair']
            res.loc['topsoil','Frnair'] = res.loc['filter','Frnair']
        #compartments that are used. For Kortright ['drain','drain_pores','filter','pond']
        #for tree trench ['drain','drain_pores','subsoil','topsoil']
        elif 'topsoil_pores' in hydro_struc[0]:
            #For first round, calculate volume
            if 'V' not in res.columns:
                res.loc[:,'V'] = res.Area * res.Depth #Volume m³
                res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth)
                res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment doesn't exist
                res.loc[np.isinf(res.loc[:,'P']),'P'] = 0
                res.loc['topsoil_pores','V'] = res.V.topsoil*res.FrnWat.topsoil
                res.loc['water','V'] = res.V.subsoil*res.FrnWat.subsoil
                res.loc['drain_pores','V'] = res.V.drain*res.FrnWat.drain
            #First, topsoil. 
            res,Qinf_topsoil,Qcaptopsoil = filterzone(0,dt,params,res,['topsoil','topsoil_pores','water'])
            #Next, subsoil. Water enters here
            Qin_subsoil = inflow + Qinf_topsoil - Qcaptopsoil
            res,Qinf_subsoil,Qcapsubsoil = filterzone(Qin_subsoil,dt,params,res,['subsoil','water','drain_pores','topsoil'])
            #Finally, drainage zone. Qin_d = Qinf_f
            res = drainzone(Qinf_subsoil,Qcapsubsoil,dt,params,res,['drain','drain_pores','subsoil','topsoil'])
            #Update topsoil manually 
            res.loc['topsoil','QET'] = res.loc['topsoil_pores','QET'] #
            res.loc['topsoil','Q_exf'] = res.loc['topsoil_pores','Q_exf'] #
            res.loc['topsoil','Q_out'] = res.loc['topsoil_pores','Q_out']
            #Only water in to topsoil is capillary rise
            res.loc['topsoil','Q_in'] = Qcaptopsoil
            #Infiltration from topsoil to subsoil.
            res.loc['topsoil','Q_towater'] = Qinf_topsoil
        else:
            exit("Need to define a recognized hydrological structure")
        #Recalculate perimeter at end of time step.
        res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
        res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment has no volume
        res.loc[np.isinf(res.loc[:,'P']),'P'] = 0 #If the compartment has no volume
        return res


    def flow_time(self,locsumm,params,numc,timeseries):
        """
        Step through the flow calculations in time.             

        Attributes:
        -----------        
        locsumm (df) - gives the dimensions of the system at the initial conditions
        params (df) - parameters of the system
        numc (str) - list of compartments
        timeseries (df) - needs to contain inflow in m³/h, rainrate in mm/h
        """
        #dt = timeseries.time[1]-timeseries.time[0] #Set this so it works
        res = locsumm.copy(deep=True)
        ntimes = len(timeseries['time'])
        #Set up 4D output dataframe by adding time as the third multi-index
        #Index level 0 = time, level 1 = chems, level 2 = cell number
        times = timeseries.index
        res_t = dict.fromkeys(times,[]) 
        #pdb.set_trace()
        for t in range(ntimes):
            if t == 0:
                dt = timeseries.time[1]-timeseries.time[0]
            else:                
                dt = timeseries.time[t]-timeseries.time[t-1] #For adjusting step size
            #First, update params. Updates:
            #Qin, Qout, RainRate, WindSpeed, 
            #Next, update locsumm
            rainrate = timeseries.RainRate[t] #mm/h
            inflow = timeseries.Qin[t] #m³/h
            params.loc['fvalveopen','val'] = timeseries.fvalveopen[t]
            
            #pdb.set_trace()
            #Hardcoding switch in Kf after tracer event. If Kn2 is in the params it will switch
            if timeseries.time[t] > 7.:
                try:
                    params.loc['Kn','val'] = params.val.Kn2
                except AttributeError:
                    pass
            if t == 341:#216:#Break at specific timing 216 = first inflow
                #pdb.set_trace()
                yomama = 'great'
            res = self.bc_dims(res,inflow,rainrate,dt,params)
            res_t[t] = res.copy(deep=True)
            #Add the 'time' to the resulting dataframe
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

    def draintimes(self,timeseries,res_time):
        """
        Determine the draindage times - time between peak depth and zero depth (https://wiki.sustainabletechnologies.ca/wiki/Bioretention:_Sizing)     
        
        Attributes:
        ----------- 
        timeseries (df) - timeseries inputs
        res_time - output from the flow_time method.
        """
        #Find the places where depth is zero. The time between these indices are our events. 
        zerodepths = timeseries.iloc[np.where(res_time.loc[(slice(None),'pond'),'Depth'] == 0.)].time
        zerodepths = zerodepths[zerodepths>0]
        draintimes = []
        drainidx = []
        #pdb.set_trace()
        for ind,idx in enumerate(zerodepths.index):
            #Skip first cell and any consecutive indices 
            if (idx == zerodepths.index[0]) | (idx-1 in zerodepths.index):
                pass
            else:
                lastzero = zerodepths.index[np.where(zerodepths == zerodepths.iloc[ind-1])[0][0]]
                maxDind = lastzero+ np.where(res_time.loc[(slice(lastzero,idx),'pond'),'Depth']
                == max(res_time.loc[(slice(lastzero,idx),'pond'),'Depth']))[0][0]
                draintimes.append(zerodepths[idx] - timeseries.loc[maxDind,'time'])
                drainidx.append(idx) 
        
        return draintimes, drainidx   
            
    def BC_fig(self,numc,mass_balance,time=None,compound=None,figname='BC_Model_Figure.tif',dpi=100,fontsize=8,figheight=6,dM_locs=None,M_locs=None):
            """ 
            Show modeled fluxes and mass distributions on a bioretention cell figure. 
            Just a wrapper to give the correct figure to the main function
            
            Attributes:
            -----------
                mass_balance = Either the cumulative or the instantaneous mass balance, as output by the appropriate functions, normalized or not
                figname (str, optional) = name of figure on which to display outputs. Default figure should be in the same folder
                time (float, optional) = Time to display, in hours. Default will be the last timestep.
                compounds (str, optional) = Compounds to display. Default is all.
            """
            #pdb.set_trace()
            try:
                dM_locs[1]
            except TypeError:
                #Define locations of the annotations if not given This is done by hand.    
                dM_locs = {'Meff':(0.835,0.215),'Min':(0.013,0.490),'Mexf':(0.581,0.235),'Mrwater':(0.850,0.360),
                          'Mrsubsoil':(0.82,0.280),'Mrrootbody':(0.560,0.780),'Mrrootxylem':(0.73,0.835),
                          'Mrrootcyl':(0.835,0.775),'Mrshoots':(0.135,0.760),'Mrair':(0.025,0.830),
                          'Mrpond':(0.794,0.560),'Mnetwatersubsoil':(0.046,0.295),'Mnetwaterpond':(0.18,0.360),
                          'Mnetsubsoilrootbody':(0.679,0.664),'Mnetsubsoilshoots':(0.260,0.387),
                          'Mnetsubsoilair':(0.636,0.545),'Mnetsubsoilpond':(0.013,0.390),'Mnetrootbodyrootxylem':(0.835,0.635),
                          'Mnetrootxylemrootcyl':(0.875,0.680),'Mnetrootcylshoots':(0.50,0.443),
                          'Mnetshootsair':(0.489,0.585),'Mnetairpond':(0.090,0.545),'Madvair':(0.828,0.885),'Madvpond':(0.850,0.475)}
                #Location of the massess
                M_locs = {'Mwater':(0.075,0.242),'Msubsoil':(0.075,0.217),'Mrootbody':(0.530,0.930),
                        'Mrootxylem':(0.747,0.930),'Mrootcyl':(0.747,0.961),'Mshoots':(0.530,0.961),
                        'Mair':(0.095,0.955),'Mpond':(0.739,0.453)}             
            fig,ax = self.model_fig(numc,mass_balance=mass_balance,time = time,
                                    compound=compound,figname=figname,dpi=dpi,
                                    fontsize=fontsize,figheight=figheight,
                                    dM_locs=dM_locs,M_locs=M_locs)
            return fig,ax
    
    def run_BC(self,locsumm,chemsumm,timeseries,numc,params,pp=None):
        """
        Run the BC Blues model! This will run the entire model with a given
        parameterization.              

        Attributes:
        -----------        
        locsumm (df) - gives the dimensions of the system at the initial conditions
        chemsumm (df) - physicochemical parameters of the chemicals under investigation
        timeseries (df) - timeseries data for the model run.          
        params (df) - parameters of the system
        numc (str) - list of compartments
        timeseries (df) - needs to contain inflow in m³/h, rainrate in mm/h
        pp (df) - (optional) polyparameter linear free energy relationship parameters to 
        use in the model. 
        """
        bioretention_cell = BCBlues(locsumm,chemsumm,params,timeseries,numc)
        #Run the model
        #pdb.set_trace()
        #The flows are calculated only with the water and soil compartments as numc
        flow_time = bioretention_cell.flow_time(locsumm,params,['water','subsoil'],timeseries)
        mask = timeseries.time>=0
        minslice = np.min(np.where(mask))
        maxslice = np.max(np.where(mask))#minslice + 5 #
        flow_time = df_sliced_index(flow_time.loc[(slice(minslice,maxslice),slice(None)),:])
        input_calcs = bioretention_cell.input_calc(locsumm,chemsumm,params,pp,numc,timeseries,flow_time=flow_time)
        model_outs = bioretention_cell.run_it(locsumm,chemsumm,params,pp,numc,
                                         timeseries,input_calcs=input_calcs)       
        return model_outs
    
    
    def plot_flows(self,flow_time,Qmeas=None,compartments=['water','drain'],yvar ='Q_exf',**kwargs):
        
        #pdb.set_trace()
        #yvar = 'Q_out'
        #for comp in compartments:
            
        #comp1 = 'water'
        #comp2 = 'drain'
        #comp3 = 'pond'
        #shiftdist = 0
        mask = flow_time.time>=0
        #pltdata = flow_time.loc[(mask,compartments),:]

        #pltdata = flow_time.loc[(slice(210,timeseries.index[-1]),comp2),:] #To plot 
        #pltdata2 = flow_time.loc[(slice(210,timeseries.index[-1]),comp3),:]
        #res_time.loc[(plttime,slice(None),slice(None)),slice(None)] #Just at plttime
        #ylim = ylim #[0, 4]
        #xlim = [0,48]
        ylabel = 'Flow Rate (m³/h)'
        xlabel = 'Time'
        #pltdata = res_time #All times at once
        fig,ax = plt.subplots(1,1,figsize=(14,8),dpi=300)
        #fig = plt.figure(figsize=(14,8))
        #ax = sns.lineplot(x = pltdata.index.get_level_values(0),hue = pltdata.index.get_level_values(1),y=yvar,data = pltdata)
        #ax = sns.lineplot(data = pltdata,x = 'time',hue = pltdata.index.get_level_values(1),y=yvar)
        #ax = plt.plot()
        for comp in compartments:            
            ax.plot(flow_time.loc[(mask,comp),'time'],flow_time.loc[(mask,comp),yvar])
        #ax2 = sns.lineplot(x = pltdata2.loc[(slice(None),'pond'),'time'],hue = pltdata2.index.get_level_values(1),y='Depth',data = pltdata2)
        #ax2.set_xlim(xlim)
        #ax.set_ylim(ylim)#KGE
        if Qmeas is None:
            pass#labels = compartments
        else:
            #pltdata.loc[(slice(None),'Qmeas'),yvar] = Qmeas
            startind = len(Qmeas)- len(flow_time.loc[(mask,compartments[0]),'time'])
            ax.plot(flow_time.loc[(mask,compartments[0]),'time'],Qmeas[startind:],'#808080',zorder = 1) 
            compartments.append('Qmeas')
            #Calculate KGE
            
            #labels = [compartments,'Qmeas']
            #ax.legend(labels = [compartments,'Qmeas'])
            #leg = ax.get_legend_handles_labels()
            #leg[0].append(art[0])
            #leg[1].append('Qmeas')
            #ax.get_legend().remove()
            #leg = [*leg,[art,'Qmeas']]
        
        ax.legend(labels=compartments)            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_xlim(xlim)
        ax.tick_params(axis='both', which='major', labelsize=15)    
    