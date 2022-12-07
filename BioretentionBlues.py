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
import seaborn as sns; sns.set_style('ticks')
import matplotlib.pyplot as plt
import pdb #Turn on for error checking
from warnings import simplefilter
from hydroeval import kge #Kling-Gupta efficiency (Kling-Gupta et al., 2009)
import hydroeval
from scipy.optimize import minimize
import psutil
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category= RuntimeWarning)


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
        if params.val.covered == 0: #Bioretention system - entire soil column discretized
            L = locsumm.loc['subsoil','Depth']+locsumm.loc['topsoil','Depth']
        else: #Tree trench - only portion below influent pipe discretized
            L = locsumm.loc['subsoil','Depth']
        params.loc['L','val'] = L
        #dx = 0.1
        if dx == None:
            dx = params.val.dxs
        #Set up results dataframe - for discretized compartments this is the length of the flowing water
        #20220712 - if L/dx isn't an integer, will adjust dx so that L is included.
        numx = int(L/dx) #count 
        #Update dx to ensure continuity
        dx = L/numx
        res = pd.DataFrame(np.arange(0.0+dx/2.0,L,dx),columns = ['x'])
        #This is the number of discretized cells
        
        #Now we will add a final one for the drainage layer/end of system
        res.loc[len(res.index),'x'] = L + locsumm.loc['drain','Depth']/2
        res.loc[:,'dx'] = dx
        res.loc[len(res.index)-1,'dx'] = locsumm.loc['drain','Depth']

        if len(numc) > 2: #Don't worry about this for the 2-compartment version
            res = res.append(pd.DataFrame([999],columns = ['x']), ignore_index=True) #Add a final x for those that aren't discretized

        #Then, we put the times in by reindexing. Updated from old loop 20220111
        resind = [timeseries.index.levels[0],res.index]
        res = res.reindex(index=pd.MultiIndex.from_product(resind),level=1)
        
        #Add the 'time' column to the res dataframe
        #pdb.set_trace()
        #Old way (worked in pandas XXXX) - this worked for everywhere that is currently the long dataframe/array formula.
        #res.loc[:,'time'] = timeseries.loc[:,'time'].reindex(res.index,method = 'bfill')
        res.loc[:,'time'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'time']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]        
        #Set up advection between compartments.
        #The non-discretized compartments don't need to worry about flow the same way - define a mask "discretized mask" or dm
        #Currently set up so that the drainage one is not-dm, rest are. 
        
        res.loc[:,'dm'] = (res.x!=L+locsumm.loc['drain','Depth']/2)
        res.loc[res.x==999,'dm'] = False
        #numx = res[res.dm].index.levels[1] #count 
        
        #Now, we are going to define the soil as a single compartment containing the topsoil and filter
        #Assuming topsoil is same depth at all times.
        if params.val.covered == 1: 
            res.loc[:,'maskts'] = False
            #res.loc[:,'maskss'] = (True ^ res.dm)
        else:
            res.loc[:,'maskts'] = res.x < timeseries.loc[(min(timeseries.index.levels[0]),'topsoil'),'Depth']
        res.loc[:,'maskss'] = (res.maskts ^ res.dm)
        res.loc[res.maskts,'porositywater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'topsoil'),'Porosity']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]
        res.loc[res.maskss,'porositywater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Porosity']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0] #added so that porosity can vary with x
        #Drainage zone
        #Changed indexing 20220712 from res.loc[(slice(None),numx[-1]+1),'porositywater']
        res.loc[(slice(None),numx),'porositywater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Porosity']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0] #added so that porosity can vary with x
        #Now we define the flow area as the area of the compartment * porosity * mobile fraction water
        res.loc[res.maskts,'Awater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'topsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                        * res.loc[res.maskts,'porositywater']* params.val.thetam #right now not worrying about having different areas
        res.loc[res.maskss,'Awater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                        * res.loc[res.maskss,'porositywater']* params.val.thetam
        #drainage
        res.loc[(slice(None),numx),'Awater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                                    * res.loc[(slice(None),numx),'porositywater']
        #Now we calculate the volume of the soil
        res.loc[res.dm,'Vsubsoil'] = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        - res.loc[res.dm,'Awater'])*res.dx
        res.loc[(slice(None),numx),'Vsubsoil'] =(pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        - res.loc[(slice(None),numx),'Awater'])*res.dx
        res.loc[:,'V2'] = res.loc[:,'Vsubsoil'] #Limit soil sorption to surface
        #Subsoil area - surface area of contact with the water, based on the specific surface area per m³ soil and water
        res.loc[res.dm,'Asubsoil'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'subsoil'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]\
                                    *res.dm*params.val.AF_soil
        res.loc[(slice(None),numx),'Asubsoil'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Area']),
                                                           index=res.index.levels[0]).reindex(res.index,level=0)[0]*params.val.AF_soil
        #For the water compartment assume a linear flow gradient from Qin to Qout - so Q can be increasing or decreasing as we go
        #res.loc[res.dm,'Qwater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
        #                     index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        #            - (pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
        #                         index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        #            - pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
        #                         index=res.index.levels[0]).reindex(res.index,level=0)[0])/L*res.x
        Qslope = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        - pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        +pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_towater']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0])/L
        res.loc[res.dm,'Qin'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]-\
            Qslope*(res.x - res.dx/2)
        #Out of each cell
        res.loc[res.dm,'Qout'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]-\
            Qslope*(res.x + res.dx/2)
        #Water out in last cell is flow to the drain, this is also water into the drain. Net change with capillary flow
        res.loc[(slice(None),numx),'Qin'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        -pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_towater']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        
        #We will put the flow going into the ponding zone into the non-discretized cell.
        #pdb.set_trace()
        #20220712 changed indexing from res.loc[(slice(None),numx[-1]+2),'Qin']
        if 'pond' in numc:    
            res.loc[(slice(None),numx+1),'Qin'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_in']),
                         index=res.index.levels[0]).reindex(res.index,level=0)[0]
        else:#Put the flow from the subsoil to the topsoil. 
            res.loc[(slice(None),numx+1),'Qin'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'topsoil_pores'),'Q_in']),
                         index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #Now we will define Qwater as the flow at x in each cell
        res.loc[res.dm,'Qwater'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]-\
            Qslope*(res.x)
        #For the drainage compartment Qin = Qwater (it either exfiltrates or goes out the pipe)
        res.loc[(slice(None),numx),'Qwater'] = res.loc[(slice(None),numx),'Qin']
        #Assume ET flow from filter zone only, assume proportional to water flow
        #To calculate the proportion of ET flow in each cell, divide the total ETflow for the timestep
        #by the average of the inlet and outlet flows, then divide evenly across the x cells (i.e. divide by number of cells)
        #to get the proportion in each cell, and multiply by Qwater
        #ETprop = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'QET']),
        #             index=res.index.levels[0]).reindex(res.index,level=0)[0]/(\
        #                 (pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
        #                              index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        #                  +pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
        #                               index=res.index.levels[0]).reindex(res.index,level=0)[0])/2)
        #ETprop[np.isnan(ETprop)] = 0 #returns nan if there is no flow so replace with 0
        #res.loc[res.dm,'Qet'] = ETprop/numx * res.Qwater
        #***20220809 Above is BROKEN*** Even allocation of Qet works great.
        res.loc[res.dm,'Qet'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'QET']/numx),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        res.loc[(slice(None),numx),'Qet'] = 0
        #Qet in plants is additive from bottom to top at each time step.
        #pdb.set_trace()
        res.loc[res.dm,'Qetplant'] = res.Qet[::-1].groupby(level=[0]).cumsum()
        res.loc[:,'Qetsubsoil'] = (1-params.val.froot_top)*res.Qet
        res.loc[:,'Qettopsoil'] = (params.val.froot_top)*res.Qet
        #This is exfiltration across the entire filter zone
        #exfprop = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_exf']),
        #             index=res.index.levels[0]).reindex(res.index,level=0)[0]/(\
        #                 (pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_todrain']),
        #                              index=res.index.levels[0]).reindex(res.index,level=0)[0]\
        #                  +pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'Q_in']),
        #                               index=res.index.levels[0]).reindex(res.index,level=0)[0])/2)
        #exfprop[np.isnan(exfprop)] = 0
        #exfprop[np.isinf(exfprop)] = 0
        exfprop = 0 #No exfiltration outside of drainage zone at the moment.
        res.loc[res.dm,'Qwaterexf'] = exfprop/len(res.index.levels[1]) * res.Qwater #Amount of exfiltration from the system, for unlined systems
        #We will put the drainage zone exfiltration in the final drainage cell
        res.loc[(slice(None),numx),'Qwaterexf'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_exf']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #Pipe flow is the end of the system
        #20220711 - changed from "Qout" to "Q_todrain" as Qout is now the sum of ALL outflows, pipeflow moved to Q_todrain
        res.loc[(slice(None),numx),'Qout'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'Q_todrain']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #If we want more capillary rise than drain zone to filter - currently just net flow is recored.
        res.loc[res.dm,'Qcap'] = 0 #
        #Then water volume and velocity
        dt = timeseries.loc[(slice(None),numc[0]),'time'] - timeseries.loc[(slice(None),numc[0]),'time'].shift(1)
        dt[np.isnan(dt)] = dt.iloc[1]
        #2022-10-11 since all flows are spread among all cells, this is true
        res.loc[(slice(None),slice(numx-1)),'V1'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'water'),'V']/numx),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        
        '''2022-10-11 Removed as it was equivalent to above expression for current set-up
        #Matrix solution to volumes. Changed from running through each t to running through each x, hopefully faster.
        #dV = V(t) - V(t-1); dV = sumQ*dt
        #NOTE THAT THIS USES LOTS OF RAM!! - uses up to 100s of GB (scales as numt**2)
        #A slower solution would do better RAM-wise
        #2022-10-11 edited to account for amount of RAM available. This will now take 
        #almost all the RAM you have available rather than blowing past it! 
        memavail = psutil.virtual_memory()[1] #In bytes (memavail-psutil.virtual_memory()[1])/1e9
        numloops = int(np.ceil(np.sqrt((len(res.index.levels[0])**2*numx*8)/memavail)))
        tmax = min(res.index.levels[0])#Starting step
        laststep = timeseries.loc[(min(res.index.levels[0]),'water'),'V']/(numx)
        for ind in range(numloops): #Define number of timesteps per loop 
            if ind != (numloops-1):
                numt = int(np.floor(len(res.index.levels[0])/numloops))
                #tmax += numt
            else:
                numt = max(res.index.levels[0]) - tmax +1#len(res.index.levels[0]) - tmax
                #tmax = max(res.index.levels[0])
            tmax += numt
            #We are going to repeat the calculation for the "last-step" cell each time - expanded to "numt+1"
            #for ind>0
            if ind > 0:
                addcell = 1
            else:
                addcell = 0
            mat = np.zeros([numx,numt,numt],dtype=np.int8)#1 byte per value
            matsol = np.zeros([numx,numt+addcell,numt+addcell],dtype=np.float32)
            inp = np.zeros([numx,numt+addcell])
            m_vals = np.arange(0,numt+addcell,1)
            b_vals = np.arange(1,numt+addcell,1)
            mat[:,m_vals,m_vals] = 1       
            mat[:,b_vals,m_vals[0:numt-1+addcell]] = -1
            #Set last one manually
            #mat[:,numt-1,numt-2] = -1
            #20220111 Fixed bug where all Xs were being set by x = 0. 
            startt = tmax-numt-addcell
            for x in range(numx):
                #RHS of the equation are the net ins and outs from the cell.
                #inp[x,:] = np.array((res.loc[(slice(None),0),'Qin']+res.loc[(slice(None),0),'Qcap']-res.loc[(slice(None),0),'Qet']\
                #           -res.loc[(slice(None),0),'Qwaterexf']-res.loc[(slice(None),0),'Qout']))*np.array(dt)
                inp[x,:] = np.array((res.loc[(slice(startt,tmax-1),x),'Qin']+res.loc[(slice(startt,tmax-1),x),'Qcap']-res.loc[(slice(startt,tmax-1),x),'Qet']\
                           -res.loc[(slice(startt,tmax-1),x),'Qwaterexf']-res.loc[(slice(startt,tmax-1),x),'Qout']))*np.array([dt[:numt+addcell]])  
            #Since flow calculations are explicit, V(t) = V(t-1) + sum(Q(t-1))*dt, need to shift down one
            inp = np.roll(inp,1,axis=1)#Note that this will put the final column in zero, we fix below
            #Specify initial conditions for each loop
            #if ind == 0:
                #For initial conditions at t = 0 use last time step
            #    inp[:,0] = timeseries.loc[(min(res.index.levels[0]),'water'),'V']/(numx)
            #else:#Otherwise, use the stored results from the last step to start off. 
            inp[:,0] = laststep     
            matsol = np.linalg.solve(mat,inp)
            #Add the drain zone cell and the non-discretized cell
            newrow = [matsol[0,:]*np.nan]
            matsol = np.r_[matsol,newrow]
            matsol = np.r_[matsol,newrow]
            matsol = matsol.reshape(matsol.shape[0]*matsol.shape[1],order = 'f')
            res.loc[(slice(startt,tmax-1),slice(None)),'V1'] = matsol
            #set the last step
            laststep = np.array(res.loc[(slice(tmax-1,tmax-1),slice(numx-1)),'V1'])
        '''
        '''
        #Very Old CODE - run through each t rather than each x. Significantly slower, but less RAM
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
        #Volume of drainage cell
        res.loc[(slice(None),numx),'V1'] =  pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain_pores'),'V']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        res.loc[:,'Vwater'] = res.loc[:,'V1'] #

        #Velocity
        res.loc[:,'v1'] = res.Qwater/res.Awater #velocity [L/T] at every x
        #Root volumes & area based off of soil volume fraction.
        #pdb.set_trace()
        res.loc[res.dm,'Vroot'] = res.Vsubsoil*params.val.VFroot #Total root volume per m³ ground volume
        res.loc[res.dm,'Aroot'] = res.Asubsoil*params.val.Aroot #Total root area per m² ground area
        #Don't forget your drainage area - assume roots do not go in the drainage zone
        res.loc[(slice(None),numx),'Vroot'] = 0 #Total root volume per m² ground area
        res.loc[(slice(None),numx),'Aroot'] = 0 #Total root area per m² ground area        
        #Now we loop through the compartments to set everything else.
        #Change the drainage zone to part of the dm 
        res.loc[(slice(None),numx),'dm'] = True
        
        for jind, j in enumerate(numc):
            #Area (A), Volume (V), Density (rho), organic fraction (foc), ionic strength (Ij)
            #water fraction (fwat), air fraction (fair), temperature (tempj), pH (phj)
            jind = jind+1 #The compartment number, for the advection term
            Aj, Vj, Vjind, rhoj, focj, Ij = 'A' + str(j), 'V' + str(j), 'V' + str(jind),'rho' + str(j),'foc' + str(j),'I' + str(j)
            fwatj, fairj, tempj, pHj = 'fwat' + str(j), 'fair' + str(j),'temp' + str(j), 'pH' + str(j)
            rhopartj, fpartj, advj = 'rhopart' + str(j),'fpart' + str(j),'adv' + str(j)
            compartment = j        
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
                    # We know that V2/V1 = VF12, and we have V1/A1 = X from the params file. So, A2 = VF12^(-1/3)*A1/V1*V2
                    if compartment in ['rootxylem']: #xylem surrounds the central cylinder so need to add the fractions and the volumes
                        res.loc[mask,Aj] = (params.val.VFrootxylem+params.val.VFrootcyl)**(-1/3)*res.Aroot/res.Vroot\
                        *(params.val.VFrootxylem+params.val.VFrootcyl)*res.Vroot
                    else:
                        res.loc[mask,Aj] = params.loc[VFrj,'val']**(-1/3)*res.Aroot/res.Vroot*res.loc[mask,Vj]
                res.loc[mask,Vjind] = res.loc[mask,Vj] #Needed for the FugModel module
                res.loc[mask,advj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Q_out']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
            elif compartment in ['water','subsoil']: #water and subsoil
                mask = res.dm
                res.loc[mask,advj] = 0;#Exfiltration is dealt with separately as D_exf. May need to add soil removal.
            else:#Other compartments aren't discretized
                mask = res.dm==False
                res.loc[mask,Vj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'V']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
                res.loc[mask,Vjind] = res.loc[mask,Vj] #Needed for FugModel ABC
                res.loc[mask,Aj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Area']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
                res.loc[mask,advj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Q_out']),
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
            if compartment == 'air': #Set air density based on temperature
                res.loc[mask,rhoj] = 0.029 * 101325 / (params.val.R * res.loc[:,tempj])
            elif compartment in ['topsoil']:
                res.loc[mask,'Qet'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'QET']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
                res.loc[mask,'Qtopsoil_subsoil'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Q_towater']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
                res.loc[mask,'Qsubsoil_topsoil'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Q_in']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
                #Advective out of system is exfiltration, ET goes to plants.
                res.loc[mask,advj] = pd.DataFrame(np.array(timeseries.loc[(slice(None),compartment),'Q_exf']),
                             index=res.index.levels[0]).reindex(res.index,level=0)[0]
                
                #res.loc[mask,advj] = np.sqrt(locsumm.loc['air','Area'])*locsumm.loc['air','Depth']     
        res.loc[res.dm,'Arootsubsoil'] = res.Aroot #Area of roots in direct contact with subsoil
        if 'topsoil' in numc:
            res.loc[res.dm==False,'Aroottopsoil'] = res.Atopsoil*params.val.Aroot 
        #Vertical facing soil area. Put all through.
        res.loc[:,'AsoilV'] = pd.DataFrame(np.array(timeseries.loc[(slice(None),'topsoil'),'Area']),
                     index=res.index.levels[0]).reindex(res.index,level=0)[0]
        #For bioretention, we want this interface in the top of the soil and in the non-discretized compartment (if present)
        #pdb.set_trace()
        mask = (res.x == min(res.x)) | (res.x == 999.)
        if params.val.covered is True:
            res.loc[mask,'Asoilair'] = 0
        else:
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
        #defragment in case the frame is fragmented (getting a warning but it isn't a huge frame)
        res = res.copy()
        return res

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
        
        ChangeLog
        -----------
        2022-08-05: Added native soil as a separate filter layer to estimate Kunsat,N
            -Changed equations in the filter zone from relying on Ksat*S to calculated K_unsat 
            using Van-Genuchten equation. This should make for lower flows at the beginning.
            -Added downstream limitation to the drainage zone exfiltration based on the native_soil zone                
        """
        
        res = locsumm.copy(deep=True)
        #pdb.set_trace()
        #First, set up functions for the different zones. 3 types of zone - pond, filter, drain
        #Define calculations for a ponding zone.  
        #Pulled calcs to function 20220107
        def k_unsat_vg(S,Ksat,params):
            #Using van genuchten drainage equation as presented in Roy-Poirier et al (2015) DOI: 10.2166/wst.2015.368
            #S is water content/saturation, K is saturated hydraulic conductivity
            #Params has to have the other params in it
            #Calculate effective saturation using van-genuchten equation
            if S < params.val.Sh:
                S_eff = params.val.Sh+0.01
            elif S > params.val.Sfc:
                S_eff = 1.0
            else:
                S_eff = (S - params.val.Sh)/(params.val.Sfc - params.val.Sh)
            #Calculate van genuchten m value
            m_vg = 1-1/params.val.n_vg
            #Get the unsaturated value
            K_unsat = Ksat*S_eff**(1/2)*(1-(1-S_eff**(1/m_vg))**m_vg)**2
            return K_unsat
        #compartments represents the compartments that are involved. First one must be 
        #the name of the ponding zone, then the filter layer the pond infiltrates into.
        #For the Kortright BC, this will be compartments = ['pond','filter','water']
        #pdb.set_trace()
        def pondzone(inflow,rainrate,dt,params,res,compartments):
            #First, define pond geometry based on params
            pondV = np.array(params.val.BC_Volume_Curve.split(","),dtype='float')
            pondH = np.array(params.val.BC_Depth_Curve.split(","),dtype='float')
            pondA = np.array(params.val.BC_Area_curve.split(","),dtype='float')
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
            #pdb.set_trace()
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
                if compartments[2] == None: #No capillary rise to native soil 
                    Q10_cus = 0
                else:
                    Q10_cus = (res.V[compartments[2]])/dt
                #Space available in pore_filt
                Q10_cds = 1/dt * ((1 - Sss)*res.V[compartments[0]]) - Qin_f
                Qcap = max(min(Q10_cp,Q10_cus,Q10_cds),0)
            else: 
                Qcap = 0
            #Qcap = 0
            #Estimated saturation at time step t+1
            S_est = min(1.0,Sss+Qin_f*dt/(res.V[compartments[0]]*res.Porosity[compartments[0]]))
            #Potential infiltration from filter to layer below
            #Need to define the saturated conductivity to be used:
            if compartments[0] == 'native_soil': 
                Ksat = params.val.Kn
            else:
                Ksat = params.val.Kf
            #Estimate K_unsat
            K_unsat = k_unsat_vg(S_est,Ksat,params)
            #If there is a layer above, it should be compartments[3]
            try:
                Q6_infp = res.Area[compartments[0]]*K_unsat*(res.Depth[compartments[3]]*res.FrnWat[compartments[3]]
                                                                         + res.Depth[compartments[0]])/res.Depth[compartments[0]]
            except IndexError:
                Q6_infp = res.Area[compartments[0]]*K_unsat                
            if S_est < params.val.Sh: #No volume available in filter zone if estimated below hygroscopic point
                Q6_inf_us = 0#(Qin_f+Qcap)
            else:#Here we use the saturation value at the last time step, not the actual one. This should keep at hygroscopic point.
                Q6_inf_us = 1/dt*((Sss-params.val.Sh)*res.Porosity[compartments[0]]*res.V[compartments[0]])+(Qin_f+Qcap)
            Qinf_f = max(min(Q6_infp,Q6_inf_us),0)
            #Native soil unconnected to plants - no ET. 
            if compartments[0] == 'native_soil':
                Qet_f = 0
            else:
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
                #Qet_f = 0

            #Change in pore water volume dVf at t
            #pdb.set_trace()
            dVf = (Qin_f + Qcap - Qinf_f - Qet_f)*dt
            res.loc[compartments[1],'V'] += dVf
            #subsoil Saturation (in the water depth column) at t+1
            Sss = res.V[compartments[1]] /(res.V[compartments[0]] * (res.Porosity[compartments[0]])) 
            #Estimate unsaturated conductivity based on saturation
            if compartments[0] == 'native_soil':
                params.loc['Kn_unsat','val'] = Ksat#k_unsat_vg(Sss,Ksat,params)#
                
            Q_exf = 0 #Assumed zero, can change if needed. Currently all exfiltration is at the bottom. 
            #Update the res dataframe           
            res.loc[compartments[1],'QET'] = Qet_f #
            res.loc[compartments[1],'Q_exf'] = Q_exf
            res.loc[compartments[0],'FrnWat'] = Sss*res.Porosity[compartments[0]]
            res.loc[compartments[0],'Frnair'] = res.Porosity[compartments[0]] - Sss*res.Porosity[compartments[0]]
            res.loc[compartments[1],'Q_out'] = Qet_f+Qinf_f+Q_exf
            res.loc[compartments[1],'Q_in'] = Qin_f #+ Qcap
            return res,Qinf_f,Qcap
        
        #Water flow - drain/submerged zone
        #compartments that are used. For BC ['drain','drain_pores','filter','pond']
        #for tree trench ['drain','drain_pores','subsoil','topsoil']
        def drainzone(Qin_d,Qcap,dt,params,res,compartments):
            #Exfiltration from drain zone to native soil
            #Adjust the hydraulic conductivity of the native soil
            #if 'native_soil' in compartments:
            #    Kn_unsat
            try:
                Q10_exfp = params.val.Kn_unsat * (res.Area.drain + params.val.Cs*\
                           res.P.drain*res.Depth.drain_pores/res.Depth.drain)
            except AttributeError:#If no unsat is given, will just use saturated value. Not ideal without spin-up period.
                Q10_exfp = params.val.Kn * (res.Area.drain + params.val.Cs*\
                           res.P.drain*res.Depth.drain_pores/res.Depth.drain)
            Q10_exfus = 1/dt*((1-res.Porosity.drain)*(res.Depth.drain_pores-params.val.hpipe)*res.Area.drain) + (Qin_d-Qcap)
            try:
                Sss = res.V[compartments[5]] /(res.V[compartments[4]]*(res.Porosity[compartments[4]])) 
                Q10_exfds = 1/dt * ((1-Sss) * res.Porosity[compartments[4]]*res.V[compartments[4]])
            except AttributeError:#If native soil not given, ignore it.
                Q10_exfds = Q10_exfus
            Q10_exf = max(min(Q10_exfp,Q10_exfus,Q10_exfds),0)
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
                for layer in compartments[2:4]:#For simplicity assume height of water column is depth * FrnWat 
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
            res.loc[compartments[0],'Frnair'] = res.Porosity[compartments[0]]- res.FrnWat[compartments[0]]/res.Porosity[compartments[0]]
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
        #pdb.set_trace()
        try:
            if (params.val.hydrology_structure == 'Tree_Trench') | (params.val.hydrology_structure == 'TT'):
               hydro_struc = [['topsoil','topsoil_pores'],['subsoil','water'],['drain','drain_pores']]
            elif (params.val.hydrology_structure == 'Bioretention') | (params.val.hydrology_structure == 'BC'):
               hydro_struc = [['pond'],['filter','water'],['drain','drain_pores']]
            else:#User defined (not working, just hard-coded for now)
               hydro_struc = [['pond'],['filter','water'],['drain','drain_pores'],['native_soil','native_pores']]
        except AttributeError:#Default is a bioretention system if no parameters are given
           hydro_struc = [['pond'],['filter','water'],['drain','drain_pores']]   
        #This is a bioretention system
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
                res.loc['filter','Porosity'] = res.Porosity['filter']*params.val.thetam #Effective porosity - the rest is the immobile water fraction
                res.loc['water','V'] = res.V['filter']*res.FrnWat['filter'] #water fraction in subsoil - note this is different than saturation
                res.loc['drain_pores','V'] = res.V.drain*res.FrnWat.drain
                res.loc[:,'P'] = 2 * (res.Area/res.Depth + res.Depth) #Perimeter, m ## Need to make into hydraulic perimeter##
                res.loc[np.isnan(res.loc[:,'P']),'P'] = 0 #If the compartment doesn't exist
                res.loc[np.isinf(res.loc[:,'P']),'P'] = 0
                res.loc['filter','Discrete'] = 1.
                if 'native_soil' in hydro_struc[len(hydro_struc)-1]:
                    res.loc['native_pores','V'] = res.V.native_soil*res.FrnWat.native_soil
            #Then, do pond calculations
            res,Q26 = pondzone(inflow,rainrate,dt,params,res,['pond','filter','water'])
            #Pond flows to filter zone. Qin_f = Q26
            res,Qinf_f,Qcap = filterzone(Q26,dt,params,res,['filter','water','drain_pores','pond'])

            #Native soil if present, modeled as filter zone. Qin_n = Q_exf
            if 'native_soil' in hydro_struc[len(hydro_struc)-1]:
                #Drainage zone. Qin_d = Qinf_f
                res = drainzone(Qinf_f,Qcap,dt,params,res,
                                ['drain','drain_pores','filter','pond','native_soil','native_pores'])
                Qin_n = res.loc['drain','Q_exf']
                res,Qinf_n,Qcap_n = filterzone(Qin_n,dt,params,res,['native_soil','native_pores',None,'drain'])
            else:
                res = drainzone(Qinf_f,Qcap,dt,params,res,['drain','drain_pores','filter','pond'])
            #Need to set some values for subsoil and topsoil manually
            res.loc['subsoil','FrnWat'] = res.loc['filter','FrnWat']
            res.loc['topsoil','FrnWat'] = res.loc['filter','FrnWat']
            res.loc['subsoil','Frnair'] = res.loc['filter','Frnair']
            res.loc['topsoil','Frnair'] = res.loc['filter','Frnair']
        #compartments that are used. For Kortright ['drain','drain_pores','filter','pond']
        #for tree trench ['drain','drain_pores','subsoil','topsoil']
        elif 'topsoil_pores' in hydro_struc[len(hydro_struc)-1]:
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
            #Hardcoding switch in Kn after tracer event. If Kn2 is in the params it will switch
            if timeseries.time[t] > 7.:
                try:
                    params.loc['Kn','val'] = params.val.Kn2
                except AttributeError:
                    pass
            #Update some parameters based on timeseries
            try:
                params.loc['Emax','val'] = timeseries.loc[t,'Max_ET']
            except KeyError:
                pass
            #if t == 241:#216:#Break at specific timing 216 = first inflow
            if timeseries.time[t] == 0.0:
                #pdb.set_trace()
                yomama = 'great'
            res = self.bc_dims(res,inflow,rainrate,dt,params)
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
        #Probably a better way to add these, but this should work for now
        for key, value in kwargs.items():
            if key == 'ylim':
                ax.set_ylim(value)
            elif key == 'xlim':
                ax.set_xlim(value)
                
        ax.legend(labels=compartments)            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_xlim(xlim)
        ax.tick_params(axis='both', which='major', labelsize=15)  
        return fig

    def plot_Couts(self,res,Couts=None,Cmeas=None,compounds=None,multfactor = 1):
        '''
        Cmeas should be a dataframe with timeseries measurements
        '''
        #pdb.set_trace()
        ylabel = 'Effluent Concentration (mg/L)'
        xlabel = 'Time'
        #Define what will be plotted - all chemicals
        pltnames = []
        pltnames.append('time')
        numchems = 0
        if compounds == None:
            for chem in res.index.levels[0].values:
                pltnames.append(chem+'_Coutest') 
                numchems+=1
        else:
            for chem in compounds:
                pltnames.append(chem+'_Coutest')  
                numchems+=1
            #pltnames = pltnames.append()
        pltdata = Couts[pltnames]
        #if Cmeas == None:
        #    pass
        #else:
        #   for col in Cmeas.columns:
        #        pltdata.loc[:,str(col)] = Cmeas[col]
                
        #pltdata = pltdata.melt('time',var_name = 'Test_vs_est',value_name = 'Cout (mg/L)')
        #Set up plot
        ylim = [0, 1000]
        ylabel = 'Cout (mg/L)'
        xlabel = 'Time'
        #pltdata = res_time #All times at once
        fig,axs = plt.subplots(int(np.ceil(numchems/3)),3,figsize=(15,8),dpi=300)
        for ind,ax in enumerate(axs.reshape(-1)): 
            try:
                pltdf = Couts[pltnames[ind+1]]
                sns.lineplot(x = pltdata.time, y = pltdf*multfactor,ax=ax)
                ax.set_ylim(ylim)
            except IndexError:
                pass
        #ax[0].set_ylim(ylim)
        #ax.set_ylabel(ylabel, fontsize=20)
        #ax.set_xlabel(xlabel, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        fig.suptitle('Cout (mg/L x'+str(multfactor)+')');
        fpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Manuscript/Figs/Pythonfigs'
        fig.savefig(fpath + 'simstorm1.pdf',format = 'pdf')
        return fig
        
    def calibrate_flows(self,timeseries,paramnames,param0s,bounds=((0.0,1.0),(0.0,1.0),(0.0,1.0))):
        '''
        Calibrate flows based on measured effluent in the "timeseries"
        file and a test parameter name (string) and initial value (param0)
        '''
        #numc = self.numc
        #pdb.set_trace()
        #Define the optimization function
        def optBC_flow(param,paramname):
            #No negative test values
            if (param<0).sum() > 0:
                obj = 999
            else:
                #Set up and run with the test value, defined by the optimize function as "param"
                paramtest = self.params
                locsummtest = self.locsumm
                for ind, paramname in enumerate(paramnames):
                    paramtest.loc[paramname,'val'] = param[ind]
                    #For depth of native soil, need to change locsumm
                    if paramname == 'native_depth':
                        locsummtest.loc['native_soil','Depth'] = paramtest.loc[paramname,'val']
                flowtest = self.flow_time(locsummtest,paramtest,['water','subsoil'],timeseries)
                timeseries.loc[:,'Q_drainout'] = np.array(flowtest.loc[(slice(None),'drain'),'Q_todrain'])
                #Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
                eff = hydroeval.evaluator(kge, np.array(timeseries.loc[timeseries.time>0,'Q_drainout']),\
                                      np.array(timeseries.loc[timeseries.time>0,'Qout_meas']))
                #An efficiency of 1 is ideal, therefore we want to see how far it is from 1
                obj = (1-eff[0])[0]
            return obj
        
        
        #ins = [param0]
        #bnds = ((0.0,1.0),(0.0,1.0),(0.0,1.0))
        #bnds = ((0.0,1.0),(0.0,1.0))#,(0.0,1.0))
        res = minimize(optBC_flow,param0s,args=(paramnames,),bounds=bounds,method='L-BFGS-B',options={'disp': True})
        #res = minimize(optBC_flow,param0s,args=(paramnames,),bounds=bnds,method='SLSQP',options={'disp': True})
        #res = minimize(optBC_flow,param0s,args=(paramnames,),bounds=bnds,method='nelder-mead',options={'xtol': 1e-3, 'disp': True})
        return res
        
    def calibrate_tracer(self,timeseries,paramnames,param0s,bounds,tolerance=1e-5,flows = None,):
        '''
        Calibrate based on measured effluent concentration in the "timeseries"
        file and a test parameter name (string) and initial value (param0)
        flows should be a string with the filepath of a flow pickle or None
        '''
        #numc = self.numc
        #pdb.set_trace()
        #Define the optimization function
        def optBC_tracer(param,paramname,flows):
            #No negative test values
            if (param<0).sum() > 0:
                obj = 999
            else:
                #Set up and run with the test value, defined by the optimize function as "param"
                paramtest = self.params
                locsummtest = self.locsumm
                for ind, paramname in enumerate(paramnames):
                    paramtest.loc[paramname,'val'] = param[ind]
                    #For depth of native soil, need to change locsumm
                    if paramname == 'native_depth':
                        locsummtest.loc['native_soil','Depth'] = paramtest.loc[paramname,'val']
                #flowtest = self.flow_time(locsummtest,paramtest,['water','subsoil'],timeseries)
                #input_calcs = self.input_calc(locsummtest,self.chemsumm,paramtest,self.pp,self.numc,timeseries,flow_time=flowtest)
                if flows == None:#Run flows if no flow file given
                    res = self.flow_time(locsummtest,paramtest,['water','subsoil'],timeseries)
                    mask = timeseries.time>=0
                    minslice = np.min(np.where(mask))
                    maxslice = np.max(np.where(mask))#minslice + 5 #
                    res = df_sliced_index(res.loc[(slice(minslice,maxslice),slice(None)),:])    
                else:#If there is a flow file, run with that.
                    res = pd.read_pickle(flows)
                res = self.input_calc(locsummtest,self.chemsumm,paramtest,self.pp,self.numc,timeseries,flow_time=res)
                res = self.run_it(locsummtest,self.chemsumm,paramtest,self.pp,self.numc,timeseries,input_calcs=res)                
                mass_flux = self.mass_flux(res,self.numc) 
                Couts = self.conc_out(self.numc,timeseries,self.chemsumm,res,mass_flux)
                KGE = []
                for ind,chem in enumerate(self.chemsumm.index):
                    KGE.append((hydroeval.evaluator(kge, np.array(Couts.loc[:,chem+'_Coutest']),\
                                      np.array(Couts.loc[:,chem+'_Coutmeas'])))[0])
                #timeseries.loc[:,'Q_drainout'] = np.array(flowtest.loc[(slice(None),'drain'),'Q_todrain'])
                #Kling-Gupta Efficiency (modified Nash-Sutcliffe) can be our measure of model performance
                #If multiple compounds given, this just takes the average.
                eff = np.mean(KGE)
                #An efficiency of 1 is ideal, therefore we want to see how far it is from 1
                obj = (1-eff)
                print(obj,param)
            return obj
        
        
        #ins = [param0]
        #bnds = ((0.0,1.0),(0.0,1.0),(0.0,10000))
        #bnds = ((0.0,1.0),(0.0,1.0))#,(0.0,1.0))
        res = minimize(optBC_tracer,param0s,args=(paramnames,flows),bounds=bounds,method='L-BFGS-B',tol=tolerance,options={'disp': True,'maxfun':100})
        #res = minimize(optBC_tracer,param0s,args=(paramnames,flows),bounds=bounds,method='SLSQP',options={'disp': True})
        #res = minimize(optBC_tracer,param0s,args=(paramnames,flows),bounds=bounds,method='nelder-mead',options={'xtol': 1e-3, 'disp': True})
        return res 

    def plot_idfs(self,pltdata,pltvars=['pct_stormsewer','LogD','LogI'],cmap=None,
                  pltvals=False,savefig=False,figpath=None,interplims = [0.,1.],figsize=(10,6.7),
                  xlims=[-0.75,1.380211],ylims=None,vlims=[0.15,0.85],levels=15,xticks=None,yticks=None):
        '''
        pltdata needs to contain the defined pltvars.
        pltvars gives the contour variable (pltvar) and the x and y variables (in that order)
        cmap defines the colormap which will shade the contours defined by pltvar.
            By default goes from browns to blues
        pltvals defines whether the values of pltvar will be annoted to the figure
        
        '''
        #Define the colormap if not given
        if cmap == None:
            cmap = sns.diverging_palette(30, 250, l=40,s=80,center="light", as_cmap=True)#sep=1,
        
        #Only the variables we care about
        df = pltdata.loc[:,pltvars]
        #pltdata = pltdata.loc[:,[pltvar,'LogD','Intensity']]
        fig, ax = plt.subplots(1,1,figsize = figsize,dpi=300)

        #pltvars = ['LogD','LogI']
        #pltvars = ["LogD",'Intensity']
        df = df.pivot(index=pltvars[2],columns=pltvars[1])[pltvars[0]]
        df = df.interpolate(axis=0,method='spline',order=2)
        #df = df.interpolate(axis=0,method='nearest')

        #df = df.interpolate(axis=0,method='quadratic')
        df[np.isnan(df)] = 0
        if interplims != None:
            df[df>interplims[1]] = interplims[1]
            df[df<interplims[0]] = interplims[0]

        #Soil-water differences
        pc = ax.contourf(df.columns,df.index,df.values,cmap=cmap,vmin=vlims[0],vmax=vlims[1],levels=levels)
        #Other stuff
        #pc = ax.contourf(df.columns,df.index,df.values,cmap=cmap,sep=1)
        sns.lineplot(x=pltvars[1],y=pltvars[2],data=pltdata.reset_index(),hue='Frequency',
                     ax=ax,palette=sns.color_palette('Greys')[:len(pltdata.Frequency.unique())],legend=False)
        #Set values
        ax.set_xlim(xlims)
        if xticks != None:
            ax.set_xticks(xticks[0])
            ax.set_xticklabels(xticks[1])
        if yticks != None:
            ax.set_yticks(yticks[0])
            ax.set_yticklabels(yticks[1])
        if pltvals == True:
            pltdata = pltdata.reset_index()
            for ind in pltdata.index:
                ax.annotate(str(pltdata.loc[ind,pltvars[0]]),xy= (pltdata.loc[ind,pltvars[1]],pltdata.loc[ind,pltvars[2]]))
        fig.colorbar(pc)
        return fig,ax
        
        
        
        
        
        
      