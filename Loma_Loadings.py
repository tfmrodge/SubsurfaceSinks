# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:38:57 2019

@author: Tim Rodgers
"""
from FugModel import FugModel #Import the parent FugModel class
from Subsurface_Sinks import SubsurfaceSinks
from HelperFuncs import vant_conv, arr_conv #Import helper functions
from scipy.integrate import solve_ivp
from ode_helpers import state_plotter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
#import time
import pdb #Turn on for error checking

class LomaLoadings(SubsurfaceSinks):
    """Wastewater treatment wetland implementation of the Subsurface_Sinks model.
    Created for the Oro Loma Horizontal Levee, hence the name. This model represents
    a horizontally flowing, planted wetland. It is intended to be solved as a Level V 
    1D ADRE, across space and time, although it can be modified to make a Level III or
    Level IV multimedia model
        
    Attributes:
    ----------
            
            locsumm (df): physical properties of the system
            chemsumm (df): physical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            timeseries (df): Timeseries values providing the time-dependent inputs
            to the model.
            num_compartments (int): (optional) number of non-equilibirum 
            compartments
            name (str): (optional) name of the BC model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,locsumm,chemsumm,params,timeseries,num_compartments = 8,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc) 
        
    def make_system(self,locsumm,params,numc,timeseries,dx = None):
        #This function will build the dimensions of the 1D system based on the "locsumm" input file.
        #If you want to specify more things you can can just skip this and input a dataframe directly
        L = locsumm.Length.water
        params.loc['L','val'] = L
        #dx = 0.1
        if dx == None:
            dx = params.val.dx
        #Smaller cells at influent - testing turn on/off
        #pdb.set_trace()
        samegrid = True
        if samegrid == True:
            res = pd.DataFrame(np.arange(0.0+dx/2.0,L,dx),columns = ['x'])
            res.loc[:,'dx'] = dx
        else:
            dx_alpha = 0.5
            dx_in = params.val.dx/10
            res = pd.DataFrame(np.arange(0.0+dx_in/2.0,L/10.0,dx_in),columns = ['dx'])
            res = pd.DataFrame(np.arange(0.0+dx_in/2.0,L/10.0,dx_in),columns = ['x'])
            lenin = len(res) #Length of the dataframe at the inlet resolution
            res = res.append(pd.DataFrame(np.arange(res.iloc[-1,0]+dx_in/2+dx/2,L,dx),columns = ['x']))
            res = pd.DataFrame(np.array(res),columns = ['x'])
            res.loc[0:lenin-1,'dx'] = dx_in
            res.loc[lenin:,'dx'] = dx
        #pdb.set_trace()
        #Control volume length dx - x is in centre of each cell.
        res.iloc[-1,1] = res.iloc[-2,1]/2+L-res.iloc[-1,0]
        #Integer cell number is the index, columns are values, 'x' is the centre of each cell
        #res = pd.DataFrame(np.arange(0+dx/2,L,dx),columns = ['x'])
        res.loc[:,'dm'] = True
        res.loc[:,'porositywater'] = locsumm.Porosity['water'] #added so that porosity can vary with x
        res.loc[:,'porositysubsoil'] = locsumm.Porosity['subsoil'] #added so that porosity can vary with x
        res.loc[:,'porositytopsoil'] = locsumm.Porosity['topsoil']
        #Define the geometry of the Oro Loma system
        oro_x = np.array(params.val.sys_x.split(","),dtype='float') #x-coordinates of the Oro Loma design drawing taper and mixing wells
        #With Mixing Wells
        oro_dss = np.array(params.val.sys_ss.split(","),dtype='float') #Subsoil depths from the Oro Loma design drawing. Mixing wells are represented by 3' topsoil depths
        oro_dts = np.array(params.val.sys_ts.split(","),dtype='float') #Topsoil depths. Mixing wells do not have a topsoil layer
        #Asume minimum topsoil depth
        #oro_dts[oro_dts==0] = 0.01
        #Without Mixing Wells
        #For topsoil as just surface layer
        #oro_dss = [0.2548, 0.2548, 0.8644, 0.8644, 0.8644, 0.8644, 0.8644, 0.8644,0.8644, 0.8644, 0.8644, 0.712 , 0.712 , 0.712 ]
        #oro_dts = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
        #oro_dss = [0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048,0.3048] #Subsoil depths from the Oro Loma design drawing. Mixing wells are represented by 3' topsoil depths
        #oro_dts = [0.,0.,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.6096,0.4572,0.4572,0.4572]
        f_dss = interp1d(oro_x,oro_dss,'linear') #Function to define subsoil depth
        f_dts = interp1d(oro_x,oro_dts,'linear') #Function to define topsoil depth 
        res.loc[:,'depth_ss'] = f_dss(res.x)#Depth of the subsoil
        res.loc[:,'depth_ts'] = f_dts(res.x)#Depth of the topsoil
        #Include immobile phase water content, so that Vw is only mobile phase & V2 includes immobile phase
        #Cross sectional "flow area"
        res.loc[:,'Awater'] = locsumm.Width[0] * res.depth_ss * res.porositywater * params.val.thetam
        #Here we set as the area of the soil
        #pdb.set_trace()
        res.loc[:,'Asubsoil'] =  locsumm.Width[0] * res.depth_ss * \
        (res.porositysubsoil + res.porositywater*(1-params.val.thetam))

        #Then, we put the times in by copying res across them
        res_t = dict.fromkeys(timeseries.index,[]) 
        for t in timeseries.index:
            res_t[t] = res.copy(deep=True)
            res_t[t].loc[:,'time'] = timeseries.time[t]
        res = pd.concat(res_t)
        #Add the 'time' column to the res dataframe
        #res.loc[:,'time'] = timeseries.loc[:,'time'].reindex(res.index,method = 'bfill')
        #Next, flows.
        #Set up the water compartment
        Qslope = (timeseries.loc[:,'Qin'].reindex(res.index,level=0)\
                      - timeseries.loc[:,'Qout'].reindex(res.index,level=0))/L
        res.loc[:,'Qwater'] = timeseries.loc[:,'Qin'].reindex(res.index,level=0) - Qslope*res.x 
        res.loc[:,'Q1'] = res.Qwater
        #Flow into each cell
        res.loc[:,'Qin'] = res.loc[:,'Qwater'] + Qslope*(res.dx/2)
        res.loc[:,'Qout'] = res.loc[:,'Qwater'] - Qslope*(res.dx/2)
        res.loc[:,'Qet'] = res.Qin - res.Qout  #ET flow 
        #res.loc[(slice(None),0),'Qet'] = timeseries.loc[:,'Qin'].reindex(res.index,level=0) - res.loc[(slice(None),0),'Qet'] #Upstream boundary
        #Qet into the plants is all of it - this is used in the BC blues as different discretization.
        res.loc[res.dm,'Qetplant'] = res.loc[:,'Qet']
        res.loc[:,'Qetsubsoil'] = res.Qet*params.val.fet_ss
        res.loc[:,'Qettopsoil'] = res.Qet*params.val.fet_ts
        res.loc[:,'Qwaterexf'] = 0 #Amount of exfiltration from the system, for unlined systems
        res.loc[:,'qwater'] = res.Qwater/(locsumm.Depth[0] * locsumm.Width[0])  #darcy flux [L/T] at every x
        res.loc[:,'vwater'] = res.Qwater/res.Awater #velocity [L/T] at every x - velocity is eq
        res.loc[:,'v1'] = res.vwater
        
        #params.loc['vin','val'] = res.Qin/(res.Awater)
        #params.loc['vout','val'] = res.Qout/(res.Awater)
        #For the topsoil compartment there is a taper in the bottom 2/3 of the cell
        #res.loc[:,'depth_ts'] = 0.6096
        #res.loc[res.x<2.1336,'depth_ts'] = 0 #No topsoil compartment in the first 7 feet
        #res.loc[res.x>30.7848,'depth_ts'] = 0.6096-(res.x-30.7848)*(0.5/47) #Topsoil tapers in bottom third from 2' to 1.5'
        #res.loc[res.x>45.1104,'depth_ts'] = 0 #Bottom 2' is just gravel drain
        res.loc[:,'Atopsoil'] = locsumm.Width[0] * res.depth_ts
        
        #Now loop through the columns and set the values
        #pdb.set_trace()       
        for jind, j in enumerate(numc):
            #Area (A), Volume (V), Density (rho), organic fraction (foc), ionic strength (Ij)
            #water fraction (fwat), air fraction (fair), temperature (tempj), pH (phj)
            jind = jind+1 #The compartment number, for the advection term
            Aj, Vj, Vjind, rhoj, focj, Ij = 'A' + str(j), 'V' + str(j), 'V' + str(jind),'rho' + str(j),'foc' + str(j),'I' + str(j)
            fwatj, fairj, tempj, pHj = 'fwat' + str(j), 'fair' + str(j),'temp' + str(j), 'pH' + str(j)
            rhopartj, fpartj, advj = 'rhopart' + str(j),'fpart' + str(j),'adv' + str(j)
            #compartment = j
            if j in ['water', 'subsoil', 'topsoil']: #done above, water and subsoil as 0 and 1, topsoil as 3
                pass
            else: #Other compartments don't share the same CV
                res.loc[:,Aj] = locsumm.Width[j] * locsumm.Depth[j]
            res.loc[:,Vj] = res.loc[:,Aj] * res.dx #volume at each x [L³]
            res.loc[:,Vjind] = res.loc[:,Vj] #Needed for FugModel ABC
            res.loc[:,focj] = locsumm.FrnOC[j] #Fraction organic matter
            res.loc[:,Ij] = locsumm.cond[j]*1.6E-5 #Ionic strength from conductivity #Plants from Trapp (2000) = 0.5
            res.loc[:,fwatj] = locsumm.FrnWat[j] #Fraction water
            res.loc[:,fairj] = locsumm.FrnAir[j] #Fraction air
            res.loc[:,fpartj] = locsumm.FrnPart[j] #Fraction particles
            res.loc[:,pHj] = locsumm.pH[j] #pH
            res.loc[:,rhopartj] = locsumm.PartDensity[j] #Particle density
            res.loc[:,rhoj] = locsumm.Density[j] #density for every x [M/L³]
            res.loc[:,advj] = locsumm.Advection[j]              
            if j in ['water']: #Interpolate temperature from inlet and outlet temperatures
                Tslope = (timeseries.loc[:,'Twater_in']-timeseries.loc[:,'Twater_out'])/L
                res.loc[:,tempj] = (timeseries.loc[:,'Twater_in'].reindex(res.index,level=0)\
                - Tslope.reindex(res.index,level=0)*res.x) + 273.15
                #Interpolate the pH based on the observed pH Assume constant in time
                pHx = np.array(params.val.ph_x.split(","),dtype='float')
                pHy = np.array(params.val.pH_interp.split(","),dtype='float')
                res.loc[:,pHj] = np.interp(res.x,pHx,pHy, left = pHy[0], right = pHy[-1])
            elif j in ['subsoil','topsoil','rootbody','rootxylem','rootcyl']: #Assume temperature at equilibrium with water
                res.loc[:,tempj] = res.loc[:,'tempwater']
            elif j in ['air']: 
                res.loc[:,tempj] = timeseries.loc[:,'Tair'].reindex(res.index,level=0) + 273.15 #Temperature [K]
                res.loc[:,rhoj] = 0.029 * 101325 / (params.val.R * res.loc[:,tempj])
                res.loc[:,advj] = timeseries.WindSpeed.reindex(res.index,level=0)*locsumm.Depth.air*36000
            else: #Assume shoots at air temp
                res.loc[:,tempj] = timeseries.loc[:,'Tair'].reindex(res.index,level=0) + 273.15 #Temperature [K]
        #Root volumes & area based off of soil volume fraction
        res.loc[:,'Vroot'] = params.val.VFroot*locsumm.Width['water']*res.dx #Total root volume per m² ground area
        res.loc[:,'Aroot'] = params.val.Aroot*locsumm.Width['water']*res.dx #Need to define how much is in each section top and sub soil
        
        #For area of roots in contact with sub and topsoil assume that roots in both zones are roughly cylindrical
        #with the same radius. SA = pi r² 
        res.loc[:,'Arootsubsoil'] = (1-params.val.froot_top) * params.val.Aroot*locsumm.Width['water']*res.dx #Area of roots in direct contact with subsoil
        res.loc[:,'Aroottopsoil'] = params.val.froot_top * params.val.Aroot*locsumm.Width['water']*res.dx #Area of roots in contact with topsoil
        res.loc[:,'AsoilV'] = locsumm.Width['water']*res.dx #Vertical direction area of the topsoil compartment
        res.loc[:,'Asoilair'] =  res.loc[:,'AsoilV']
        #Shoot area based off of leaf area index (LAI) 
        res.loc[:,'A_shootair'] = params.val.LAI*res.dx*locsumm.Width['water']
        #res.loc[res.depth_ts==0,'AsoilV'] = 0
        res.loc[res.depth_ts==0,'Aroottopsoil'] = 0
        #Roots are broken into the body, the xylem and the central cylinder.
        res.loc[:,'Vrootbody'] = (params.val.VFrootbody+params.val.VFapoplast)*res.Vroot #Main body consists of apoplast and cytoplasm
        res.loc[:,'Vrootxylem'] = params.val.VFrootxylem*res.Vroot #Xylem
        res.loc[:,'Vrootcyl'] = params.val.VFrootcylinder*res.Vroot #Central cylinder
        #Now, change area of subsoil to be water/soil interface
        res.loc[:,'Asubsoil'] = res.loc[:,'Asubsoil']*params.val.AF_soil
        #And change the water fraction to reflect thetam
        res.loc[:,'fwatsubsoil'] = res.porositywater*(1-params.val.thetam)/\
            (res.porositysubsoil + res.porositywater*(1-params.val.thetam))

        #Longitudinal Dispersivity. Calculate using relationship from Schulze-Makuch (2005) 
        #for unconsolidated sediment unless a value of alpha [L] is given
        if 'alpha' not in params.index:
            params.loc['alpha','val'] = 0.2 * L**0.44 #alpha = c(L)^m, c = 0.2 m = 0.44
        res.loc[:,'ldisp'] = params.val.alpha * res.v1 #Resulting Ldisp is in [L²/T]
        #Replace nans with 0s for the next step
        res = res.fillna(0)
        return res
    
    def LL_fig(self,numc,mass_balance,time=None,compound=None,figname='Oro_Model_Figure.tif',dpi=100,fontsize=8,figheight=6,dM_locs=None,M_locs=None):
            """ 
            Show modeled fluxes and mass distributions on a bioretention cell figure. 
            Just a wrapper to give the correct figure to the main function
            Attributes:
                mass_balance = Either the cumulative or the instantaneous mass balance, as output by the appropriate functions, normalized or not
                figname (str, optional) = name of figure on which to display outputs. Default figure should be in the same folder
                time (float, optional) = Time to display, in hours. Default will be the last timestep.
                compounds (str, optional) = Compounds to display. Default is all.
            """
            #pdb.set_trace()
            try:
                dM_locs[1]
            except TypeError:
                #Define locations of the annotations if not give    
                dM_locs = {'Meff':(0.835,0.170),'Min':(0.11,0.40),'Mrwater':(0.25,0.18),
                          'Mrsubsoil':(0.83,0.232),'Mrtopsoil':(0.82,0.37),'Mrrootbody':(0.60,0.745),'Mrrootxylem':(0.73,0.70),
                          'Mrrootcyl':(0.82,0.745),'Mrshoots':(0.135,0.760),'Mrair':(0.025,0.830),
                          'Mnetwatersubsoil':(0.28,0.22),'Mnetwatertopsoil':(0.2,0.35),'Mnetsubsoiltopsoil':(0.01,0.36),
                          'Mnetsubsoilrootbody':(0.63,0.52),'Mnettopsoilrootbody':(0.65,0.58),'Mnettopsoilshoots':(0.25,0.47),
                          'Mnettopsoilair':(0.029,0.50),'Mnetrootbodyrootxylem':(0.82,0.534),'Mnetrootxylemrootcyl':(0.826,0.584),
                          'Mnetrootcylshoots':(0.513,0.451),'Mnetshootsair':(0.06,0.601),'Madvair':(0.675,0.975)}
                #Location of the massess
                M_locs = {'Mwater':(0.61,0.18),'Msubsoil':(0.61,0.22),'Mtopsoil':(0.11,0.40),'Mrootbody':(0.76,0.89),
                        'Mrootxylem':(0.76,0.845),'Mrootcyl':(0.76,0.80),'Mshoots':(0.76,0.935),
                        'Mair':(0.11,0.965)}             
            fig,ax = self.model_fig(numc,mass_balance=mass_balance,dM_locs=dM_locs,
                                    M_locs=M_locs,time=time,compound=compound,
                                    figname=figname,dpi=dpi,fontsize=fontsize,
                                    figheight=figheight,)
            return fig,ax    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    