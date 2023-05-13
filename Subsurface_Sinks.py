# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:52:42 2018

@author: Tim Rodgers
"""
from FugModel import FugModel #Import the parent FugModel class
from HelperFuncs import ppLFER, vant_conv, arr_conv, make_ppLFER, find_nearest #Import helper functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb #Turn on for error checking

class SubsurfaceSinks(FugModel):
    """ Model of 1D contaminant transport in a vegetated, flowing water system.
    This is the main class that will be run to solve the contaminant transport
    and fate through a vegetated subsurface system. Code here can be used for
    the BCBlues submodel, which is parameterized to represent a vertically-flowing
    bioretention cell.
    SubsurfaceSinks objects have the following properties:
        
    Attributes:
    ----------
            
            locsumm (df): physical properties of the systmem
            chemsumm (df): physical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            results (df): Results of the model
            num_compartments (int): (optional) number of non-equilibirum 
            compartments
            name (str): (optional) name of the BC model 
            pplfer_system (df): (optional) input ppLFERs to use in the model
    """
    
    def __init__(self,locsumm,chemsumm,params,num_compartments = 8,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        #self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp,self.numc)
                    
    def make_chems(self,chemsumm,pp):
        """This code will calculate partition coefficients 
        from ppLFERs in the "HelperFuncs.py" module. 
        All chemical specific information that doesn't vary with 
        space should be in this method
    
        Attributes:
        ----------
                chemsumm (df): physical-chemical properties of modelled compounds
                params (df): Other parameters of the model
                results (df): Results of the model
                num_compartments (int): (optional) number of non-equilibirum 
                compartments
                name (str): (optional) name of the BC model 
                pplfer_system (df): (optional) input ppLFERs to use in the model
        """
        #Copy the chemsumm dataframe, define the ideal gas constant
        res = chemsumm.copy(deep=True)
        R = 8.3144598
        
        #ppLFER system parameters - initialize defaults if not there already
        if pp is None:
            pp = pd.DataFrame(index = ['l','s','a','b','v','c'])
            pp = make_ppLFER(pp)
        
        #Check if partition coefficients & dU values have been provided, or only solute descriptors
        #add based on ppLFER if not, then adjust partition coefficients to 298.15K if they aren't already.
        #The main code assumes that all provided partition coefficients
        #at 298K.
        #Aerosol-Air (Kqa), use octanol-air enthalpy
        if 'LogKqa' not in res.columns:
            res.loc[:,'LogKqa'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKqa.l,pp.logKqa.s,pp.logKqa.a,pp.logKqa.b,pp.logKqa.v,pp.logKqa.c)
        if 'dUoa' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUoa'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUoa.l,pp.dUoa.s,pp.dUoa.a,pp.dUoa.b,pp.dUoa.v,pp.dUoa.c)
        res.loc[:,'LogKqa'] = np.log10(vant_conv(res.dUoa,298.15,10.**res.LogKqa,T1 = 288.15))
        #Organic carbon-water (KocW), use octanol-water enthalpy (dUow)
        if 'LogKocW' not in res.columns:
            res.loc[:,'LogKocW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKocW.l,pp.logKocW.s,pp.logKocW.a,pp.logKocW.b,pp.logKocW.v,pp.logKocW.c)
        if 'dUow' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUow'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUow.l,pp.dUow.s,pp.dUow.a,pp.dUow.b,pp.dUow.v,pp.dUow.c)
        #Storage Lipid Water (KslW), use ppLFER for dUslW (kJ/mol) convert to J/mol/K
        if 'LogKslW' not in res.columns:
            res.loc[:,'LogKslW'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKslW.l,pp.logKslW.s,pp.logKslW.a,pp.logKslW.b,pp.logKslW.v,pp.logKslW.c)
        if 'dUslW' not in res.columns:
            res.loc[:,'dUslW'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUslW.l,pp.dUslW.s,pp.dUslW.a,pp.dUslW.b,pp.dUslW.v,pp.dUslW.c)
        res.loc[:,'LogKslW'] = np.log10(vant_conv(res.dUslW,298.15,10.**res.LogKslW,T1 = 310.15))
        #Air-Water (Kaw) use dUaw
        if 'LogKaw' not in res.columns:
            res.loc[:,'LogKaw'] = ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.logKaw.l,pp.logKaw.s,pp.logKaw.a,pp.logKaw.b,pp.logKaw.v,pp.logKaw.c)
        if 'dUaw' not in res.columns: #!!!This might be broken - need to check units & sign!!!
            res.loc[:,'dUaw'] = 1000 * ppLFER(res.L,res.S,
            res.A,res.B,res.V,pp.dUaw.l,pp.dUaw.s,pp.dUaw.a,pp.dUaw.b,pp.dUaw.v,pp.dUaw.c)

        #Define storage lipid-air (KslA) and organic carbon-air (KocA) using the thermodynamic cycle
        res.loc[:,'LogKslA'] = np.log10(10.0**res.LogKslW / 10.0**res.LogKaw)
        res.loc[:,'LogKocA'] = np.log10(10.0**res.LogKocW / 10.0**res.LogKaw)
        #Calculate Henry's law constant (H, Pa m³/mol) at 298.15K
        res.loc[:,'H'] = 10.**res.LogKaw * R * 298.15
        return res

    def sys_chem(self,locsumm,chemsumm,params,pp,numc,timeseries,flow_time = None):
        """Put together the system and the chemical parameters into the 3D dataframe
        that will be used to calculate Z and D values.
        Attributes:
        ----------
                
                locsumm (df): physical properties of the systmem
                chemsumm (df): physical-chemical properties of modelled compounds
                params (df): Other parameters of the model
                pp (df): (optional) input ppLFERs to use in the model
                numc (str): (optional) List of compartments
                timeseries (df): dataframe with the time-series of the model values.
        """        
        #Set up the output dataframe, res, a multi indexed pandas dataframe with the 
        #index level 0 as the chemical names, 1 as the integer cell number along x
        #First, call make_system
        pdb.set_trace
        res = self.make_system(locsumm,params,numc,timeseries,params.val.dx,flow_time)
        #Then, fill out the chemsumm file
        chemsumm = self.make_chems(chemsumm,pp)
        #add the chemicals as level 0 of the multi index
        chems = chemsumm.index
        numchems = len(chems)
        resi = dict.fromkeys(chems,[])
        #Using the chems as the keys of the dict(resi) then concatenate
        for i in range(numchems):
            resi[chems[i]] = res.copy(deep=True)
        res = pd.concat(resi)
        
        #Parameters that vary by chem and x
        #Add a dummy variable
        res.loc[:,'dummy'] = 1
        #Read dU values in for temperature conversions. 
        res.loc[:,'dUoa'] = res['dummy'].mul(chemsumm.dUoa,level = 0)
        res.loc[:,'dUow'] = res['dummy'].mul(chemsumm.dUow,level = 0)
        res.loc[:,'dUslw'] = res['dummy'].mul(chemsumm.dUslW,level = 0)
        res.loc[:,'dUaw'] = res['dummy'].mul(chemsumm.dUaw,level = 0)
        #Equilibrium constants  - call by compartment name
        #Calculate temperature-corrected media reaction rates (/h)
        #These can vary in x, although this makes the dataframe larger. 
        for j in numc:
            Kdj, Kdij, focj, tempj = 'Kd' +str(j),'Kdi' +str(j),'foc' +str(j),'temp' +str(j)
            Kawj, rrxnj = 'Kaw' +str(j),'rrxn' +str(j)
            #Kaw is only neutral
            res.loc[:,Kawj] = vant_conv(res.dUaw,res.loc[:,tempj],res['dummy'].mul(10.**chemsumm.LogKaw,level = 0))
            #Kd neutral and ionic
            if j in ['water','subsoil','topsoil','pond','drain']: #for water, subsoil and topsoil if not directly input
                if 'Kd' in chemsumm.columns: #If none given leave blank, will estimate from Koc & foc
                    res.loc[:,Kdj] = res['dummy'].mul(chemsumm.Kd,level = 0)
                    maskn = np.isnan(res.loc[:,Kdj])
                else:
                    maskn = res.dummy==1
                if 'Kdi' in chemsumm.columns:#If none given leave blank, will estimate from Koc & foc
                    res.loc[:,Kdij] = res['dummy'].mul(chemsumm.Kdi,level = 0)
                    maski = np.isnan(res.loc[:,Kdij])
                else:
                    maski = res.dummy == 1
                #pdb.set_trace()
                res.loc[maskn,Kdj] = res.loc[:,focj].mul(10.**chemsumm.LogKocW, level = 0)
                res.loc[maski,Kdij] = res.loc[:,focj].mul(10.**(chemsumm.LogKocW-3.5), level = 0) #3.5 log units lower from Franco & Trapp (2010)
                #20230413 - Adding amendments. We are going to assume linear addition of amendment
                #with organic matter so that Kd(overall) (L/kg) = Koc*foc+Kdam*fam. We are also going to assume that
                #the Kd value given is the same for neutral and ionic (if ionizable)
                if j in ['subsoil']:
                    try:
                        res.loc[maskn,Kdj] += res['dummy'].mul((10.**chemsumm.LogKdam)*params.loc['famendment','val'], level = 0)
                        res.loc[maski,Kdij] += res['dummy'].mul((10.**chemsumm.LogKdam)*params.loc['famendment','val'], level = 0)
                    except AttributeError: 
                        pass
                res.loc[:,Kdj] = vant_conv(res.dUow,res.loc[:,tempj],res.loc[:,Kdj]) #Convert with dUow
                #The ionic Kd value is based off of pKa and neutral Kow
                res.loc[:,Kdij] = vant_conv(res.dUow,res.loc[:,tempj],res.loc[:,Kdij])
                if j in ['water','pond','drain']:
                    rrxnq_j = 'rrxnq_'+str(j)
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/chemsumm.WatHL,level = 0)
                    res.loc[:,rrxnj] = arr_conv(params.val.Ea,res.loc[:,tempj],res.loc[:,rrxnj])
                    res.loc[:,rrxnq_j] = res.loc[:,rrxnj] * 0.1 #Can change particle bound rrxn if need be
                if j in ['subsoil','topsoil']:
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/chemsumm.SoilHL,level = 0)
                    res.loc[:,rrxnj] = arr_conv(params.val.Ea,res.loc[:,tempj],res.loc[:,rrxnj])
            elif j in ['shoots','rootbody','rootxylem','rootcyl']:
                res.loc[maskn,Kdj] = vant_conv(res.dUslw,res.loc[:,tempj],res.loc[:,focj].mul(10.**chemsumm.LogKslW, level = 0))
                res.loc[maski,Kdij] = res.loc[maski,Kdj]
                if 'VegHL' in chemsumm.columns:
                    chemsumm.loc[np.isnan(chemsumm.VegHL),'VegHL'] = chemsumm.WatHL*0.1
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/chemsumm.VegHL,level = 0)
                else:#If no HL for vegetation specified, assume 0.1 * wat HL - based on Wan (2017) wheat plants?
                    res.loc[:,rrxnj] = res['dummy'].mul(np.log(2)/(chemsumm.WatHL*0.1),level = 0)
                res.loc[:,rrxnj] = arr_conv(params.val.Ea,res.loc[:,tempj],res.loc[:,rrxnj])
                if j in ['shoots']:
                    #Mass transfer coefficients (MTC) [l]/[T]
                    #Chemical but not location specific mass transport values
                    #Membrane neutral and ionic mass transfer coefficients, Trapp 2000
                    res.loc[:,'kmvn'] = 10.**(1.2*res['dummy'].mul(chemsumm.LogKow, level = 0) - 7.5) * 3600 #Convert from m/s to m/h
                    res.loc[:,'kmvi'] = 10.**(1.2*(res['dummy'].mul(chemsumm.LogKow, level = 0) -3.5) - 7.5)* 3600 #Convert from m/s to m/h
                    res.loc[:,'kspn'] = 1/(1/params.val.kcw + 1/res.kmvn) #Neutral MTC between soil and plant. Assuming that there is a typo in Trapp (2000)
                    res.loc[:,'kspi'] = 1/(1/params.val.kcw + 1/res.kmvi)
                    #Correct for kmin = 10E-10 m/s for ions
                    kspimin = (10e-10)*3600
                    res.loc[res.kspi<kspimin,'kspi'] = kspimin
                    #Air side MTC for veg (from Diamond 2001)
                    #Back calculate windspeed in m/s. As the delta_blv requires a windspeed to be calculated, then replace with minwindspeed
                    try:
                        windspeed = res.loc[:,'advair']/np.array(np.sqrt(locsumm.Area.air)*locsumm.Depth.air*3600)
                    except AttributeError:
                        windspeed = timeseries.WindSpeed
                    windspeed.loc[windspeed==0] = params.val.MinWindSpeed #This will also replace all other compartments
                    res.loc[:,'delta_blv'] = 0.004 * ((0.07 / windspeed.reindex(res.index,level = 1)) ** 0.5) 
                    res.loc[:,'AirDiffCoeff'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)
                    res.loc[:,'kav'] = res.AirDiffCoeff/res.delta_blv #m/h
                    #Veg side (veg-air) MTC from Trapp (2007). Consists of stomata and cuticles in parallel
                    #Stomata - First need to calculate saturation concentration of water
                    C_h2o = (610.7*10.**(7.5*(res.tempshoots-273.15)/(res.tempshoots-36.15)))/(461.9*res.tempshoots)
                    g_h2o = res.Qet/(res.A_shootair*(C_h2o-params.val.RH/100*C_h2o)) #MTC for water
                    g_s = g_h2o*np.sqrt(18)/np.sqrt(res['dummy'].mul(chemsumm.MolMass, level = 0))
                    res.loc[:,'kst'] = g_s * res['dummy'].mul((10.**chemsumm.LogKaw), level = 0) #MTC of stomata [L/T] (defined by Qet so m/h)
                    #Cuticle
                    res.loc[:,'kcut'] = 10.**(0.704*res['dummy'].mul((chemsumm.LogKow), level = 0)-11.2)*3600 #m/h
                    res.loc[:,'kcuta'] = 1/(1/res.kcut + 1*res['dummy'].mul((10.**chemsumm.LogKaw), level = 0)/(res.kav)) #m/h
                    res.loc[:,'kvv'] = res.kcuta+res.kst #m/h
            elif j in ['air']:
                res.loc[maskn,Kdj] = vant_conv(res.dUoa,res.loc[:,tempj],res.loc[:,focj].mul(10.**chemsumm.LogKqa, level = 0))
                res.loc[maski,Kdij] = res.loc[maski,Kdj]
                res.loc[:,rrxnj] = res['dummy'].mul(chemsumm.AirOHRateConst, level = 0)  * params.val.OHConc
                res.loc[:,rrxnj] = arr_conv(params.val.EaAir,res.loc[:,tempj],res.loc[:,rrxnj])
                if 'AirQOHRateConst' not in res.columns:
                    res.loc[:,'rrxnq_air'] = 0.1 * res.loc[:,rrxnj]
                else:
                    res.loc[:,'rrxnq_air'] = res['dummy'].mul(chemsumm.AirOHRateConst*0.1, level = 0)*params.val.OHConc    
                    res.loc[:,'rrxnq_air'] = arr_conv(params.val.EaAir,res.loc[:,tempj],res.airq_rrxn)    
        #Then the final transport parameters
        #Deff = 1/tortuosity^2, tortuosity(j)^2 = 1-2.02*ln(porosity) (Shen and Chen, 2007)
        #the mask dm is used to differentiate compartments that are discretized vs those that are not
        res.loc[res.dm,'tausq_water'] = 1/(1-2.02*np.log(res.porositywater))
        res.loc[res.dm,'Deff_water'] = res['tausq_water'].mul(chemsumm.WatDiffCoeff, level = 0) #Effective water diffusion coefficient 
        #Subsoil
        res.loc[res.dm,'Deff_subsoil'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*res.porositywater**2\
        /((1-res.porositywater)*res.rhosubsoil+res.porositywater)
        if 'pond' in numc: #Add pond for BC model
            res.loc[res.dm==False,'Deff_pond'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)
            res.loc[res.dm,'Bea_subsoil'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)*\
            res.fairsubsoil**(10/3)/(res.fairsubsoil +res.fwatsubsoil)**2 #Effective air diffusion coefficient
            res.loc[res.dm==False,'D_air'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)
        if 'topsoil' in numc:
            res.loc[res.dm,'Deff_topsoil'] = res['dummy'].mul(chemsumm.WatDiffCoeff, level = 0)*\
                res.fwattopsoil**(10/3)/(res.fairtopsoil +res.fwattopsoil)**2 #Effective water diffusion coefficient 
            res.loc[res.dm,'Bea_topsoil'] = res['dummy'].mul(chemsumm.AirDiffCoeff, level = 0)*\
                res.fairtopsoil**(10/3)/(res.fairtopsoil +res.fwattopsoil)**2 #Effective air diffusion coefficient 
        #Dispersivity as the sum of the effective diffusion coefficient (Deff) and ldisp.
        res.loc[res.dm,'disp'] = res.ldisp + res.Deff_water #
        return chemsumm, res
    
    def input_calc(self,locsumm,chemsumm,params,pp,numc,timeseries,flow_time = None):
        """Calculate Z, D and inp values using the compartment parameters from
        bcsumm and the chemical parameters from chemsumm, along with other 
        parameters in params.
        
        Attributes:
        ----------
                
                locsumm (df): physical properties of the systmem
                chemsumm (df): physical-chemical properties of modelled compounds
                params (df): Other parameters of the model
                pp (df): (optional) input ppLFERs to use in the model
                numc (str): (optional) List of compartments
                timeseries (df): dataframe with the time-series of the model values.
                flow_time (df): dataframe with the outputs from the flow_time module of BC Blues
        """                 
        #Initialize the results data frame as a pandas multi indexed object with
        #indices of the compound names and cell numbers
        #pdb.set_trace()
        #Make the system and add chemical properties
        chemsumm, res = self.sys_chem(locsumm,chemsumm,params,pp,numc,timeseries,flow_time)
        if flow_time is None:
            pass
        else:#Need to set flow_time as the timeseries.
            timeseries = flow_time
        #Declare constants
        Ymob_immob = params.val.Ymob_immob #Diffusion path length from mobile to immobile flow       
        try:
            res.loc[:,'Y_subsoil'] = locsumm.loc['subsoil','Depth']/2
        except AttributeError:    
            pass

        try:
             res.loc[:,'Y_topsoil'] = params.val.Ytopsoil
        except AttributeError:
            res.loc[:,'Y_topsoil'] = locsumm.loc['topsoil','Depth']/2 #Diffusion path is half the depth. 
        Ifd = 1 - np.exp(-2.8 * params.val.Beta) #Vegetation dry deposition interception fraction
        
        #Calculate activity-based Z-values (m³/m³). This is where things start
        #to get interesting if compounds are not neutral. Z(j) is the bulk Z value
        #Refs - Csiszar et al (2011), Trapp, Franco & MacKay (2010), Mackay et al (2011)
        #pdb.set_trace()
        res.loc[:,'pKa'] = res['dummy'].mul(chemsumm.pKa, level = 0) #999 = neutral
        if 'pKb' in chemsumm.columns: #Check for zwitters
            res.loc[:,'pKb'] = res['dummy'].mul(chemsumm.pKb, level = 0) #Only fill in for zwitterionic compounds
        else:
            res.loc[:,'pKb'] = np.nan
        res.loc[:,'chemcharge'] = res['dummy'].mul(chemsumm.chemcharge, level = 0) #0 = neutral, -1 acid first, 1 - base first
        for jind, j in enumerate(numc): #Loop through compartments
            #This causes fragmentation errors, can be avoided by calling:
            res = res.copy()
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            dissi_j, dissn_j, pHj, Zwi_j = 'dissi_' + str(j),'dissn_' + str(j),\
            'pH' + str(j),'Zwi_' + str(j)
            gammi_j, gammn_j, Ij, Zwn_j = 'gammi_' + str(j),'gammn_' + str(j),\
            'I' + str(j),'Zwn_' + str(j)
            Zqi_j, Zqn_j, Kdj, Kdij,rhopartj = 'Zqi_' + str(j),'Zqn_' + str(j),\
            'Kd' +str(j),'Kdi' +str(j),'rhopart' +str(j)
            #Dissociation of compounds in environmental media using Henderson-Hasselbalch equation
            #dissi_j - fraction ionic, dissn_j - fraction neutral. A pka of 999 = neutral
            #Multiplying by chemcharge takes care of the cations and the anions
            res.loc[:,dissi_j] = 1-1/(1+10.**(res.chemcharge*(res.pKa-res.loc[:,pHj])))
            #Deal with the amphoterics
            mask = np.isnan(res.pKb) == False
            if mask.sum() != 0:
                res.loc[mask,dissi_j] = 1/(1+10.**(res.pKa-res.loc[:,pHj])\
                       + 10.**(res.loc[:,pHj]-res.pKb))
            #Deal with the neutrals
            mask = res.chemcharge == 0
            res.loc[mask,dissi_j] = 0
            #Then set the neutral fraction
            res.loc[:,dissn_j] = 1-res.loc[:,dissi_j]
            #Now calculate the activity of water in each compartment or sub compartment
            #From Trapp (2010) gamman = 10^(ks*I), ks = 0.3 /M Setchenov approximation
            res.loc[:,gammn_j] = 10.**(0.3*res.loc[:,Ij])
            #Trapp (2010) Yi = -A*Z^2(sqrt(I)/(1+sqrt(I))-0.3I), A = 0.5 @ 15-20°C Davies approximation
            res.loc[:,gammi_j] = 10.**(-0.5*res.chemcharge**2*(np.sqrt(res.loc[:,Ij])/\
                   (1+np.sqrt(res.loc[:,Ij]))-0.3*res.loc[:,Ij]))
            #Now, we can calculate Zw for every compartment based on the above
            #Z values for neutral and ionic 
            res.loc[:,Zwi_j] =  res.loc[:,dissi_j]/res.loc[:,gammi_j]
            res.loc[:,Zwn_j] =  res.loc[:,dissn_j]/res.loc[:,gammn_j]
            #Now we can calculate the solids based on Kd, water diss and gamma for that compartment
            #pdb.set_trace()
            res.loc[:,Zqi_j] =  res.loc[:,Kdij] * res.loc[:,rhopartj]/1000 * res.loc[:,Zwi_j]
            res.loc[:,Zqn_j] =  res.loc[:,Kdj] * res.loc[:,rhopartj]/1000 * res.loc[:,Zwn_j]
            Zij, Znj, Zwj, Zqj,Zjind,Zj ='Zi_'+str(j),'Zn_'+str(j),'Zw_'+str(j),\
            'Zq_'+str(j),'Z'+str(jind),'Z'+str(j)
            fpartj,fwatj,fairj,Kawj='fpart'+str(j),'fwat'+ str(j),'fair'+ str(j),'Kaw'+ str(j)
            #Set the mask for whether the compartment is discretized or not.
            if locsumm.loc[j,'Discrete'] == 1:
                mask = res.dm.copy(deep=True)
            else: 
                mask = res.dm.copy(deep=True) ==False
            #This mask may not have worked - for now switch to true so that it just doesn't make a difference
            #mask.loc[:]  = True
            #Finally, lets calculate the Z values in the compartments
            if j in 'air': #Air we need to determine hygroscopic growth
                #Aerosol particles - composed of water and particle, with the water fraction defined
                #by hygroscopic growth of the aerosol. Growth is defined as per the Berlin Spring aerosol from Arp et al. (2008)
                #Max RH is 100
                timeseries.loc[timeseries.RH>100.,'RH'] = 100
                #Berlin Spring aerosol from Arp et al. (2008)
                try:
                    GF = np.interp(timeseries.RH.reindex(res.index,level=1)/100,xp = [0.12,0.28,0.77,0.92],fp = \
                            [1.0,1.08,1.43,2.2],left = 1.0,right = 2.5)
                except TypeError:
                    GF = np.interp(timeseries.loc[(slice(None),'air'),'RH']/100,xp = [0.12,0.28,0.77,0.92],fp = \
                            [1.0,1.08,1.43,2.2],left = 1.0,right = 2.5)
                    GF = pd.DataFrame(GF,index = res.index.levels[1])
                    GF = GF.reindex(res.index,level = 1)
                #Volume fraction of water in aerosol 
                VFQW_a = (GF - 1) * locsumm.Density.water / ((GF - 1) * \
                          locsumm.Density.water + locsumm.PartDensity.air)
                res.loc[mask,fwatj] = res.loc[mask,fwatj] + res.loc[mask,fpartj]*VFQW_a[0] #add aerosol water from locsumm
                res.loc[mask,fairj] = 1 - res.loc[mask,fwatj] - res.loc[mask,fpartj]
                #mask = res.dm #Change the mask so that Z values will be calculated across all x, to calculate diffusion
            res.loc[mask,Zij] = res.loc[mask,fwatj]*res.loc[mask,Zwi_j]+res.loc[mask,fpartj]\
            *res.loc[mask,Zqi_j] #No Zair for ionics
            res.loc[mask,Znj] = res.loc[mask,fwatj]*res.loc[mask,Zwn_j]+res.loc[mask,fpartj]\
            *res.loc[mask,Zqn_j]+res.loc[mask,fairj]*res.loc[mask,Kawj]
            res.loc[mask,Zwj] = res.loc[mask,fwatj]*res.loc[mask,Zwi_j] + res.loc[mask,fwatj]\
            *res.loc[mask,Zwn_j] #pure water
            res.loc[mask,Zqj] = res.loc[mask,fpartj]*res.loc[mask,Zqi_j] + res.loc[mask,fpartj]\
            *res.loc[mask,Zqn_j] #Solid/particle phase Z value
            res.loc[mask,Zj] = res.loc[mask,Zij] + res.loc[mask,Znj] #Overall Z value - need to copy for the index version bad code but w/e
            res.loc[mask,Zjind] = res.loc[mask,Zj] 
            if j in ['air','water','pond','drain']: #=Calculate the particle-bound fraction in air and water
                phij = 'phi'+str(j)
                res.loc[mask,phij] = res.loc[mask,fpartj]*(res.loc[mask,Zqi_j] + res.loc[mask,Zqn_j])/res.loc[mask,Zj]
        
        #Set the rainrate for wet deposition processes
        if 'air' in numc:
            try:#Moved from 'pond' to 'drain' 20220111
                rainrate = pd.DataFrame(np.array(timeseries.loc[(slice(None),'drain'),'QET']),index = timeseries.index.levels[0]).reindex(res.index,level=1).loc[:,0]
                #Test this.
                #rainrate = rainrate.reindex(res.index,level=1).loc[:,0]
            except TypeError:
                rainrate = timeseries.RainRate.reindex(res.index,level=1)
        #D values (m³/h), N (mol/h) = a*D (activity based)
        #Loop through compartments to set D values
        #pdb.set_trace()
        for jind, j in enumerate(numc): #Loop through compartments
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            Drj, Dadvj, Zj, rrxnj, Vj= 'Dr' + str(j),'Dadv' + str(j),'Z' + \
            str(jind),'rrxn' + str(j),'V' + str(j)
            advj, Dtj = 'adv' + str(j),'DT' + str(jind)
            if locsumm.loc[j,'Discrete'] == 1:
                mask = res.dm
            else: 
                mask = res.dm == False
            #Assuming that degradation is not species specific and happens on 
            #the bulk medium (unless over-written)
            if j in ['air','water','pond','drain']:#differentiate particle & bulk portions
                phij , rrxnq_j = 'phi'+str(j),'rrxnq_'+str(j)
                res.loc[mask,Drj] = (1-res.loc[mask,phij])*res.loc[mask,Vj]* res.loc[mask,rrxnj]\
                +res.loc[mask,phij]*res.loc[mask,rrxnq_j]
            else:
                res.loc[mask,Drj] = res.loc[mask,Zj] * res.loc[mask,Vj] * res.loc[mask,rrxnj] 
            res.loc[mask,Dadvj] = res.loc[mask,Zj] * res.loc[mask,advj]
            res.loc[mask,Dtj] = res.loc[mask,Drj] + res.loc[mask,Dadvj] #Initialize total D value
            #Now we will go through compartments. Since this is a model of transport in water, we assume there is always 
            #a water compartment and that the water compartment is always first. This "water" is the mobile subsurface water.
            if j in ['water']: #interacts with subsoil and topsoil and pond (if present)
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment go here
                            Zwk = 'Zw_'+str(k)
                            #Water exfiltration
                            res.loc[mask,'D_waterexf'] = res.loc[mask,'Qwaterexf']*res.loc[mask,'Zw_water']
                            #Drainage (if present). 20220819 - made this implicit.
                            #pdb.set_trace()
                            res.loc[:,'D_waterdrain'] = 0 
                            try:
                                draincell = int(res.dm.groupby(level=2).mean().sum()-1)
                                res.loc[(slice(None),slice(None),draincell),'D_waterdrain'] = \
                                    res.loc[(slice(None),slice(None),draincell),'Qout']\
                                        *res.loc[(slice(None),slice(None),draincell),'Zw_water']
                            except KeyError:
                                pass

                            res.loc[mask,D_jk] = res.loc[:,'D_waterexf'] + res.loc[:,'D_waterdrain']                       
                        elif k in ['subsoil','topsoil']:
                            if k in ['subsoil']:
                                y = Ymob_immob
                                A = res.Asubsoil
                            elif k in ['topsoil']:
                                y = res.Y_topsoil
                                A = res.AsoilV
                            D_djk,D_mjk,Detjk,Zwk,Zk,Qetk = 'D_d'+str(j)+str(k),'D_m'+str(j)+str(k),'Det'+str(j)+str(k),\
                            'Zw_'+str(k),'Z'+str(k),'Qet'+str(k)
                            fwatk, Vk, D_mkj, = 'fwat'+str(k),'V' + str(k),'D_m'+str(k)+str(j),
                            #Calculate D water subsoil from Paraiba (2002)
                            Lw = res.Zwater*res.Deff_water/y
                            Lss = res.loc[mask,Zk]*res.Deff_subsoil/y
                            res.loc[mask,D_djk] = A*Lw*Lss/(Lw+Lss)
                            res.loc[mask,D_mjk] = params.val.wmim*res.loc[mask,fwatk]*res.loc[mask,Vk]*res.loc[mask,'Zwater'] #Mixing of mobile & immobile water
                            res.loc[mask,D_mkj] = params.val.wmim*res.loc[mask,fwatk]*res.loc[mask,Vk]*res.loc[mask,Zwk] #Mixing of mobile & immobile water
                            res.loc[mask,Detjk] = res.loc[mask,Qetk]*res.loc[mask,Zwk] #ET flow goes through subsoil first - may need to change
                            res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[mask,Detjk] + res.loc[mask,D_mjk]
                            res.loc[mask,D_kj] = res.loc[mask,D_djk]+ res.loc[mask,D_mkj]/params.val.immobmobfac
                        #If there is a ponding zone, then the activity at the upstream boundary condition is the activity in the pond.
                        #We will have to make this explicit at the beginning of the ADRE
                        elif k in ['pond']:
                            Zwk = 'Zw_'+str(k)
                            #Flow from pond to first cell. This assumes that all particles are captured in the soil, for now.
                            res.loc[res.dm==False,'D_infps'] = np.array(res.loc[(slice(None),slice(None),0),'Qin'])\
                                *res.loc[res.dm==False,Zwk] #Infiltration from the ponding zone to mobile water
                            #Overall D values - We are going to calculate advection explicitly so won't put it here. 
                            res.loc[mask,D_jk] = 0#res.loc[:,'D_inf']
                            res.loc[mask,D_kj] = 0  
                        else: #Other compartments Djk = 0
                            res.loc[mask,D_jk] = 0
                    #Add Djk to Dt & set nans to zero        
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0        
                    res.loc[mask,Dtj] += res.loc[mask,D_jk]
                    
            #Subsoil- water, topsoil, roots, air (if no topsoil or pond),drain, pond(if present)
            elif j in ['subsoil','topsoil']:
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            res.loc[mask,D_jk] = 0 #Nothing else from subsoil (lined bottom or to drain layer)
                        elif k in ['water']:
                            if j == 'subsoil':
                                y = Ymob_immob
                                A = res.Asubsoil
                            elif j == 'topsoil':
                                y = res.Y_topsoil
                                A = res.AsoilV
                            D_djk,D_mjk,D_etkj,Zwk,Zk,Qetj = 'D_d'+str(j)+str(k),'D_m'+str(j)+str(k),'D_et'+str(k)+str(j),\
                            'Zw_'+str(k),'Z'+str(k),'Qet'+str(j)
                            #pdb.set_trace()
                            res.loc[mask,D_djk] =  1/(1/(params.val.kxw*A*res.Zwater)+y/(A*res.Deff_water*res.loc[mask,Zwk])) #Diffusion from water to soil
                            res.loc[mask,D_mjk] = params.val.wmim*res.loc[mask,fwatk]*res.loc[mask,Vk]*res.loc[mask,Zwk] #Mixing of mobile & immobile water
                            res.loc[mask,D_etkj] = res.loc[mask,Qetj]*res.loc[mask,Zwk] #ET flow goes through subsoil first - may need to change
                            res.loc[mask,D_jk] = res.loc[mask,D_djk] + res.loc[:,D_etkj] + res.loc[mask,D_mjk]
                            res.loc[mask,D_kj] = res.loc[mask,D_djk] + res.loc[mask,D_mjk]
                        elif k in ['subsoil','topsoil']:#Subsoil - topsoil and vice-versa
                            D_djk,D_skj,Zwj,Zwk,Qj_k,Qk_j = 'D_d'+str(j)+str(k),'D_s'+str(k)+str(j),\
                            'Zw_'+str(j),'Zw_'+str(k),'Q'+str(j)+'_'+str(k),'Q'+str(k)+'_'+str(j)                     
                            #Check if topsoil is discretized. If not, transfer from top of subsoil to non-discretized compartment.
                            #Added 20220111 for the Quebec st. tree trench
                            if locsumm.loc['topsoil','Discrete'] == 0:
                                    maskss = (res.x == min(res.x)) 
                                    maskts = res.dm==False
                            else:
                                maskss = maskts = res.dm
                            res.loc[maskts,D_djk] = res.loc[maskss,D_djk] = 1/(np.array((res.Y_subsoil/(res.AsoilV*res.Deff_water*res.Zw_water))[maskss])\
                                   +np.array((res.Y_topsoil/(res.AsoilV*res.Deff_topsoil*res.Zw_topsoil))[maskts])) #Diffusion - both ways
                            res.loc[maskts,D_skj] = np.array((params.val.U42*res.AsoilV)[maskss])*np.array(res.Zq_topsoil[maskts]) #Particle settling - only from top to subsoil
                            #Water flow between compartments. 
                            res.loc[maskss,'D_infsubsoiltopsoil'] = res.loc[maskss,Qj_k]*res.loc[maskss,Zwj] 
                            res.loc[maskts,'D_inftopsoilsubsoil'] = res.loc[maskts,Qk_j]*res.loc[maskts,Zwk] 
                            if j == 'subsoil':
                                res.loc[maskss,D_jk] = res.loc[mask,D_djk] + res.loc[mask,'D_infsubsoiltopsoil'] #sub- to topsoil
                                res.loc[maskts,D_kj] = res.loc[mask,D_djk] + res.loc[mask,D_skj] + res.loc[mask,'D_inftopsoilsubsoil'] #top- to subsoil  
                            else: #k = topsoil
                                res.loc[maskts,D_jk] = res.loc[mask,D_djk] + res.loc[mask,D_skj] +res.loc[mask,'D_inftopsoilsubsoil']#top- to subsoil 
                                res.loc[maskss,D_kj] = res.loc[mask,D_djk] + res.loc[mask,'D_infsubsoiltopsoil']#sub- to topsoil
                        elif k in ['rootbody','rootxylem','rootcyl']:
                            D_rdkj,Vk,Zk= 'D_rd'+str(k)+str(j),'V'+str(k),'Z'+str(kind)
                            if (j in ['topsoil'] and params.val.covered == 1):
                                res.loc[mask,D_jk] = 0
                                res.loc[mask,D_kj] = 0 #Assume roots don't interact with topsoil in covered system
                            elif k in ['rootbody']:
                                Nj,Nk,Qetj,tempj,tempk = 'N'+str(j),'N'+str(k),'Qet'+str(j),'temp'+str(j),'temp'+str(k)
                                Dsr_nj,Dsr_ij,Zw_j,Zwn_j,Zwi_j,Arootj = 'Dsr_n'+str(j),'Dsr_i'+str(j),'Zw_'+str(j),'Zwn_'+str(j),'Zwi_'+str(j),'Aroot'+str(j)
                                Drs_nj,Drs_ij,Zw_k,Zwn_k,Zwi_k = 'Drs_n'+str(j),'Drs_i'+str(j),'Zw_'+str(k),'Zwn_'+str(k),'Zwi_'+str(k)
                                D_apoj,Qetj = 'D_apo'+str(j),'Qet'+str(j)
                                #First, calculate the value of N =zeF/RT
                                res.loc[mask,Nj] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[mask,tempj])
                                res.loc[mask,Nk] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[mask,tempk])       
                                res.loc[mask,Dsr_nj] = res.loc[mask,Arootj]*(res.kspn*res.loc[mask,Zwn_j])
                                res.loc[mask,Dsr_ij] = res.loc[mask,Arootj]*(res.kspi*res.loc[mask,Zwi_j]*res.loc[mask,Nj]/(np.exp(res.loc[mask,Nj])-1))
                                #20210726 Adding free flow between soil solution and root free space. Applies to both sides.
                                res.loc[mask,'Drs_fs'] = res.loc[mask,Vk] * params.val.k_fs * res.loc[mask,Zw_k] 
                                #Root back to soil
                                res.loc[mask,Drs_nj] = res.loc[mask,Arootj]*(res.kspn*res.loc[mask,Zwn_k])
                                res.loc[mask,Drs_ij] = res.loc[mask,Arootj]*(res.kspi*res.loc[mask,Zwi_k]*res.loc[mask,Nk]/(np.exp(res.loc[mask,Nk])-1))
                                res.loc[mask,'Dsr_fs'] = res.loc[mask,Vk] * params.val.k_fs * res.loc[mask,Zw_j] 
                                res.loc[res.chemcharge == 0,Dsr_ij], res.loc[res.chemcharge == 0,Drs_ij] = 0,0 #Set neutral to zero
                                
                                #Overall D values
                                res.loc[mask,D_jk] = res.loc[mask,Dsr_nj] + res.loc[mask,Dsr_ij] + res.loc[mask,'Dsr_fs']
                                res.loc[mask,D_kj] = res.loc[mask,Drs_nj] + res.loc[mask,Drs_ij] + res.loc[mask,'Drs_fs']
                            elif k in ['rootxylem']: 
                                res.loc[mask,D_apoj] = res.loc[mask,Qetj]*(params.val.f_apo)*(res.loc[mask,Zw_j]) #Apoplast bypass straight to the xylem
                                res.loc[mask,D_jk] = res.loc[mask,D_apoj] 
                                res.loc[mask,D_kj] = 0                                
                            else:#Central cylinder just root death (added below)
                                res.loc[mask,D_jk] = 0
                                res.loc[mask,D_kj] = 0
                            if (j not in ['topsoil'] or params.val.covered != 1):
                                res.loc[mask,D_rdkj] = (1-params.val.froot_top)*res.loc[mask,Vk]*res.loc[mask,Zk]*params.val.k_rd  #Root death
                                res.loc[mask,D_kj] += res.loc[mask,D_rdkj]
                        elif k in ['shoots']:
                            if ('topsoil' in numc and j =='subsoil') or (params.val.covered == 1):
                                res.loc[mask,D_jk] = 0
                                res.loc[mask,D_kj] = 0
                            else:
                                #Canopy drip 
                                if locsumm.loc['shoots','Discrete'] == 0:
                                    masksh = res.dm==False
                                else:
                                    masksh = res.dm
                                res.loc[masksh,'D_cd'] = res.A_shootair * rainrate*(params.val.Ifw - params.val.Ilw)*params.val.lamb * res.Zshoots  
                                #Wax erosion
                                res.loc[masksh,'D_we'] = res.A_shootair * params.val.kwe * res.Zshoots   
                                #litterfall & plant death?
                                res.loc[masksh,'D_lf'] = res.Vshoots * res.Zshoots   * params.val.Rlf    
                                #Overall D Values
                                res.loc[mask,D_jk] = 0
                                res.loc[masksh,D_kj] = res.D_cd + res.D_we + res.D_lf
                        elif k in ['air']:#No soil-air transport from covered layers - subsoil where topsoil is present, topsoil if covered
                            if ('topsoil' in numc and j =='subsoil') or (params.val.covered == 1):
                                res.loc[mask,D_jk] = 0
                                res.loc[mask,D_kj] = 0
                            else:
                                if ('topsoil' not in numc) and (j == 'subsoil'):
                                    y = np.array(res.iloc[0].x/2) #Half the top cell.
                                    Bea = np.array(res.loc[(res.x == min(res.x)),'Bea_subsoil'])
                                    #For the BC, only the top of the soil will interact with the air
                                    masks = (res.x == min(res.x)) 
                                    maska = res.dm==False
                                else:
                                    y = res.Y_topsoil
                                    Bea= res.Bea_topsoil
                                    #***CHECK BEFORE USE***
                                    masks = res.dm
                                    maska = res.dm
                                Zw_j,D_djk,Deff_j = 'Zw_'+str(j),'D_d'+str(j)+str(k),'Deff_'+str(j)
                                Zq_k,Zw_k = 'Zq_'+str(k),'Zw_'+str(k)
                                #Getting the values in the correct cells took some creative indexing here, making this formula complicated.
                                #20221207 - Changed so that diffusion is only driven by the gas-phase Zair. 
                                res.loc[masks,D_djk] = 1/(1/(params.val.ksa*res[masks].Asoilair*np.array((1-res.phiair[maska])*res.Zair[maska]))\
                                       +y/(res[masks].Asoilair*Bea*np.array((1-res.phiair[maska])*res.Zair[maska])+\
                                    res[masks].Asoilair*np.array(res.loc[masks,Deff_j])*np.array(res.loc[masks,Zw_j]))) #Dry diffusion
                                res.loc[maska,D_djk] = np.array(res.loc[masks,D_djk])
        
                                #From air to top cell of soil. 
                                #res.loc[maska,'D_wdairsoil'] = res[maska].Asoilair*np.array(res.loc[maska,Zw_k])*rainrate.reindex(res.index,level=1).loc[:,0]*(1-params.val.Ifw) #Wet gas deposion
                                res.loc[maska,'D_wdairsoil'] = res[maska].Asoilair*np.array(res.loc[maska,Zw_k])*rainrate*(1-params.val.Ifw) #Wet gas deposion
                                res.loc[masks,'D_wdairsoil'] = 0
                                res.loc[maska,'D_qairsoil'] = res[res.dm==False].Asoilair * res.loc[maska,Zq_k] \
                                    *rainrate*res[maska].fpartair*params.val.Q*(1-params.val.Ifw)  #Wet dep of aerosol
                                res.loc[masks,'D_qairsoil'] = 0
                                res.loc[maska,'D_dairsoil'] = res[maska].Asoilair * res.loc[maska,Zq_k]\
                                    *  params.val.Up * res[maska].fpartair* (1-Ifd) #dry dep of aerosol
                                res.loc[masks,'D_dairsoil'] = 0
                                #Overall D values
                                #pdb.set_trace()
                                #Soil to air - only diffusion
                                res.loc[masks,D_jk] = res.loc[masks,D_djk] 
                                #Air to soil - diffusion, wet & dry gas and particle deposition.
                                res.loc[maska,D_kj] = res.loc[maska,D_djk]+res[maska].D_wdairsoil+res[maska].D_qairsoil+res[maska].D_dairsoil
                        #Soil/Pond. We are going to treat the advective portion explicitly, then the rest will be implicit.
                        elif k in ['pond']:
                            Zq_j,Zq_k,Zw_j,Zw_k = 'Zq_'+str(j),'Zq_'+str(k),'Zw_'+str(j),'Zw_'+str(k),
                            mask0 = (res.x == min(res.x)) 
                            #Define pond/water area - for now, same as pond/air
                            y = np.array(res.iloc[0].x/2) #Half the top cell.
                            pondD = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Depth']),index = timeseries.index.levels[0])/2).reindex(res.index,level=1).loc[:,0]
                            Apondsoil = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Area']),index = timeseries.index.levels[0])).reindex(res.index,level=1).loc[:,0]
                            Apondsoil[pondD==0] = 0 #The pond area is based on a minimum surveyed value of 5.74m², if there is no depth there is no area.
                            #Particle capture from pond to first cell. This will be advective & explicit, assume 100% capture for now.
                            res.loc[res.dm==False,'D_qps'] = np.array(res.loc[(slice(None),slice(None),0),'Qin'])*res.loc[res.dm==False,Zq_k]
                            #Diffusive transfer
                            res.loc[res.dm==False,'D_dsoilpond'] = 1/(1/(params.val.kxw*Apondsoil[res.dm==False]*res.loc[res.dm==False,Zw_k])\
                                   +(y)/(Apondsoil[res.dm==False]*np.array(res[mask0].Deff_subsoil)*np.array(res.loc[mask0,Zw_j])))
                            res.loc[mask0,'D_dsoilpond'] = np.array(res.loc[res.dm==False,'D_dsoilpond']) #This goes both ways, need in both cells.
                            res.loc[np.isnan(res.D_dsoilpond),'D_dsoilpond'] = 0 #Set nans to zero just in case
                            #Particle Resuspension
                            res.loc[mask0,'D_r94'] = params.val.Urx*Apondsoil*res.loc[:,Zq_j]
                            #Overall D Values - for soil to pond we have diffusion and resuspension
                            res.loc[mask0,D_jk] = res.D_dsoilpond + res.D_r94
                            #For pond to soil we will not put the particle transport in as it is calculated explicitly. So, only diffusion.
                            res.loc[res.dm==False,D_kj] = res.D_dsoilpond
                        else: #Other compartments Djk  = 0
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk]
                    
            #Roots interact with subsoil, root body, root central cylinder, xylem, shoots
            elif j in ['rootbody','rootxylem','rootcyl']: 
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            Vj,Zj,D_gj = 'V'+str(j),'Z'+str(jind),"D_g"+str(j)
                            res.loc[mask,D_gj] = params.val.k_rg*res.loc[mask,Vj]*res.loc[mask,Zj] #Root growth as first order
                            res.loc[mask,D_jk] = res.loc[mask,D_gj]
                        #Root body interacts with xylem.
                        elif (j in ['rootbody']) & (k in ['rootxylem']): 
                            Nj,Nk,Qetj,tempj,tempk = 'N'+str(j),'N'+str(k),'Qet'+str(j),'temp'+str(j),'temp'+str(k)
                            Zw_j,Zwn_j,Zwi_j,Zwn_k,Zwi_k = 'Zw_'+str(j),'Zwn_'+str(j),'Zwi_'+str(j),'Zwn_'+str(k),'Zwi_'+str(k),
                            #These should have been done above but just in case.
                            res.loc[mask,Nj] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[:,tempj])
                            res.loc[mask,Nk] = res.chemcharge*params.val.E_plas*params.val.F/(params.val.R*res.loc[mask,tempk])
                            #Root body to xylem - interfacial area is Arootxylem
                            res.loc[mask,'Drx_n'] = res.Arootxylem*(res.kspn*res.fwatrootbody*res.loc[:,Zwn_j]) #A7 is root/xylem interface
                            res.loc[mask,'Drx_i'] = res.Arootxylem*(res.kspi*res.fwatrootbody*res.loc[:,Zwi_j]\
                                   *res.loc[mask,Nj]/(np.exp(res.loc[mask,Nj]) - 1))
                            #xylem to root body - interfacial area is Arootxylem
                            res.loc[mask,'Dxr_n'] = res.Arootxylem*(res.kspn*res.fwatrootbody*res.loc[:,Zwn_k]) #A7 is root/xylem interface
                            res.loc[mask,'Dxr_i'] = res.Arootxylem*(res.kspi*res.fwatrootbody*res.loc[:,Zwi_k]\
                                   *res.loc[mask,Nk]/(np.exp(res.loc[mask,Nk]) - 1))
                            #Set neutral to zero
                            res.loc[res.chemcharge == 0,'Drx_i'], res.loc[res.chemcharge == 0,'Dxr_i'] = 0,0 #Set neutral to zero
                            #Overall D values - note that there isn't any ET flux, all transport across the membrane is diffusive
                            res.loc[mask,D_jk] = res.loc[:,'Drx_n']+res.loc[:,'Drx_i'] 
                            res.loc[mask,D_kj] = res.loc[:,'Dxr_n']+res.loc[:,'Dxr_i']       
                        elif (j in ['rootxylem']) & (k in ['rootcyl']):
                            #Xylem goes to central cylinder, transport to xylem is accounted for above. Only ET flow (no more membranes!)
                            Zw_j = 'Zw_'+str(j)
                            res.loc[mask,'D_xc'] = res.Qet*res.loc[:,Zw_j]
                            #Overall D values - only one way.
                            res.loc[mask,D_jk] = res.loc[:,'D_xc']
                            res.loc[mask,D_kj] = 0
                        elif (j in ['rootcyl']) & (k in ['shoots']):
                            #Cylinder goes to shoots
                            Zw_j = 'Zw_'+str(j)
                            #pdb.set_trace()
                            res.loc[mask,'D_csh'] = res.Qetplant*res.loc[:,Zw_j]
                            #If the system flows vertically - flux goes up discretized units
                            if params.val.vert_flow == 1:
                                #Flux up the central cylinder. Stays in same compartment but goes vertically up discretized units.
                                res.loc[(mask) & (res.x != min(res.x)),'D_'+str(jind)+str(jind)] += res.loc[mask,'D_csh']
                                res.loc[(mask) & (res.x != min(res.x)),Dtj] += res.loc[mask,'D_csh'] 
                                #Overall D values - only one way, only from top cell.
                                res.loc[(res.x == min(res.x)) ,D_jk] = res.loc[:,'D_csh']
                                res.loc[mask,D_kj] = 0               
                            else:
                                res.loc[mask,D_jk] = res.loc[:,'D_csh'] #From central cylinder to shoots
                                res.loc[mask,D_kj] = 0
                                          
                        else: #Other compartments Djk = Dkj = 0
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk]                            
                            
            elif j in ['shoots']: #Shoots interact with air, central cylinder & soil. Only air still needs to be done here.
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            #Shoot growth - Modelled as first order decay
                            Vj,Zj,D_gj = 'V'+str(j),'Z'+str(jind),"D_g"+str(j)
                            #pdb.set_trace()
                            res.loc[mask,D_gj] = params.val.k_sg*res.loc[mask,Vj]*res.loc[mask,Zj] #shoot growth as first order
                            res.loc[mask,D_jk] = res.loc[mask,D_gj]
                        elif k in ['air']:
                            Zn_j,Zw_k,Zq_k = 'Zn_'+str(j),'Zw_'+str(k),'Zq_'+str(k)
                            #Volatilization to air, only neutral species. Ashoots is the interfacial area
                            res.loc[mask,'D_dshootsair'] = res.kvv*res.A_shootair*res.loc[:,Zn_j]                   
                            res.loc[mask,'D_rv'] = res.A_shootair * res.loc[:,Zw_k]*rainrate* params.val.Ifw  #Wet dep of gas to shoots
                            res.loc[mask,'D_qv'] = res.A_shootair * res.loc[:,Zq_k]*rainrate\
                                * params.val.Q * params.val.Ifw #Wet dep of aerosol
                            res.loc[mask,'D_dv'] = res.A_shootair * res.loc[:,Zq_k] * params.val.Up *Ifd  #dry dep of aerosol
                            #Overall D values- only diffusion from shoots to air
                            res.loc[mask,D_jk] = res.loc[:,'D_dshootsair']
                            res.loc[mask,D_kj] = res.loc[:,'D_dshootsair'] +res.loc[:,'D_rv']+res.loc[:,'D_qv']+res.loc[:,'D_dv']       
                        else: #Other compartments Djk = Dkj = 0
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk] 

            elif j in ['air']: #Air interacts with shoots, soil, pond. Only need to do pond here. 
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            #Shoot growth - Modelled as first order decay
                            Vj,Zj = 'V'+str(j),'Z'+str(jind)
                            res.loc[mask,D_jk] = 0 #No additional loss processes from air.
                        elif k in ['pond']: #Volatilization from pond to air, then normal processes from air to pond
                            Zn_j,Zw_j,Zq_j,Zw_k = 'Zn_'+str(j),'Zw_'+str(j),'Zq_'+str(j),'Zw_'+str(k)
                            y = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Depth']),index = timeseries.index.levels[0])/2).reindex(res.index,level=1).loc[:,0]
                            Apondair = (pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Area']),index = timeseries.index.levels[0])).reindex(res.index,level=1).loc[:,0]
                            Apondair[y==0] = 0 #The pond area is based on a minimum surveyed value of 5.74m², if there is no depth there is no area.
                            #Volatilization to air, only neutral species. Ashoots is the interfacial area
                            res.loc[mask,'D_dairpond'] = 1/(1/(params.val.kma*Apondair[mask]*(1-res.phiair[mask])*res.Zair[mask])\
                                   +y/(Apondair[mask]*res.loc[mask,'D_air']*(1-res.phiair[mask])*res[mask].Zair+\
                                Apondair[mask]*res.loc[mask,'Deff_pond']*res.loc[mask,Zw_k])) #Dry diffusion
                            res.loc[np.isnan(res.D_dairpond),'D_dairpond'] = 0 #Set NaNs to zero
                            try: #Code problem - need different indexing for subsurface sinks vs BC blues.
                                #pdb.set_trace()
                                res.loc[mask,'D_rp'] = Apondair * res.loc[:,Zw_j]*rainrate* params.val.Ifw #rainrate.reindex(res.index,level=1).loc[:,0]\
                                    #Wet dep of gas to pond
                                res.loc[mask,'D_qp'] = Apondair * res.loc[:,Zq_j]*rainrate* params.val.Q  #.reindex(res.index,level=1).loc[:,0]\
                                     #Wet dep of aerosol
                            except KeyError:
                                res.loc[mask,'D_rp'] = Apondair * res.loc[:,Zw_j]*rainrate.reindex(res.index,level=1)\
                                    * params.val.Ifw  #Wet dep of gas to pond
                                res.loc[mask,'D_qp'] = Apondair * res.loc[:,Zq_j]*rainrate.reindex(res.index,level=1)\
                                    * params.val.Q * params.val.Ifw #Wet dep of aerosol
                            res.loc[mask,'D_dp'] = Apondair * res.loc[:,Zq_j] * params.val.Up *Ifd  #dry dep of aerosol
                            #Overall D values- only diffusion from pond to air. jk is air-pond, kj is pond-air. 
                            res.loc[mask,D_jk] = res.loc[:,'D_dairpond'] +res.loc[:,'D_rp']+res.loc[:,'D_qp']+res.loc[:,'D_dp']
                            res.loc[mask,D_kj] = res.loc[:,'D_dairpond']                  
                        else: #Other compartments Djk = 0.
                            res.loc[mask,D_jk] = 0
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk]                             
                            
            elif j in ['pond']: #Pond interacts with soil, air and water - sometimes through advection. All done previously 
                for kind, k in enumerate(numc): #Inner loop for inter-compartmental values
                    kind = kind+1 #The compartment number, for overall & intercompartmental D values
                    D_jk,D_kj = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind)
                    #Pond overflow is done explicitly - so we will set Dadvj to 0
                    try: #First, check if intercompartmental transport between jk already done
                        res.loc[:,D_jk]
                    except KeyError:
                        if k == j: #Processes that come directly out of the compartment (not to another compartment) go here            
                            #Exfiltration from pond out of cell & weir overflow
                            Qpondexf = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_exf']),index = timeseries.index.levels[0]).reindex(res.index,level=1).loc[:,0]
                            res.loc[mask,'D_pondexf'] = Qpondexf*res.loc[mask,'Zw_pond']
                            #Weir overflow - 2022105 removed as it is already in D_adv
                            #Qpondover = pd.DataFrame(np.array(timeseries.loc[(slice(None),'pond'),'Q_out']),index = timeseries.index.levels[0]).reindex(res.index,level=1).loc[:,0]
                            #res.loc[mask,'D_pondover'] = Qpondover*res.loc[mask,'Zpond']
                            res.loc[mask,D_jk] = res.loc[:,'D_pondexf']#+res.loc[:,'D_pondover']
                        else: #Other compartments Djk = 0.
                            res.loc[mask,D_jk] = 0
                            
                    #Add D_jk to DTj for each compartment.
                    #Set nans to zero to prevent problems.
                    res.loc[np.isnan(res.loc[:,D_jk]),D_jk] = 0
                    res.loc[:,Dtj] += res.loc[:,D_jk] 
        return res
       
    def run_it(self,locsumm,chemsumm,params,pp,numc,timeseries,flow_time=None,
               input_calcs=None,last_step = None):
        """Feed the calculated values into the ADRE equation over time.
        Can be run either by itself, in which case the input_calc method 
        will be called, or with various inputs already defined. 
        
        Attributes:
        ----------
                
                locsumm (df): physical properties of the systmem
                chemsumm (df): physical-chemical properties of modelled compounds
                params (df): Other parameters of the model
                pp (df): (optional) input ppLFERs to use in the model
                numc (str): (optional) List of compartments
                timeseries (df): dataframe with the time-series of the model values.
                flow_time (optional, df): Output of the flow_time method in BioretentionBlues
                input_calcs (optional, df): Output of the input_calc method
                last_step (optional, df): Single time-step output of this 
                function used to initialize the model run.
                
                Index level 0 = chems, level 1 = time, level 2 = cell number 
        """
        #pdb.set_trace()
        if input_calcs is None:
            input_calcs = self.input_calc(locsumm,chemsumm,params,pp,numc,timeseries,flow_time=flow_time)
            
        #try: #See if there is a compartment index in the timeseries
        #    input_calcs.index.levels[2]
        #except AttributeError: #Runs the full flow_time calcs
        #    input_calcs = self.input_calc(locsumm,chemsumm,params,pp,numc,timeseries)
        #except IndexError:  #Runs just the input_calcs part
        #    input_calcs = self.input_calc(locsumm,chemsumm,params,pp,numc,timeseries,flow_time = input_calcs)
        #Now, this will be our outputs dataframe
        #res = input_calcs.copy(deep=True)
        res = input_calcs
        #If we want to run the code in segments, we just need one previous timestep. 
        #Now, we can add the influent concentrations as the upstream boundary condition
        #Assuming chemical concentration in g/m³ activity [mol/m³] = C/Z/molar mass,
        #using Z1 in the first cell (x) (0)
        #pdb.set_trace()
        if params.val.Pulse == True:
            for chem in chemsumm.index:
                chem_Min = str(chem) + '_Min'
                res.loc[(chem,slice(None),0),'Min'] = \
                timeseries.loc[:,chem_Min].reindex(input_calcs.index, level = 1)/chemsumm.MolMass[chem] #mol
                res.loc[(chem,slice(None),slice(None)),'bc_us'] = 0
        else:
            #Initialize mass in as zero - will be calcualated from upstream BC
            res.loc[:,'Min'] = 0
            for chem in chemsumm.index:
                chem_Cin = str(chem) + '_Cin'
                res.loc[(chem,slice(None),slice(None)),'Cin'] = \
                timeseries.loc[:,chem_Cin].reindex(input_calcs.index, level = 1)
                res.loc[(chem,slice(None),slice(None)),'bc_us'] = res.loc[(chem,slice(None),slice(None)),'Cin']/\
                    chemsumm.MolMass[chem]/res.loc[(chem,slice(None),0),'Z1']
        
        #Update params on the outside of the timeloop
        if params.val.vert_flow == 1:
            params.loc['L','val'] = locsumm.Depth.subsoil + locsumm.Depth.topsoil
        else:
            params.loc['L','val'] = locsumm.Length.water
        #Initialize total mass 
        res.loc[:,'M_tot'] = 0
        
        #Give number of discretized compartments to params
        #pdb.set_trace()
        params.loc['numc_disc','val'] = locsumm.loc[numc,'Discrete'].sum()
        #lexsort outside the timeloop for better performance
        res = res.sort_index()
        #Start the time loop! If you are a future coder
        #you can probably make this run faster. 
        for t in res.index.levels[1]: #Just in case index doesn't start at zero
            #Set inlet and outlet flow rates and velocities for each time
            try:
                params.val.Qout = timeseries.Qout[t]
                params.val.Qin = timeseries.Qin[t]
            except AttributeError: #Qout is calculated not given
                params.loc['Qin','val'] = res.loc[(res.index.levels[0][0],t,0),'Qwater'] #Qin to top of the water compartment
                params.loc['Qout','val'] = res.loc[(res.index.levels[0][0],t,0),'Qout'] #Qout from water compartment
            params.val.vin = params.val.Qin/(res.Awater[0])
            params.val.vout = params.val.Qout/(res.Awater[res.dm][-1])

            #Initial conditions for each compartment
            if t == res.index.levels[1][0]: #Set initial conditions here. 
                #initial Conditions
                for j in range(0,len(numc)):
                    a_val = 'a'+str(j+1) + '_t'
                    try: #From steady-state non spatially discrete model
                        res.loc[(slice(None),t,slice(None)),a_val] = last_step.iloc[:,j].reindex(res.index,level=0)
                    except AttributeError:
                        res.loc[(slice(None),t,slice(None)),a_val] = 0 #1#Can make different for the different compartments
                    except TypeError: #From last time step of spatially discrete model
                        res.loc[(slice(None),t,slice(None)),a_val] = np.array(last_step.loc[(slice(None),max(last_step.index.levels[1]),slice(None)),a_val])
                dt = timeseries.time[1]-timeseries.time[0]
 
                
            else: #Set the previous solution aj_t1 to the inital condition (aj_t)
                for j in range(0,len(numc)):
                    a_val, a_valt1 = 'a'+str(j+1) + '_t', 'a'+str(j+1) + '_t1'
                    M_val, V_val, Z_val = 'M'+str(j+1) + '_t1', 'V' + str(j+1), 'Z' + str(j+1)
                    #Define a_t so as to be mass conservative - since system is explicit for volumes etc. this can cause mass loss
                    #pdb.set_trace()
                    
                    res.loc[(slice(None),t,slice(None)),a_val] = np.array(res.loc[(slice(None),(t-1),slice(None)),M_val])\
                    /np.array(res.loc[(slice(None),(t),slice(None)),V_val])/np.array(res.loc[(slice(None),(t),slice(None)),Z_val])
                    #For the pond compartment, need to add Qin - advective step at the beginning. Otherwsie when pond dries up mass disappears.
                    if numc[j] in ['pond']:
                        res.loc[(slice(None),t,slice(None)),a_val] = np.array(res.loc[(slice(None),(t-1),slice(None)),M_val])\
                        /np.array(res.loc[(slice(None),(t),slice(None)),V_val]+dt*res.loc[(slice(None),(t),slice(max(res.index.levels[2]))),'Qin'])\
                        /np.array(res.loc[(slice(None),(t),slice(None)),Z_val])
                    #Set nans to zero - this will happen to compartments with zero volume such as the roots in the drain cell and the pond
                    res.loc[np.isnan(res.loc[:,a_val]),a_val] = 0
                dt = timeseries.time[t] - timeseries.time[t-1] #timestep can vary
            #Now - run it forwards a time step!
            #Feed the time to params
            res_t = res.loc[(slice(None),t,slice(None)),:]
            #For error checking, stop at specific time
            if (t == 260) or (t==265):#260: 467 #216:#412: #630# 260 is location of mass influx from tracer test; stop at spot for error checking
                #pdb.set_trace() #143.39999999996508
                dangit = 'cute_cat'
            #Call the ADRE code in the FugModel module
            #print(t)
            res_t = self.ADRE_1DUSS(res_t,params,numc,dt)
            for j in range(0,len(numc)): #Put sthe results - a value at the next time step and input mass - in the dataframe
                a_valt1,M_val = 'a'+str(j+1) + '_t1','M'+str(j+1) + '_t1'
                res.loc[(slice(None),t,slice(None)),a_valt1] = res_t.loc[(slice(None),t,slice(None)),a_valt1]
                res.loc[(slice(None),t,slice(None)),M_val] = res_t.loc[(slice(None),t,slice(None)),M_val]
                res.loc[(slice(None),t,slice(None)),'M_tot'] += res.loc[(slice(None),t,slice(None)),M_val]
            #Also put the water and soil inputs in.
            if params.val.vert_flow == 1:
                res.loc[(slice(None),t,slice(None)),'Mqin'] = res_t.loc[(slice(None),t,slice(None)),'Mqin']
                res.loc[(slice(None),t,slice(None)),'Min_p'] = res_t.loc[(slice(None),t,slice(None)),'Min_p']
            else: #Set upstream mass in here
                res.loc[(slice(None),t,slice(None)),'Min'] = res_t.loc[(slice(None),t,slice(None)),'Min']
                res.loc[(slice(None),t,slice(None)),'Mqin'] = 0
                res.loc[(slice(None),t,slice(None)),'Min_p'] = res.loc[(slice(None),t,slice(None)),'Min']
            try:#Add pond overflow
                res.loc[(slice(None),t,slice(None)),'Mover_p'] = res_t.loc[(slice(None),t,slice(None)),'Mover_p']
            except KeyError:
                pass
            res.loc[(slice(None),t,slice(None)),'M_xf'] = res_t.loc[(slice(None),t,slice(None)),'M_xf']
            res.loc[(slice(None),t,slice(None)),'M_n'] = res_t.loc[(slice(None),t,slice(None)),'M_n']

        return res
    
    def mass_flux(self,res_time,numc):
        """ This function calculates mass fluxes (mol/h) between compartments and
        out of the overall system. Calculations are done at the same discretization 
        level as the system, to get the overall mass fluxes for a compartment use 
        mass_flux.loc[:,'Variable'].groupby(level=[0,1]).sum() (result is in mol/h, multiply by dt for total mass)
        """
        #pdb.set_trace()
        #First determine the number of spatial discretizations 
        numx = res_time.loc[(res_time.index.levels[0][0],res_time.index.levels[1][0],slice(None)),'dm'].sum()
        res_time.loc[:,'dt'] =  res_time['time'] - res_time['time'].groupby(level=2).shift(1)
        res_time.loc[(slice(None),min(res_time.index.levels[1]),slice(None)),'dt'] = \
            np.array(res_time.loc[(slice(None),min(res_time.index.levels[1])+1,slice(None)),'dt'])
        #Make a dataframe to display mass flux on figure
        mass_flux = pd.DataFrame(index = res_time.index)
        mass_flux.loc[:,'dt'] = np.array(res_time.loc[:,'dt'])
        #pdb.set_trace()
        #First, we will add the advective transport out and in to the first and last
        #cell of each compound/time, respectively
        #N is mass flux, mol/hr
        #N_effluent = np.array(res_time.M_n[slice(None),slice(None),numx-1] - res_time.M_xf[slice(None),slice(None),numx-1])
        #Old code, changed calculation of drainage cell 20220318 so now this doesn't work
        #N_effluent = np.array(res_time.a1_t1[slice(None),slice(None),numx-1]*res_time.Z1[slice(None),slice(None),numx-1]\
        #                                    *res_time.Qout[slice(None),slice(None),numx-1])
        mass_flux.loc[:,'N_effluent'] = 0
        #mass_flux.loc[(slice(None),slice(None),numx-1),'N_effluent'] = N_effluent
        #20220819 - made drainage implicit. 
        mass_flux.loc[:,'N_effluent'] = res_time.a1_t1*res_time.D_waterdrain
        #mass_flux.loc[:,'N_effluent'] = mass_flux.N_effluent/mass_flux.dt
        mass_flux.loc[:,'N_influent'] = res_time.Min/res_time.dt #This assumes inputs are zero
        #Now, lets get to compartment-specific transport
        for jind, j in enumerate(numc):#j is compartment mass is leaving
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            Drj,Nrj,a_val, NTj, DTj= 'Dr' + str(j),'Nr' + str(j),'a'+str(jind) + '_t1','NT' + str(j),'DT' + str(jind)
            Nadvj,Dadvj = 'Nadv' + str(j),'Dadv' + str(j)
            #Transformation (reaction) in each compartment Mr = Dr*a*V
            mass_flux.loc[:,Nrj] = (res_time.loc[:,Drj] * res_time.loc[:,a_val])#Reactive mass loss
            mass_flux.loc[:,NTj] = (res_time.loc[:,DTj] * res_time.loc[:,a_val])#Total mass out
            mass_flux.loc[:,Nadvj] = (res_time.loc[:,Dadvj] * res_time.loc[:,a_val])#Advective mass out.
            if j == 'water': #Water compartment, exfiltration losses
                mass_flux.loc[:,'N_exf'] = (res_time.loc[:,'D_waterexf'] * res_time.loc[:,a_val])#Exfiltration mass loss
            elif j in ['rootbody','rootxylem','rootcyl','shoots']:
                #Growth dilution processes - in DT but nowhere else.
                Nrg_j,D_rgj = "Ng_"+str(j),"D_g"+str(j)
                mass_flux.loc[:,Nrg_j] = (res_time.loc[:,D_rgj] * res_time.loc[:,a_val])    
            elif j == 'pond':#Pond overflow mass loss
                mass_flux.loc[:,Nadvj] = res_time.loc[:,'Mover_p']/res_time.dt
            for kind, k in enumerate(numc):#From compartment j to compartment k
                if j != k:
                    kind = kind+1
                    Djk,Dkj,Njk,Nkj,Nnet_jk,ak_val = 'D_'+str(jind)+str(kind),'D_'+str(kind)+str(jind),\
                    'N' +str(j)+str(k),'N' +str(k)+str(j),'Nnet_' +str(j)+str(k),'a'+str(kind) + '_t1'
                    mass_flux.loc[:,Njk] = (res_time.loc[:,Djk] * res_time.loc[:,a_val])
                    mass_flux.loc[:,Nkj] = (res_time.loc[:,Dkj] * res_time.loc[:,ak_val])
                    mass_flux.loc[:,Nnet_jk]  = mass_flux.loc[:,Njk] - mass_flux.loc[:,Nkj]
                    
        return mass_flux
    
    def mass_balance(self,res_time,numc,mass_flux = None,normalized = False):
        """ This function calculates a mass balance and the mass transfers (g) between compartments 
        on a whole-compartment basis.
        Attributes:
            res_time (dataframe) - output from self.run_it
            numc (list) - Compartments in the system.
            mass_flux (dataframe, optional) - output from self.mass_flux. Will be calculated from res_time if absent.
            normalized (bool, optional) = Normalize the mass transfers to the total mass that has entered the system.
                Note that this will only normalize certain outputs
        """        
        #pdb.set_trace()
        try:
            mass_flux.loc[:,'dt']
        except AttributeError:
            mass_flux = self.mass_flux(res_time,numc)
        #Mass balance at teach time step.
        mbal = pd.DataFrame(index = (mass_flux.N_effluent).groupby(level=[0,1]).sum().index)
        #First, add the things that are always going to be there.
        mbal.loc[:,'time'] = np.array(res_time.loc[(slice(None),slice(None),slice(0)),'time'])
        mbal.loc[:,'Min'] = res_time.Min.groupby(level=[0,1]).sum().groupby(level=0).cumsum()#I think this is in moles?
        mbal.loc[:,'Mtot'] = res_time.M_tot.groupby(level=[0,1]).sum()
        if normalized == True:
            divisor = mbal.Min+mbal.loc[(slice(None),mbal.index.levels[1][0]),'Mtot'].reindex(mbal.index,method ='ffill')
        else:
            divisor = 1
        mbal.loc[:,'Meff'] = (mass_flux.dt*mass_flux.N_effluent).groupby(level=[0,1]).sum()/divisor
        mbal.loc[:,'Mexf'] = (mass_flux.dt*mass_flux.N_exf).groupby(level=[0,1]).sum()/divisor
        mbal.loc[:,'Mout'] = ((mass_flux.dt*mass_flux.N_effluent).groupby(level=[0,1]).sum() +\
                             (mass_flux.dt*mass_flux.N_exf).groupby(level=[0,1]).sum()).groupby(level=0).cumsum()
        for jind, j in enumerate(numc):#j is compartment mass is leaving
            jind = jind+1 #The compartment number, for overall & intercompartmental D values
            Nrj,Nadvj,Moutj,Mbalj,Mjind,Mj,Minj,Madvj,Mrj= 'Nr' + str(j),'Nadv' + str(j),'Mout' + str(j),'Mbal' + str(j),\
                'M'+str(jind)+'_t1','M'+str(j),'Min'+str(j),'Madv'+str(j),'Mr'+str(j)
            #Moutj here gives the mass out of the entire system.
            mbal.loc[:,Mrj] = (mass_flux.dt*mass_flux.loc[:,Nrj]).groupby(level=[0,1]).sum()/divisor
            mbal.loc[:,Madvj] = (mass_flux.dt*mass_flux.loc[:,Nadvj]).groupby(level=[0,1]).sum()/divisor                                 
            mbal.loc[:,Moutj] = (mbal.loc[:,Mrj]+mbal.loc[:,Madvj])
            #Total mass out of system - cumulative
            mbal.loc[:,'Mout'] += ((mass_flux.dt*mass_flux.loc[:,Nrj]).groupby(level=[0,1]).sum() +\
                                  (mass_flux.dt*mass_flux.loc[:,Nadvj]).groupby(level=[0,1]).sum()).groupby(level=0).cumsum()
            #(mbal.loc[:,Mrj].groupby(level=0).cumsum()+mbal.loc[:,Madvj].groupby(level=0).cumsum())
            #initialize
            mbal.loc[:,Minj] = 0
            for kind, k in enumerate(numc):
                #We will define net transfer as Mjk - Mkj (positive indicates positive net transfer from j to k)
                if ('Mnet'+str(k)+str(j)) in mbal.columns:
                    pass 
                elif k != j:
                    Njk,Nkj,Mjk,Mkj,Mnetjk = 'N' +str(j)+str(k),'N' +str(k)+str(j),'M'+str(j)+str(k),'M'+str(k)+str(j),'Mnet'+str(j)+str(k)
                    #Mass into each compartment will be recorded as that compartment's Mjk
                    mbal.loc[:,Mkj] = (mass_flux.dt*mass_flux.loc[:,Nkj]).groupby(level=[0,1]).sum()/divisor
                    mbal.loc[:,Minj] += mbal.loc[:,Mkj]
                    #Mass out per time step. 
                    mbal.loc[:,Mjk] = (mass_flux.dt*mass_flux.loc[:,Njk]).groupby(level=[0,1]).sum()/divisor              
                    mbal.loc[:,Moutj] += mbal.loc[:,Mjk]
                    mbal.loc[:,Mnetjk] = mbal.loc[:,Mjk] - mbal.loc[:,Mkj]
            #Mass balance for each compartment            
            #For water and soil, we may also have mass coming in from the pond zone
            if (j in 'water') and ('pond' in numc):
                mbal.loc[:,'Minp']=res_time.Min_p.groupby(level=[0,1]).sum()/divisor
                mbal.loc[:,Minj]+= mbal.loc[:,'Minp']
                mbal.loc[:,'Mnetwaterpond'] += -mbal.loc[:,'Minp'] #Negative as from pond to water.
                #For water also need to account for advection out of the system.
                mbal.loc[:,Moutj]+=mbal.loc[:,'Meff']+mbal.loc[:,'Mexf']
                
            elif j in 'subsoil'and ('pond' in numc):
                mbal.loc[:,'Minq'] = res_time.Mqin.groupby(level=[0,1]).sum()/divisor
                mbal.loc[:,Minj] += mbal.loc[:,'Minq']
                mbal.loc[:,'Mnetsubsoilpond'] += -mbal.loc[:,'Minq'] #Negative as from pond to water

            elif j in ['rootbody','rootxylem','rootcyl','shoots']:
                #Growth dilution processes - in DT but nowhere else.
                Nrg_j,Mgj = "Ng_"+str(j),"Mg"+str(j)
                mbal.loc[:,Mgj] = (mass_flux.dt*mass_flux.loc[:,Nrg_j]).groupby(level=[0,1]).sum()/divisor
                mbal.loc[:,Moutj] += mbal.loc[:,Mgj]
                mbal.loc[:,'Mout'] += (mass_flux.dt*mass_flux.loc[:,Nrg_j]).groupby(level=[0,1]).sum().groupby(level=0).cumsum()
            #If not normalized, Mj = absolute mass in compartment. Otherwise, percentage of mass in compartment at a time step (distribution)
            if normalized == True:
                mbal.loc[:,Mj] = res_time.loc[:,Mjind].groupby(level=[0,1]).sum()/divisor
            else:
                mbal.loc[:,Mj] = res_time.loc[:,Mjind].groupby(level=[0,1]).sum()
            #Positive indicates more mass entered than left in a time step
            mbal.loc[:,Mbalj] = (-res_time.loc[:,Mjind].groupby(level=[0,1]).sum()+res_time.loc[:,Mjind].groupby(level=[0,1]).sum().shift(1))/divisor\
                                +mbal.loc[:,Minj]-mbal.loc[:,Moutj]                                
            mbal.loc[(slice(None),min(mbal.index.levels[1])),Mbalj] = 0.0         
        mbal.loc[:,'Mbal'] = (mbal.loc[:,'Mout']+mbal.loc[:,'Mtot'])/divisor
        if np.sum(divisor) == 1:
            mbal.loc[mbal.Min==0,'Mbal'] = 0
        else: 
            mbal.loc[mbal.Min==0,'Mbal'] = 1
        
        #mbal.loc[:,'Mbal2'] = (mbal.loc[:,'Mbal2'])/mbal.Min           
        return mbal         

    def mass_balance_cumulative(self,numc,res_time=None,mass_flux = None,mass_balance = None,normalized=False):
        """ This function calculates cumulative mass transfers between compartments 
        on a whole-compartment (non spatially discretized) basis. 
        Attributes:
            numc - list of compartments.
            res_time (optional) - output from self.run_it. Needed if mass_balance not given.
            mass_flux (optional) - output from self.mass_flux. Needed if mass_balance not given.
            mass_balance (dataframe, optional) - Non-normalized output from mass_balance
            normalized (bool, optional) = Normalize the mass transfers to the total mass that has entered the system.
                Note that this will only normalize certain outputs
        """  
        #pdb.set_trace()
        #Set up mbal. Need to run as non-normalized in order for normalization here to work. Might fix this later, for now it is good enough
        try:
            mbal = mass_balance
        except AttributeError:                
            try:
                mbal = self.mass_balance(res_time,numc,mass_flux,normalized = False)
            except AttributeError:
                mbal = self.mass_balance(res_time,numc,normalized = False)
        #We need to get the cumulative mass fluxes. The overall values in mbal are cumulative already, so need some caution.
        #Start us off with those that will always be in the mbal dataframe, don't need to be cumulatively summed
        mbal_cum = mbal.loc[:,['time','Min','Mtot']]
        if normalized == True:
            divisor = mbal.Min+mbal.loc[(slice(None),mbal.index.levels[1][0]),'Mtot'].reindex(mbal.index,method ='ffill')
        else:
            divisor = 1    
        try:
            mbal_cum =  pd.concat([mbal_cum,mbal.loc[:,['Meff','Mexf','Minp','Minq']].groupby(level=0).cumsum()],axis=1)
            mbal_cum.loc[:,['Meff','Mexf','Mtot','Minp','Minq']] =  mbal_cum.loc[:,['Meff','Mexf','Mtot','Minp','Minq']].mul(1/divisor,axis="index")
        except KeyError:
            mbal_cum =  pd.concat([mbal_cum,mbal.loc[:,['Meff','Mexf']].groupby(level=0).cumsum()],axis=1)
            mbal_cum.loc[:,['Meff','Mexf','Mtot']] =  mbal_cum.loc[:,['Meff','Mexf','Mtot']].mul(1/divisor,axis="index")
        for j in numc:#j is compartment mass is leaving
            Mj,Madvj,Mrj= 'M'+str(j),'Madv'+str(j),'Mr'+str(j)
            #Mass is not cumulative.
            if normalized == True:
                mbal_cum.loc[:,Mj] = mbal.loc[:,Mj]/divisor
            else:
                mbal_cum.loc[:,Mj] = mbal.loc[:,Mj]
            mbal_cum.loc[:,Madvj] = mbal.loc[:,Madvj].groupby(level=0).cumsum()/divisor
            mbal_cum.loc[:,Mrj] = mbal.loc[:,Mrj].groupby(level=0).cumsum()/divisor
            #pdb.set_trace()
            if j in ['rootbody','rootxylem','rootcyl','shoots']:
                #Growth dilution processes - in DT but nowhere else.
                Mgj = "Mg"+str(j)
                mbal_cum.loc[:,Mgj] = mbal.loc[:,Mgj].groupby(level=0).cumsum()/divisor
            #Intercompartmental transfer. For net transfer, we only need to calculate once - as Mnetjk = Mjk-Mkj
            for k in numc:
                if ('Mnet'+str(k)+str(j)) in mbal_cum.columns:
                    pass
                elif k != j:
                    Mjk, Mkj, Mnetjk = 'M'+str(j)+str(k),'M'+str(k)+str(j),'Mnet'+str(j)+str(k)
                    mbal_cum.loc[:,Mjk] = mbal.loc[:,Mjk].groupby(level=0).cumsum()/divisor
                    mbal_cum.loc[:,Mkj] = mbal.loc[:,Mkj].groupby(level=0).cumsum()/divisor
                    mbal_cum.loc[:,Mnetjk] = mbal.loc[:,Mnetjk].groupby(level=0).cumsum()/divisor
        return mbal_cum
    
    def conc_out(self,numc,timeseries,chemsumm,res_time,mass_flux=None):
        """ This function calculates modeled concentrations at the outlet for all chemicals present. All values g/m³
            numc (Str): list of names of compartments
            timeseries (df): Input timeseries. 
            chemsumm (df): physical-chemical properties of modelled compounds
            res_time (optional) - output from self.run_it. 
            mass_flux (optional) - output from self.mass_flux. 
            
        """
        #pdb.set_trace()
        try:
            mass_flux.loc[:,'dt']
        except AttributeError:
            mass_flux = self.mass_flux(res_time,numc)
        numx = res_time.loc[(res_time.index.levels[0][0],res_time.index.levels[1][0],slice(None)),'dm'].sum()
        #pdb.set_trace()
        Couts = pd.DataFrame(np.array(res_time.loc[(min(res_time.index.levels[0]),slice(None),numx-1),'time']),
                                          index = res_time.index.levels[1],columns=['time'])
        try: #If there are measured and estimated flows, bring both in
            Couts.loc[:,'Qout_meas'] = timeseries.loc[:,'Qout_meas']
            Couts.loc[:,'Qout'] = np.array(res_time.loc[(min(res_time.index.levels[0]),slice(None),numx-1),'Qout'])
        except KeyError:
            pass
            #Couts.loc[:,'Qout'] = timeseries.loc[:,'Qout']
        for chem in mass_flux.index.levels[0]:
            try: #If there are measurements
                Couts.loc[:,chem+'_Coutmeas'] = timeseries.loc[:,chem+'_Coutmeas']
            except KeyError:
                pass
            #Concentration = mass flux/Q*MW
            Couts.loc[:,chem+'_Coutest'] = np.array(mass_flux.loc[(chem,slice(None),slice(None)),'N_effluent'].groupby(level=1).sum())\
                                            /np.array(Couts.loc[:,'Qout'])*np.array(chemsumm.loc[chem,'MolMass'])
            Couts.loc[np.isnan(Couts.loc[:,chem+'_Coutest']),chem+'_Coutest'] = 0.
            Couts.loc[np.isinf(Couts.loc[:,chem+'_Coutest']),chem+'_Coutest'] = 0.   
                   
        
        return Couts
    
    def concentrations(self,numc,res_time):
        """This method calculates modeled concentrations within the system.
        
        Attributes:
            numc (Str): list of names of compartments
            res_time (dataframe) - output from self.run_it
        """
        #pdb.set_trace()
        concentrations = pd.DataFrame(res_time.loc[res_time.time>=0,['time','x']])
        for jind,j in enumerate(numc):
            jind = jind+1
            aval,Zval,colname = 'a'+str(jind) + '_t1','Z'+str(jind),j+'_conc'
            concentrations.loc[:,colname] = res_time.loc[:,aval]*res_time.loc[:,Zval]
        return concentrations
    
    def mass_distribution(self,numc,res_time,timeseries,chemsumm,normalized = 'compartment'):
        """ 
        This function calculates modeled mass distributions for the different compartments. Large dataframe, same indices as res_time
        
        Attributes:
            normalized (string, optional) - 'True'/'t' will normalize to total mass in system (inlfuent + M0), 'False'/'f' = no normalization, 
                'compartment'/'c' to mass in each compartment at each time, 'overall'/'o' is overall mass M_tot
        """
        #pdb.set_trace()
        mdist = res_time.loc[:,['x','time']] #If timeseries has spin-up period, skip to actual times
        if normalized[0].lower() == 't':
            res_time.loc[np.isnan(res_time.Min),'Min'] = 0    
            divisor = res_time.Min+res_time.loc[(slice(None),res_time.index.levels[1][0],slice(None)),
                                                    'M_tot'].reindex(res_time.index,method ='ffill')
            divisor = divisor.groupby(level=[0]).cumsum()
        elif normalized[0].lower() == 'o': #overall/total mass 
            divisor = res_time.M_tot
        else:
            divisor = 1.
            
        for jind,j in enumerate(numc):
            jind = jind+1
            M_val= 'M'+str(jind) + '_t1'
            #Divide by the total mass in each compartment
            if normalized[0].lower() == 'c':
                divisor = res_time.loc[:,M_val].groupby(level=[0,1]).sum().reindex(res_time.index,method ='ffill')        
            mdist.loc[:,'M'+str(j)] = res_time.loc[:,M_val]/divisor
        return mdist
    
            
    def model_fig(self,numc,mass_balance,dM_locs,M_locs,time=None,compound=None,figname=None,dpi=100,fontsize=8,figheight=6):
        """ 
        Show modeled fluxes and mass distributions on a figure. 
        Attributes:
            mass_balance = Either the cumulative or the instantaneous mass balance, as output by the appropriate functions, normalized or not
            figname (str, optional) = name of figure on which to display outputs. Default figure should be in the same folder
            time (float, optional) = Time to display, in hours. Default will be the last timestep.
            compounds (str, optional) = Compounds to display. Default is all.
        """
        #pdb.set_trace()
        #Set up attributes that weren't given
        mbal = mass_balance
        if time is None:
            time = mbal.index.levels[1][-1]#Default is at the end of the model run.
        else:#Otherwise, find the time index from the given time in hours    
            time = mbal.index.levels[1][np.where(mbal.loc[(mbal.index.levels[0][0],
                                        slice(None)),'time']== find_nearest(mbal.loc[(mbal.index.levels[0][0],
                                        slice(None)),'time'],time))][0]
        if compound != None:
            mbal = mbal.loc[(compound,time),:]
        else:
            mbal = mbal.loc[(mbal.index.levels[0][0],time),:]
        img = plt.imread(figname)
        figsize = (img.shape[1]/img.shape[0]*figheight,figheight)
        fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
        ax.grid(False)
        plt.axis('off')    
        #Define the locations where the mass transfers (g) will be placed.            
        for j in numc:#j is compartment mass is leaving
            Mj,Mrj = 'M'+str(j),'Mr'+str(j) 
            pass
            ax.annotate(f'{mbal.loc[Mj]:.2e}',xy = M_locs[Mj],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction') #Mass distribution if normalized, abs. mass if not. At time t        
            ax.annotate(f'{mbal.loc[Mrj]:.2e}',xy = dM_locs[Mrj],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
            if j in ['water']: #Add effluent and exfiltration advection
                ax.annotate(f'{mbal.loc["Meff"]:.2e}',xy = dM_locs['Meff'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
                try:
                    ax.annotate(f'{mbal.loc["Mexf"]:.2e}',xy = dM_locs['Mexf'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
                except KeyError:
                    pass
                if 'pond' in numc: #Need to reverse some to display with appropriate conventions.
                    mbal.loc['Mnetwaterpond'] = -mbal.loc['Mnetwaterpond'] 
            elif j in ['subsoil']:#Mass enters the pond.
                if 'pond' in numc:
                    mbal.loc['Mnetsubsoilpond'] = -mbal.loc['Mnetsubsoilpond'] 
                if 'shoots' in numc:
                    mbal.loc['Mnetsubsoilshoots'] = -mbal.loc['Mnetsubsoilshoots']                 
            elif j in ['pond']:#Add mass in (mol) - goes to pond zone, NOT normalized. 
                ax.annotate(f'{mbal.loc["Min"]:.2e}',xy = dM_locs['Min'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
                ax.annotate(f'{mbal.loc["Madvpond"]:.2e}',xy = dM_locs['Madvpond'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
            elif j in ['air']:#Air has advection out the back end
                ax.annotate(f'{mbal.loc["Madvair"]:.2e}',xy = dM_locs['Madvair'],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
            for k in numc:
                if (('Mnet'+str(j)+str(k)) not in dM_locs.keys()):
                    pass
                else:
                    Mnetjk = 'Mnet'+str(j)+str(k)
                    ax.annotate(f'{mbal[Mnetjk]:.2e}',xy = dM_locs[Mnetjk],fontsize = fontsize, fontweight = 'bold',xycoords='axes fraction')
        ax.annotate(compound,xy = (0,1),fontsize = fontsize+2, fontweight = 'bold',xycoords='axes fraction')
#       '''
        #stop = 'stop'               
        ax.imshow(img,aspect='auto')
        return fig,ax