# -*- coding: utf-8 -*-
"""
Fugacity Model class, containing all other fugacity models within
Created on Wed Jul 25 15:52:23 2018

@author: Tim Rodgers
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:48:58 2018

@author: Tim Rodgers
"""
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import time
import pdb #Turn on for error checking
import sys
#import xarray as xr #cite as https://openresearchsoftware.metajnl.com/articles/10.5334/jors.148/

class FugModel(metaclass=ABCMeta):
    """ Fugacity model object, as described by Mackay (2001). This class 
    contains fugacity models, such as ppLFERMUM (Rodgers et al., 2018),
    and the Bioretention Cell Blues (BCBlues Rodgers et al., unpublished).
    The FugModel object is itself an abstract base class (ABC) and so cannot be
    instantiated by itself. The input_calcs abstractmethod needs to be defined for each model.
        
    Attributes:
    ----------

            locsumm (df): Properties of the compartments
            chemsumm (df): phyical-chemical properties of modelled compounds
            params (df): Other parameters of the model
            num_compartments (int): (optional) number of non-equilibirum 
            compartments and size of D value matrix
            name (str): (optional) name of the model run
    
    Methods:
    ----------

            run_model(self,calctype): run the selected model using the calculation type specified:
                fss for forward steady state, bss for backward steady state
            forward_calc_ss(self,ic,num_compartments): Do forwards steady-state calcs
            forward_step_uss(self,ic,num_compartments):
            
    Sub-Classes:
    ----------
            SubsurfaceSinks - Activity based level V multimedia model of 
            contaminant transport and fate in a bioretention cell. 
            BCBlues - BioretentionCell Blues model of Rodgers et al. (in prep)
            
    """ 
    #(self,locsumm,chemsumm,params,input_calcs,num_compartments,name)
    def __init__(self,locsumm,chemsumm,params,num_compartments,name):
        self.locsumm = locsumm
        self.chemsumm = chemsumm
        self.params = params
        self.numc = num_compartments
        self.name = name
        
    #This method needs to be instantiated for every child class
    @abstractmethod
    def input_calc(self):
        pass
        
    def run_model(self,calctype='fss'):
        if calctype == 'fss': #Peform forward steady-state calcs
            return self.forward_calc_ss(self.ic,self.numc)
        elif calctype == 'bss': #Perform backward calcs with lair concentration as target and emissions location
            return self.backward_calc_ss(self.ic,self.numc)
        
    def forward_calc_ss(self,ic,num_compartments):
        """ Perform forward calculations to determine model steady state fugacities
        based on input emissions. Initial_calcs (ic) are calculated at the initialization
        of the chosen model and include the matrix values DTi, and D_ij for each compartment
        as well as a column named compound num_compartments (numc) defines the size of the matrix
        """
        #Determine number of chemicals
        #pdb.set_trace()
        numc  = num_compartments
        try:
            numchems = len(ic.Compound)
        except AttributeError: #Error handling for different models
            numchems = len(ic.index)
            
        #Initialize output - the calculated fugacity of every compartment
        col_name = pd.Series(index = range(numc))
        for i in range(numc):
            col_name[i] = 'f'+str(i+1)
        try:
            fw_out = pd.DataFrame(index = ic['Compound'],columns = col_name)
        except KeyError:
            fw_out = pd.DataFrame(index = ic.index,columns = col_name)
        #generate matrix. Names of D values in ic must conform to these labels:
        #DTj for total D val from compartment j and D_jk for transfer between compartments j and k
        #Initialize a blank matrix of D values. We will iterate across this to solve for each compound
        D_mat = np.zeros((numchems,numc,numc))
        #initialize a blank dataframe for input vectors, RHS of matrix
        inp_val = np.zeros([numchems,numc])
        for j in range(numc): #compartment j, row index
            #Define RHS input for every compartment j
            inp_name = 'inp_' + str(j + 1) 
            try:
                inp_val[:,j] = -ic.loc[:,inp_name]
            except KeyError: #If input not given, assume zero
                inp_val[:,j] = 0
            for k in range(numc): #compartment k, column index
                if j == k:
                    DT = 'DT' + str(j + 1)
                    D_mat[:,j,k] = -ic.loc[:,DT]
                else:
                    D_val = 'D_' +str(k+1)+str(j+1) #label compartments from 1
                    if D_val in ic.columns: #Check if there is transfer between the two compartments
                        D_mat[:,j,k] = ic.loc[:,D_val]
            #Solve for fugacities f = D_mat\inp_val
        fugs = np.linalg.solve(D_mat,inp_val)
        fw_out.iloc[:,:] = fugs
        
        return fw_out

    def backward_calc_ss(self,ic,num_compartments,target_conc = 1,target_emiss = 1):
        """ Inverse modelling to determine emissions from measured concentrations
        as selected by the user through the 'target' attribute at steady state.
        Initial_calcs (ic) are calculated at the initialization of the model and 
        include the matrix values DTi, D_ij and the target fugacity (where given)
        for each compartment. This method needs a target fugacity (NOT concentration)
        to function, but the input in chemsumm is a concentration. num_compartments (numc) defines the 
        size of the matrix, target_conc tells what compartment (numbered from 1 not 0)
        the concentration corresponds with, while target_emiss defines which compartment
        the emissions are to. Default = 1, Lair in ppLFER-MUM. Currently, the output is
        a dataframe with the fugacities of each compartment and the emissions in g/h.
        
        Overall, we are going to solve a matrix of (numc-1) x (numc-1), eliminating the
        compartment with the target concentration and moving  the known fugacity 
        to the right hand side (RHS) of the equations for the other compartments. Then we 
        solve for the fugacities in all the other compartments and the inputs to the target
        compartment, which we put in the D_mat
        """
        #Initialize outputs
        #pdb.set_trace()
        col_name = pd.Series(index = range(num_compartments))
        for i in range(num_compartments):
            col_name[i] = 'f'+str(i+1) #Fugacity for every compartment
        #Emissions for the target_emiss compartment
        col_name[num_compartments+1] = 'emiss_'+str(target_emiss)
        bw_out = pd.DataFrame(index = ic['Compound'],columns = col_name)        
        #Define target name and check if there is a value for it in the ic dataframe. If not, abort
        targ_name = 'targ_' + str(target_conc)
        if targ_name not in ic.columns:
            return'Please define a target concentration for the chosen compartment, comp_' + str(target_conc)
        #initialize a matrix of numc x numc compartments.
        D_mat = pd.DataFrame(index = range(num_compartments),columns = range(num_compartments))
        #initialize a blank dataframe for input vectors, RHS of matrix.
        inp_val = pd.DataFrame(index = range(num_compartments),columns = ic.Compound)
        #Loop over the chemicals, solving for each.
        for chem in ic.index: #Index of chemical i starting at 0
            #Put the target fugacity into the output
            bw_out.iloc[chem,target_conc-1] = ic.loc[chem,targ_name]
            #Double loop to set matrix values
            #This first while loop takes the values from the input calcs and 
            #gets them into a form where they can be put in our (numc-1) x (numc-1)
            #matrix which we are solving
            j = 0 #Index to pull values from ic
            while j < num_compartments: #compartment j, index of D_mat
                #Define RHS = -Inp(j) - D(Tj)*f(T) for every compartment j using target T
                D_val = 'D_' +str(target_conc)+str(j+1) #label compartments from 1
                inp_name = 'inp_' + str(j + 1) #must have an input for every compartment, even if it is zero
                if j+1 == target_conc: #Need to use DT value for target concentration
                    DT = 'DT' + str(j + 1)
                    if j+1 == target_emiss: #Set -Inp(j) to zero for the targ_emiss row, we will subtract GCb(target_emiss) later
                        inp_val.iloc[j,chem] = ic.loc[chem,DT] * bw_out.iloc[chem,target_conc-1]
                    else:
                        inp_val.iloc[j,chem] = ic.loc[chem,DT] * bw_out.iloc[chem,target_conc-1]-ic.loc[chem,inp_name]
                elif D_val in ic.columns: #check if there is a D(Tj) value
                    if j+1 == target_emiss: #This is clunky but hopefully functional
                        inp_val.iloc[j,chem] = -ic.loc[chem,D_val] * bw_out.iloc[chem,target_conc-1]
                    else:
                        inp_val.iloc[j,chem] = -ic.loc[chem,inp_name] - ic.loc[chem,D_val]*bw_out.iloc[chem,target_conc-1]
                else: #If there is no D(Tj) then RHS = -Inp(j), unless it is the target_emiss column again
                    if j+1 == target_emiss: 
                        inp_val.iloc[j,chem] = 0
                    else:
                        inp_val.iloc[j,chem] = -ic.loc[chem,inp_name]
          
                #Set D values across each row
                k = 0 #Compartment index
                kk = 0 #Index to fill matrix
                while k < num_compartments: #compartment k, column of D_mat
                    if (k+1) == target_conc:
                        k += 1
                    if j == k:
                        DT = 'DT' + str(j + 1)
                        D_mat.iloc[j,kk] = -ic.loc[chem,DT]
                    else:
                        D_val = 'D_' +str(k+1)+str(j+1) #label compartments from 1
                        if D_val in ic.columns: #Check if there is transfer between the two compartments
                            D_mat.iloc[j,kk] = ic.loc[chem,D_val]
                        else:
                            D_mat.iloc[j,kk] = 0 #If no transfer, set to 0
                    if k+1 == num_compartments: #Final column is the input to the target_emiss compartment
                        if (j+1) == target_emiss: #This is 1 for the target_emiss column and 0 everywhere else
                            D_mat.iloc[j,kk+1] = 1
                        else:
                            D_mat.iloc[j,kk+1] = 0
                    k +=1
                    kk += 1
                j += 1
            #Solve for fugsinp = D_mat\inp_val, the last value in fugs is the total inputs
            lhs = np.array(D_mat,dtype = float)
            rhs = np.array(inp_val.iloc[:,chem],dtype = float)
            fugsinp = np.linalg.solve(lhs,rhs)
            #Subtract out the Gcb to get emissions from total inputs
            gcb_name = 'Gcb_' + str(target_emiss)
            fugsinp[-1] = fugsinp[-1] - ic.loc[chem,gcb_name]
            #Multiply by molar mass to get g/h output
            fugsinp[-1] = fugsinp[-1] * ic.loc[chem,'MolMass']
            #bwout units are mol/m³/pa for fugacities, mol/h for emissions
            bw_out.iloc[chem,0:target_conc-1] = fugsinp[0:target_conc-1]
            bw_out.iloc[chem,target_conc:] = fugsinp[target_conc-1:]
        return bw_out
    

    def ADRE_1DUSS(self,ic,params,numc,dt):
        
        """ Perform a single time step in a 1D ADRE multimedia model.
        This solution to the 1D ADRE requires flow modelling to be done seperately, 
        and will calculate across the entire spatial range of the modelled system.
        Based on the DISCUS algorithm or Manson and Wallis (2001) and inspiration
        from Kilic et al, (2009, DOI 10.1016/j.scitotenv.2009.01.057). Generalized
        so it can be used with a single mobile phase and any number of stationary
        compartments, where the mobile phase can be through porous media or surface
        flow (if porosity phi = 1). The model is unconditionally stable (hopefully),
        but accuracy might be lost if the time step is too large. This model
        can be used with activity or fugacity, but was built with activity in mind
        so watch the units if you are using it for fugacity!
        
        Attributes:
        ----------
        ic (df): should be an input file in the form of a multi indexed dataframe
        with index level 0 the chemicals & level 1 an index for every x
        All values need to be defined at every x, even if they don't change
        Required inputs for ic:
        x values at the centre of nodes, dx values for each x
        Volumes Vj, Areas Aj, Darcy's flux/velocity (q) and dispersivity (disp)
        All D values between every compartment and all others, labelled D_jk
        Z values for all compartments labelled Zj
        Activity/fugacity from current time step (t) in a column aj_t
        Upstream boundary condition bc_us (per compound), activity/fugacity
        (optional) Inputs to every compartment, inp_j
        
        params(df): Index is the parameter name, values in a "val" column
        Required inputs:
        Inflow and outflow [L³] Qin, Qout
        
        num_compartments (int): number of compartments
        dt (float): time step. Don't know if a float will work actually
        """
        #pdb.set_trace()
        #Initialize outputs  #res.loc[(slice(None),slice(None),res.dm),:][1:numx]
        res = ic.copy(deep=True)
        chems = res.index.levels[0]
        numchems = len(chems)
        numx = int(res.dm.sum()/numchems) #Number of discretized xs. 
        #Set up masks for the discretized 0th and last cells
        res.loc[:,'dm0'] = False
        res.loc[:,'dmn'] = False
        res.loc[:,'dmd'] = False
        res.loc[(slice(None),slice(None),0),'dm0'] = True #0th discretized cell
        if params.val.vert_flow == 1:
            res.loc[(slice(None),slice(None),slice(numx-2,numx-2)),'dmn'] = True #Last discretized cell excluding drain
            res.loc[(slice(None),slice(None),slice(numx-1,numx-1)),'dmd'] = True #Drainage cell
        else:
            res.loc[(slice(None),slice(None),slice(numx-1,numx-1)),'dmn'] = True #Last discretized cell 
        res.loc[:,'ndm'] = (res.dm==False) #Non-discretized cell
        #We are going to declare the drainage cell non-discretized for the purposes of the ADRE - advection only into it        
        #res.loc[res.dmd==True,'dm'] = False
        if 'dx' not in res.columns: #
            res.loc[:,'dx'] = res.groupby(level = 0)['x'].diff()
            res.loc[(slice(None),0),'dx'] = res.x[0]
        #Calculate forward and backward facial values for the discretized cells
        #Back and forward facial Volumes (L³)
        res.loc[:,'V1_b'] = (res.V1.shift(1) + res.V1)/2
        res.loc[(slice(None),slice(None),res.dm0),'V1_b'] = res.loc[(slice(None),slice(None),res.dm0),'V1'] 
        res.loc[:,'V1_f'] = (res.V1.shift(-1) + res.V1)/2
        res.loc[(slice(None),slice(None),res.dmn),'V1_f'] = res.loc[(slice(None),slice(None),res.dmn),'V1'] 
        #Fluid velocity, v, (L/T). How far the fluid moves through the media in each time step
        res.loc[:,'v_b'] = (res.v1.shift(1) + res.v1)/2
        res.loc[(slice(None),slice(None),res.dm0),'v_b'] = params.val.vin
        res.loc[:,'v_f'] = (res.v1.shift(-1) + res.v1)/2
        res.loc[(slice(None),slice(None),res.dmn),'v_f'] = params.val.vout
        #Dispersivity disp [l²/T]
        res.loc[:,'disp_b'] = (res.disp.shift(1) + res.disp)/2
        res.loc[(slice(None),slice(None),res.dm0),'disp_b'] = res.loc[(slice(None),slice(None),res.dm0),'disp']
        res.loc[:,'disp_f'] = (res.disp.shift(-1) + res.disp)/2
        res.loc[(slice(None),slice(None),res.dmn),'disp_f'] = res.loc[(slice(None),slice(None),res.dmn),'disp']
        #Activity/Fugacity capacity Z [mol]
        res.loc[:,'Z1_b'] = (res.Z1.shift(1) + res.Z1)/2
        res.loc[(slice(None),slice(None),res.dm0),'Z1_b'] = res.loc[(slice(None),slice(None),res.dm0),'Z1']
        res.loc[:,'Z1_f'] = (res.Z1.shift(-1) + res.Z1)/2
        res.loc[(slice(None),slice(None),res.dmn),'Z1_f'] = res.loc[(slice(None),slice(None),res.dmn),'Z1']
        
        #DISCUS algorithm semi-lagrangian 1D ADRE from Manson & Wallis (2000) DOI: 10.1016/S0043-1354(00)00131-7
        
        #Outside of the time loop, if flow is steady, or inside if flow changes
        #Courant number, used to determine time
        res.loc[res.dm,'c'] = res.v1*dt/res.dx
        res.loc[res.dm,'Pe'] = res.v1*res.dx/(res.disp) #Grid peclet number
        #time it takes to pass through each cell
        res.loc[res.dm,'del_0'] = res.dx/((res.v_b + res.v_f)/2)
        #Set up dummy variables to be used inside the loop
        delb_test = pd.Series().reindex_like(res[res.dm])
        delb_test[:] = 0 #Challenger time, accepted if <= dt
        #Time taken traversing full cells, not the final partial
        delb_test1 = pd.Series().reindex_like(res[res.dm]) 
        delrb_test = pd.Series().reindex_like(res[res.dm])
        delrb_test[:] = 0
        #"Dumb" distance variable
        xb_test = pd.Series().reindex_like(res[res.dm])
        xb_test[:] = 0
        #This is a bit clunky, but basically this one will stay zero until del_test>dt
        xb_test1 = pd.Series().reindex_like(res[res.dm])
        #Forward variables are the same as the backjward variables
        #but they will be shifted one fewer times (dels instead of dels+1)
        delf_test = delb_test.copy(deep = True)
        delf_test1 = delb_test1.copy(deep = True)
        delrf_test = delrb_test.copy(deep = True)
        xf_test = xb_test.copy(deep = True)
        xf_test1 = xb_test1.copy(deep = True)
        #This loop calculates the distance & time backwards that a water packet takes
        #in a time step. 
        dels = 0
        #for dels in range(int(max(1,max(np.floor(res.c))))): #Max cells any go through (could be wrong if q increases)
        #pdb.set_trace()
        try: 
            numcells = int(max(np.floor(res.c)))#max(int(max(np.floor(res.c))),1)
        except OverflowError:
            numcells = 1
        for dels in range(numcells):
            #Time to traverse a full cell
            delb_test += res.groupby(level = 0)['del_0'].shift(dels+1)
            delf_test += res.groupby(level = 0)['del_0'].shift(dels)
            #Calculate del_test1 only where a full spatial step is traversed
            delb_test1[delb_test<=dt] = delb_test[delb_test<=dt]
            delf_test1[delf_test<=dt] = delf_test[delf_test<=dt]
            #Do the same thing in reverse for delrb_test, if delrb_test is zero to prevent overwriting
            #Create a mask showing the cells that are finished
            maskb = (delb_test>dt) & (delrb_test==0)
            delrb_test[maskb] = dt - delb_test1 #Time remaining in the time step, back face
            maskf = (delf_test>dt) & (delrf_test==0)
            delrf_test[maskf] = dt - delf_test1 #Time remaining in the time step, forward face
            #Using delrb_test and the Darcy flux of the current cell, calculate  the total distance each face travels
            xb_test1[maskb] = xb_test + delrb_test * res.groupby(level = 0)['v1'].shift(dels+1)
            xf_test1[maskf] = xf_test + delrf_test * res.groupby(level = 0)['v1'].shift(dels)
            #Then, update the "dumb" distance travelled
            xb_test += res.groupby(level = 0)['dx'].shift(dels+1)
            xf_test += res.groupby(level = 0)['dx'].shift(dels)
        #Outside the loop, we clean up the boundaries and problem cases
        #Do a final iteration for remaining NaNs & 0s
        delrb_test[delrb_test==0] = dt - delb_test1
        delrf_test[delrf_test==0] = dt - delf_test1
        #TR 20231114 - replacing np.isnan with pd.isnull()
        xb_test1[pd.isnull(xb_test1)] = xb_test + delrb_test * res.groupby(level = 0)['v1'].shift(dels+1)
        xf_test1[pd.isnull(xf_test1)] = xf_test + delrf_test * res.groupby(level = 0)['v1'].shift(dels)
        #Set those which don't cross a full cell
        xb_test1[res.groupby(level = 0)['del_0'].shift(1)>=dt] = res.groupby(level = 0)['v1'].shift(1)*dt
        xf_test1[res.del_0>=dt] = res.v1*dt
        #Bring what we need to res. The above could be made a function to clean things up too.
        #Distance from the forward and back faces
        #res.loc[:,'xb'] = (res.x+res.x.shift(1))/2 - xb_test1
        #res.loc[:,'xf'] = (res.x+res.x.shift(-1))/2 - xf_test1
        #res.loc[(slice(None),slice(None),res.dmn),'xf'] = res.x + res.dx/2 - xf_test1
        res.loc[:,'xf'] = res.x + res.dx/2 - xf_test1
        
        #For continuity, xb is the xf of the previous step
        #mask = res.xb != res.groupby(level = 0)['xf'].shift(1)
        res.loc[:,'xb'] = res.groupby(level = 0)['xf'].shift(1)
        #Clean up the US boundary, anything NAN or < 0 comes from before the origin
        #maskb = (np.isnan(res.xb) | (res.xb < 0))
        #maskf = (np.isnan(res.xf)) | (res.xf < 0)
        maskb = (res.xb.isna()) | (res.xb < 0)
        maskf = (res.xf.isna()) | (res.xf < 0)
        res.loc[maskb,'xb'] = 0
        res.loc[maskf,'xf'] = 0

        #Downstream boundary, xb ends up zero so set equal to corresponding xf
        #res.loc[(slice(None),numx-1),'xb'] = np.array(res.loc[(slice(None),numx-2),'xf'])
        #Define the cumulative mass along the entire length of the system as M(x) = sum (Mi)
        #This is defined at the right hand face of each cell. M = ai*Zi*Vi, at time n
        res.loc[:,'M_i'] = res.a1_t * res.Z1 * res.V1 #Mass in each cell at time t
        res.loc[:,'M_n'] = res.groupby(level = 0)['M_i'].cumsum() #Cumulative mass
        #res.loc[:,'xbfav'] = (res.xb+res.xf)/2 #Centre of domain of influence
        #res.loc[:,'V_doi'] = (res.xf - res.xb)*res.A1
        #Then, we advect one time step. To advect, just shift everything as calculated above.
        #We will use a cubic interpolation. Unfortunately, we have to unpack the data 
        #in order to get this to work.
        for ii in chems:
            #Added the zero at the front as M_n(0) = 0
            
            xx = np.append(0,res.loc[(ii,slice(None),res.dm),'dx'].cumsum())#forward edge of each cell
            yy = np.append(0,res.loc[(ii,slice(None),res.dm),'M_n']) #Cumulative mass in the system
            #f = interp1d(xx,yy,kind='cubic',fill_value='extrapolate')
            f1 = interp1d(xx,yy,kind='linear',fill_value='extrapolate')#Linear interpolation where cubic fails
            #Determine mass at xf and xb
            res.loc[(ii,slice(None),res.dm),'M_xf'] = f1(res.loc[(ii,slice(None),res.dm),'xf'].astype('float')) 
            #Check if the mass at the RHS of the cell is more than the mass in the cell, correct.
            res.loc[res.M_xf>res.M_n,'M_xf'] = res.loc[res.M_xf>res.M_n,'M_n']
            res.loc[(ii,slice(None),res.dm),'M_xb'] = res.loc[(ii,slice(None),res.dm),'M_xf'].shift(1)
            #res.loc[(ii,slice(None),res.dmd),'M_xb'] = np.nan
            res.loc[(ii,slice(None),res.dm0),'M_xb'] = 0.
            res.loc[(ii,slice(None),res.dm),'M_star'] = res.loc[(ii,slice(None),res.dm),'M_xf']\
            - res.loc[(ii,slice(None),res.dm),'M_xb']
            
            #check if the cubic interpolation failed (<0), use linear in those places.
            mask = res.loc[(ii,slice(None),res.dm),'M_star'] < 0
            if sum(mask) == 0: #Override with linear to see if this is causing instability
                pass 
            if sum(mask) != 0:
                #pdb.set_trace()
                #Only replace where cubic failed
                #res.loc[(chems, mask),'M_star'] = f1(res.loc[(chems,mask),'xf'])\
                #- f1(res.loc[(chems, mask),'xb'])
                #Replace everywhere
                res.loc[(ii,slice(None),res.dm),'M_star'] = f1(res.loc[(ii,slice(None),res.dm),'xf'].astype('float'))\
                - f1(res.loc[(ii,slice(None),res.dm),'xb'])
                res.loc[(ii,slice(None),res.dm),'M_xf'] = f1(res.loc[(ii,slice(None),res.dm),'xf'].astype('float')) #Mass at xf, used to determine advection out of the system
                res.loc[(ii,slice(None),res.dm),'M_xb'] = res.loc[(ii,slice(None),res.dm),'M_xf'].shift(1)
        
        #US boundary conditions
        #For non-pulse systems, define mass flowing in. Some will be lost at the US BC to diffusion.
        if params.val.Pulse is False:
            M_t =  params.val.Qin*res.loc[(slice(None),slice(None),res.dm0),'bc_us']\
            *dt*res.loc[(slice(None),slice(None),res.dm0),'Z1']
            res.loc[res.dm0,'Min'] = M_t
        res.loc[:,'inp_mass1'] = 0 #initialize
        #If there is a ponding zone, we will split the particles out.
        #pdb.set_trace()
        if 'pond' in numc: 
            ponda_t, inp_val = 'a' + str(numc.index('pond')+1) +"_t", 'inp_' +str(numc.index('pond')+1)
            #The ponding zone is a single compartment, so gets the entire Min in one go.
            #So we add apond_t and the incoming mass
            pondastar = np.array(np.array(res.loc[res.ndm,ponda_t]*res.loc[res.ndm,'Zpond']*(res.Vpond[-1] + res.Qin[-1]*dt)\
                                 +np.array(res.loc[(slice(None),slice(None),0),'Min']))\
                        /res.loc[res.ndm,'Zpond']/np.array(res.Vpond[-1] + res.Qin[-1]*dt))
            pondastar[np.isnan(pondastar)] = 0 #If there is no flow or pond volume this returns NaN so correct
            Min = res.loc[(slice(None),slice(None),0),'Min']
            res.loc[res.dm0,'Mqin'] = np.array(pondastar*dt*res.loc[res.ndm,'D_qps']) #Attached to particles
            res.loc[res.dm0,'Min_p'] = np.array(pondastar*dt*res.loc[res.ndm,'D_infps'])#Water phase
            #Then, we add the Mqin to the soil 
            inp_soil = 'inp_' +str(numc.index('subsoil')+1)
            res.loc[:,inp_soil] = 0
            res.loc[res.dm0,inp_soil] = res.loc[res.dm0,'Mqin']/dt
            #20230117 - Recalculate explicit pond overflow using pondastar.
            if res.loc[res.ndm,'Dadvpond'].sum()>0:
                #Mass out
                res.loc[res.ndm,'Mover_p'] = pondastar*res.loc[res.ndm,'Dadvpond']*dt
                DTval = 'DT' + str(numc.index('pond')+1)
                res.loc[res.ndm,DTval] -= res.loc[res.ndm,'Dadvpond']#Take Dadvpond out of DTpond
                res.loc[res.ndm,'Dadvpond'] = 0.#Set Dadvpond to zero
            else:
                res.loc[res.ndm,'Mover_p'] = 0.
            #Next, we resolve this time step by making it part of the RHS for the pond cell.
            if res.Vpond[-1] == 0: #There may be a small rounding error there, so we will fix it here.
                #res.loc[res.ndm,inp_val] = 0 #Just add the rounding error to Min to keep it conservative
                if Min.sum() != 0: #Just in case, only do this correction if there is actually an Min
                    res.loc[res.dm0,'Min_p'] += np.array(pondastar*res.loc[res.ndm,'Zpond']*(res.Vpond[-1]+res.Qin[-1]*dt)\
                            -np.array(res.loc[res.dm0,'Mqin']+res.loc[res.dm0,'Min_p'])-np.array(res.loc[res.ndm,'Mover_p']))
                else:
                    res.loc[res.ndm,ponda_t] = 0 #All mass leaves pond advectively if the volume is zero.
            else:#If there is a ponding zone, the extra mass will stay in the pond. Need to account for this
                #res.loc[res.ndm,inp_val] = np.array(Min-res.loc[res.dm0,'Mqin']-res.loc[res.dm0,'Min_p'])/dt
                res.loc[res.ndm,ponda_t] = (pondastar*res.loc[res.ndm,'Zpond']*np.array(res.Vpond[-1] + res.Qin[-1]*dt)\
                            -np.array(res.loc[res.dm0,'Mqin']+res.loc[res.dm0,'Min_p'])-np.array(res.loc[res.ndm,'Mover_p']))\
                                           /res.loc[res.ndm,'Zpond']/np.array(res.Vpond[-1])
                #Check for a very small rounding error, ignore if present.
                err = (pondastar*res.loc[res.ndm,'Zpond']*np.array(res.Vpond[-1] + res.Qin[-1]*dt))\
                    - np.array(res.loc[res.dm0,'Mqin']+res.loc[res.dm0,'Min_p'])-np.array(res.loc[res.ndm,'Mover_p'])\
                    - (res.loc[res.ndm,ponda_t]*res.loc[res.ndm,'Zpond']*np.array(res.Vpond[-1]))
                relerr = err/(pondastar*res.loc[res.ndm,'Zpond']*np.array(res.Vpond[-1] + res.Qin[-1]*dt))
                if (np.sum(res.loc[res.ndm,ponda_t]<0)>0) & (np.sum(relerr<1/1000000)!=0):
                    res.loc[res.ndm,ponda_t] = 0
        else:#Otherwise initialize these.
            #db.set_trace()
            res.loc[:,'Mqin'] = 0
            res.loc[:,'Min_p'] = res.loc[:,'Min']

        #Case 1 - both xb and xf are outside the domain. 
        #All mass (moles) comes in at the influent activity & Z value (set to the first cell)
        #pdb.set_trace() #xx = res.loc[:,['xb','xf']]
        mask = (res.xb == 0) & (res.xf == 0)
        M_us = 0
        if sum(mask) != 0:
            #res[mask].groupby(level = 0)[''].sum()
            if params.val.Pulse == True:
                #20221207 - this equation was wrong, changed to fraction of time across cell/total influent time (dt)
                #M_in = res.Min_p[mask]/(res.Qin[mask]*res.del_0[mask] #Mass from pulse in cells
                M_in = res.Min_p[mask]*(res.del_0[mask]/dt) #Mass from pulse in cells
                M_in[np.isnan(M_in)] = 0 #If Qin is zero
                res.loc[mask,'M_star'] = M_in
            else:
                res.loc[mask,'M_star'] = res.bc_us[mask]*params.val.Qin*res.Z1[mask]*res.del_0[mask]#Time to traverse cell * influent
            #Record mass input from outside system for the water compartment here
            res.loc[mask,'inp_mass1'] = res.loc[mask,'M_star']
            #res.loc[mask,'M_star'] = res.bc_us*res.V1[mask]*res.Z1[mask] #Everything in these cells comes from outside of the domain
            M_us = res[mask].groupby(level = 0)['M_star'].sum()
        #Case 2 - xb is out of the range, but xf is in
        #Need to compute the sum of a spatial integral from x = 0 to xf and then the rest is temporal with the inlet
        mask = (res.xb == 0) & (res.xf != 0)
        slope = res.loc[(slice(None),slice(None),res.dm0),'M_n']/res.loc[(slice(None),slice(None),res.dm0),'dx'] #Need to ensure dimensions agree
        if sum(mask) != 0:
            #M_x = slope.reindex(res.loc[mask,'xf'].index,method = 'ffill')*np.array(res.xf[mask]) #Obsolete - just M_xf
            #For this temporal piece, we are going to just make the mass balance between the
            #c1 BCs and this case, which will only ever have one cell as long as Xf(i-1) = xb(i)
            if params.val.Pulse == True:
                try:
                    M_t = res.Min_p[res.dm0] - np.array(M_us) #Any remaining mass goes in this cell
                    M_t[np.isnan(M_t)] = 0 #If Qin is zero
                except ValueError:
                    M_t = 0
                    #M_t[np.isnan(M_t)] = 0
            else:
                M_t =  params.val.Qin*res.loc[(slice(None),slice(None),res.dm0),'bc_us']\
                *dt*res.loc[(slice(None),slice(None),res.dm0),'Z1'] - np.array(M_us)
        
            #M_t =  res.bc_us[mask]*params.val.Qin*np.array(res.Z1[slice(None),0])*np.array((dt-delf_test1.shift(1)[mask]))
            try:
                res.loc[mask,'M_star'] = np.array(res[mask].M_xf + np.array(M_t))
                res.loc[mask,'inp_mass1'] = np.array(M_t) #Mass (moles) from outside system to the water compartment)
            except ValueError:
                pdb.set_trace()
                xxx = 'whatup'
        #Case 3 - too close to origin for cubic interpolation, so we will use linear interpolation
        #Not always going to occur
        mask = (np.isnan(res.M_star)) & (~np.isnan(res.Pe)) #Peclet check excludes drain cell
        if sum(mask) !=0:
            res.loc[mask,'M_star'] = slope.reindex(res.loc[mask,'M_star'].index,method = 'ffill') * (res.xf[mask] - res.xb[mask])
        #Define advective flow for the drainage cell
        #pdb.set_trace()
        
        #Now we declare the drainage cell non-discretized for the purposes of the ADRE - advection only into it        
        res.loc[res.dmd==True,'dm'] = False
        if params.val.vert_flow == 1:
            #Mass in the drainage cell = mass at time t + effluent from penultimate cell-drain effluent
            #res.loc[res.dmd,'M_star']  = res.loc[res.dmd,'M_i'] \
            #+np.array(res.loc[res.dmn,'M_n'] - res.loc[res.dmn,'M_xf'])
            #20220819 - made this implicit by adding D_drain to input_calcs. So now, explicit advective flow in, implicit 
            #flow out
            res.loc[res.dmd,'M_star']  = res.loc[res.dmd,'M_i'] \
            +np.array(res.loc[res.dmn,'M_n'] - res.loc[res.dmn,'M_xf'])
            #20230628 - Adding capillary flow from this cell to the one above
            
            #- np.array(dt*(res.loc[res.dmd,'a1_t']*res.loc[res.dmd,'Z1']*res.loc[res.dmd,'Qout']))#This line is advective mass out. -explicit.
        #    - np.array(dt*(res.loc[res.dmd,'a1_t']*res.loc[res.dmd,'Z1']*(res.loc[res.dmd,'Qwaterexf']+res.loc[res.dmd,'Qout'])))
        #res.loc[:,'M_star'].sum()/(res.loc[:,'M_i'].sum()-np.array(dt*(res.loc[res.dmd,'a1_t']*res.loc[res.dmd,'Z1']*res.loc[res.dmd,'Qout']))) #Code to check if the mass from the advection step balances
        #- np.array(dt*(res.loc[res.dmd,'a1_t']*res.loc[res.dmd,'D_waterexf']))
        #Divide out to get back to activity/fugacity from advection
        res.loc[:,'a_star'] = res.M_star / res.Z1 / res.V1
        #checkmass =  res.loc[:,'M_i'].sum() - res.loc[:,'M_star'].sum() - np.array(dt*(res.loc[res.dmd,'a1_t']*res.loc[res.dmd,'Z1']*res.loc[res.dmd,'Qout']))
        #Error checking, does the advection part work? Advection only. ALSO CHANGE M_star to include exfiltration from final cell.
        #'''
        res.loc[:,'a1_t1'] = res.a_star 
        res.loc[:,'a2_t1'] = 0
        res.loc[:,'M1_t1'] = res.M_star 
        try:
            res.loc[:,'M2_t1'] += res.Mqin
        except KeyError:
            res.loc[:,'M2_t1'] = 0.
        #Set all others to 0
        for j in range(len(numc[2:])):
            a_val, M_val = 'a'+str(j+3) + '_t1','M'+str(j+3) + '_t1'
            res.loc[:,a_val] = 0
            res.loc[:,M_val] = 0
        
        #'''
        #Finally, we can set up & solve our implicit portion!
        #This is based on the methods of Manson and Wallis (2000) and Kilic & Aral (2009)
        #Define the spatial weighting term (P) 
        res.loc[:,'P'] =dt/(res.dx)
        #Now define the spacial weighting terms as f, m, & b. 
        #b for the (i-1) spacial step, m for (i), f for (i+1)
        #the back (b) term acting on x(i-1)
        res.loc[:,'b'] = 2*res.P*res.V1_b*res.Z1_b*res.disp_b/(res.dx + res.groupby(level = 0)['dx'].shift(1))
        #Set drainage zone to zero - no back diffusion.
        if params.val.vert_flow == 1:
            res.loc[res.dmd,'b'] = 0
        #To deal with the upstream boundary condition:
        if params.val.Pulse == False: #For continuous influx/flux allowed across U/S boundary, we can simply set dx(i-1) = dx so that:
            #20210330 - assume no flux upstream as well for the Oro Loma system.
            res.loc[(slice(None),slice(None),res.dm0),'b'] = 0#2*res.P*res.V1_b*res.Z1_b*res.disp_b/(res.dx)
        else:#For a pulse of Min we assume that none advects upstream, zero flux BC
            res.loc[(slice(None),slice(None),res.dm0),'b'] = 0
        #forward (f) term acting on x(i+1)
        res.loc[:,'f'] = 2*res.P*res.V1_f*res.Z1_f*res.disp_f/(res.dx + res.groupby(level = 0)['dx'].shift(-1))
        res.loc[(slice(None),slice(None),res.dmn),'f'] = 0 #No diffusion across downstream boundary
        if params.val.vert_flow == 1:
            res.loc[res.dmd,'f'] = 0 #Or out of drainage zone
        #Middle (m) term acting on x(i) - this will be subracted in the matrix (-m*ai)
        #Upstream and downstream BCs have been dealt with in the b and f terms
        res.loc[:,'m'] = res.f+res.b+dt*res.DT1+res.V1*res.Z1
        #For the drainage zone
        if params.val.vert_flow == 1:
            res.loc[res.dmd,'m'] = dt*res.DT1+res.V1*res.Z1#+dt*(res.loc[res.dmd,'Z1']*res.loc[res.dmd,'Qout'])
        #res.loc[res.dmd,'m'] = 0 #Turn off drainage other processes
        #Now we are going to set res.dm back to including the drainage cell.
        res.loc[:,'dm'] = res.ndm == False
        #These will make the matrix. For each spatial step, i, there will be
        #numc activities that we will track. So, in a system of water, air and sediment
        #you would have aw1, as1, aa1, aw2,as2,aa3...awnumc,asnumc,aanumc, leading to a matrix
        #of numc * i x numc * i in dimension. Then we stack these on the first axis
        #so that our matrix is numchem x numx*numc * numx*numc
        #Initialize transport matrix and RHS vector (inp)
        #pdb.set_trace()
        numcells = len(res.x) #This includes the non-discretized cell
        #pdb.set_trace()
        numc_disc = int(params.val.numc_disc) #Number of discretized cells
        numc_bulk = len(numc) - numc_disc
        mat = np.zeros([numchems,numx*numc_disc+numc_bulk,numx*numc_disc+numc_bulk])
        inp = np.zeros([numchems,numx*numc_disc+numc_bulk])
        #FILL THE MATRICES
        #First, define where the matrix values will go.
        m_vals = np.arange(0,(numx)*numc_disc,numc_disc)
        b_vals = np.arange(numc_disc,numx*numc_disc,numc_disc)
        #Then, we can set the ADRE terms. Since there will always be three no need for a loop.
        mat[:,m_vals,m_vals] = -np.array(res[res.dm].m).reshape(numchems,numx)
        mat[:,b_vals,m_vals[0:numx-1]] = np.array(res[res.dm].loc[(slice(None),slice(None),slice(1,numx)),'b']).reshape(numchems,numx-1)
        mat[:,m_vals[0:numx-1],b_vals] = np.array(res[res.dm].loc[(slice(None),slice(None),slice(0,numx-2)),'f']).reshape(numchems,numx-1)
        #mat[:,m_vals[numx-1],b_vals]
        #Next, set D values and inp values for the discretized cells. Sadly gotta loop it.
        j,k = [0,0]
        for j in range(0,numc_disc): #j is the row index
            inp_val = 'inp_' +str(j+1)
            if inp_val not in res.columns: #If no inputs given assume zero
                res.loc[:,inp_val] = 0
            #elif ('pond' in numc) & (j == numc.index('subsoil')):
                #Place the particulate influent in the uppermost soil cell.
                #inp[:,j]  = np.array(-res[res.dm0].Mqin)            
            if j == 0: #Water compartment is M* and any external inputs
                inp[:,m_vals] = np.array(-res[res.dm].M_star).reshape(numchems,numx) - dt * np.array(res.loc[res.dm,'inp_1']).reshape(numchems,numx)
                #res.loc[:,'inp_mass1'] += - dt * np.array(res.loc[:,'inp_1']) #Mass from inputs to water compartment
            else: #For the other compartments the input is the source term less the value at time n
                a_val, V_val, Z_val = 'a'+str(j+1) + '_t', 'V' + str(j+1), 'Z' + str(j+1)
                #RHS of the equation is the source plus the mass at time n for compartment j
                #pdb.set_trace()
                inp[:,m_vals+j] += - dt * np.array(res.loc[res.dm,inp_val]).reshape(numchems,numx)\
                - np.array(res.loc[res.dm,a_val]).reshape(numchems,numx)\
                *np.array(res.loc[res.dm,Z_val]).reshape(numchems,numx)\
                *np.array(res.loc[res.dm,V_val]).reshape(numchems,numx)
            for k in range(0,len(numc)): #k is the column index
                if (j == k): 
                    if j == 0:#Skip DT1 as it is in the m value
                        try:
                            #2023-06-28 added capillary from drain below to last water cell above
                            #mat[:,m_vals[0:numx-1]+j,b_vals+j] = dt * np.array(res.loc[res.dm,'D_cap'].shift(-1)).reshape(numchems,numx)[:,slice(0,numx-1)]
                            mat[:,max(m_vals[0:numx-1]+j),max(b_vals+j)] = dt*np.array(res.loc[(slice(None),slice(None),numx-1),'D_cap'])
                        except KeyError:  
                            pass            
                    else: #Otherwise, place the DT values in the matrix
                        D_val, D_valm, V_val, Z_val = 'DT' + str(k+1),'DTm' + str(k+1), 'V' + str(k+1), 'Z' + str(k+1)
                        #Modify DT to reflect the differential equation
                        if np.any(res.loc[res.dm,D_val]< 0):
                            pass
                        #For the root cylinder, we need to pass the evapotranspiration up the roots for vertical systems
                        elif (numc[j] in ['rootcyl']) and (params.val.vert_flow == 1):
                            #pdb.set_trace()
                            #From root cylinder below to root cylinder above.
                            mat[:,m_vals[0:numx-1]+j,b_vals+j] = dt * np.array(res.loc[res.dm,'D_csh'].shift(-1)).reshape(numchems,numx)[:,slice(0,numx-1)]
                        res.loc[:,D_valm] = dt*res.loc[:,D_val] + res.loc[:,V_val]*res.loc[:,Z_val]
                        #Diagonal cannot be zero to solve the equation, set to -1. Since the a value here will always be 0 this shouldn't be a problem.
                        res.loc[res.loc[:,D_valm]==0,D_valm] = 1
                        mat[:,m_vals+j,m_vals+j] = -np.array(res.loc[res.dm,D_valm]).reshape(numchems,numx)
                elif k < numc_disc: #Place the intercompartmental D values for the discretized cells
                    D_val = 'D_' +str(k+1)+str(j+1)
                    if np.any(res.loc[res.dm,D_val]< 0): 
                        pass
                    mat[:,m_vals+j,m_vals+k] = dt * np.array(res.loc[res.dm,D_val]).reshape(numchems,numx)
                else: #Mass transfer from non-discretized cells is only through the top cell (for BCs at least may need to fix later)
                    D_val = 'D_' +str(k+1)+str(j+1)
                    if numc_disc == 0:#np.any(res.loc[res.dm,D_val]< 0):
                        pass
                    else:#Only in top finite volume. Transport from discretized to non-discretized cells
                        mat[:,j,(numx)*numc_disc+k-numc_disc] = dt * np.array(res.loc[res.ndm,D_val]).reshape(numchems)
                    
        #Upstream boundary - need to add the diffusion term. DS boundary is dealt with already
        if params.val.Pulse == False: #no flux US boundary for mass pulse, 
            inp[:,0] += -res.loc[(slice(None),slice(None),res.dm0),'b']*(res.loc[(slice(None),slice(None),res.dm0),'bc_us'] - res.loc[(slice(None),slice(None),res.dm0),'a_star'])#Activity gradient is ~bc_us - a_star
            res.loc[(slice(None),slice(None),res.dm0),'inp_mass1']  += np.array(res.loc[(slice(None),slice(None),res.dm0),'b']*(res.loc[(slice(None),slice(None),res.dm0),'bc_us'] - res.loc[(slice(None),slice(None),res.dm0),'a_star'])) #Mass added from diffusion
        #else:
        #Now, we will to do the undiscretized compartments. Probably could do in the above loop but this makes things a little more clearly separated
        if numc_bulk == 0:#Check if any undiscretized compartments exist.
            pass
        else:
            for j in range(j+1,len(numc)):
                inp_val = 'inp_' +str(j+1)
                if inp_val not in res.columns: #If no inputs given assume zero
                    res.loc[:,inp_val] = 0  
                #Input is the source term less the value at time n
                a_val, V_val, Z_val = 'a'+str(j+1) + '_t', 'V' + str(j+1), 'Z' + str(j+1)
                #RHS of the equation is the source plus the mass at time n for compartment j
                inp[:,numx*numc_disc+j-numc_disc] += - dt * np.array(res.loc[res.ndm,inp_val])\
                    - np.array(res.loc[res.ndm,a_val])\
                    *np.array(res.loc[res.ndm,Z_val])\
                    *np.array(res.loc[res.ndm,V_val])
                #k = 0
                for k in range(0,len(numc)): #k is the column index
                    if (j == k):
                        D_val, D_valm, V_val, Z_val = 'DT' + str(k+1),'DTm' + str(k+1), 'V' + str(k+1), 'Z' + str(k+1)
                        #Modify DT to reflect the differential equation
                        res.loc[res.ndm,D_valm] = dt*res.loc[:,D_val] + res.loc[:,V_val]*res.loc[:,Z_val]
                        #Diagonal cannot be zero to solve the equation, set to 1. Since the a value here will always be 0 this shouldn't be a problem.
                        res.loc[res.loc[:,D_valm]==0,D_valm] = 1
                        mat[:,numx*numc_disc+j-numc_disc,numx*numc_disc+j-numc_disc] = -np.array(res.loc[res.ndm,D_valm]).reshape(numchems)   
                    elif k < numc_disc: #Place the intercompartmental D values for the discretized cells in the leftmost column
                        D_val = 'D_' +str(k+1)+str(j+1)
                        if np.any(res.loc[res.dm,D_val]< 0):
                            pass
                        mat[:,(numx)*numc_disc+j-numc_disc,k] = dt * np.array(res.loc[res.dm0,D_val]).reshape(numchems)
                    else: #Mass transfer from non-discretized cells 
                        D_val = 'D_' +str(k+1)+str(j+1)
                        if np.any(res.loc[res.dm,D_val]< 0):
                            pass
                        #Transport between non-discretized cells
                        mat[:,(numx)*numc_disc+j-numc_disc,(numx)*numc_disc+k-numc_disc] = dt * np.array(res.loc[res.ndm,D_val]).reshape(numchems)
                    
        
        #Now, we will solve the matrix for each compound simultaneously (each D-matrix and input is stacked by compound)
        #matsol = np.linalg.solve(mat,inp)
        #20240807 - updated to add explicit shape as numpy update caused a size mismatch
        inp = inp.reshape([numchems,numx*numc_disc+numc_bulk,1])
        matsol = np.linalg.solve(mat,inp)
        matsol = matsol.reshape([numchems,numx*numc_disc+numc_bulk])
        # for chem in numchems:
        #     matsol = np.linalg.solve(mat[chem,:,:],inp[chem,:])
        #(matsol < 0).sum()
        #Error checking - Check solutions ~ inputs
        #np.dot(mat[0],matsol[0]) - inp[0]
        #Loop through the compartments and put the output values into our output res dataframe
        #numx*numc_disc+numc_bulk
        for j in range(len(numc)):
            a_val, M_val, inp_mass = 'a'+str(j+1) + '_t1','M'+str(j+1) + '_t1','inp_mass'+str(j+1)
            V_val, Z_val =  'V' + str(j+1), 'Z' + str(j+1)
            #For discretized cells
            if j < numc_disc:
                res.loc[res.dm,a_val] = matsol[:,:numx*numc_disc].reshape(numx*numchems,numc_disc)[:,j]
            else:
                res.loc[res.ndm,a_val] = matsol[:,(numx)*numc_disc+j-numc_disc].reshape(numchems)
            #Remove nans so that things don't get weird.
            res.loc[np.isnan(res.loc[:,a_val]),a_val] = 0
            #Mass can be calculated directly
            res.loc[(slice(None),slice(None),slice(None)),M_val] = res.loc[(slice(None),slice(None),slice(None)),a_val]\
                *res.loc[(slice(None),slice(None),slice(None)),V_val]*res.loc[(slice(None),slice(None),slice(None)),Z_val] 
            #Remove nans so that things don't get weird.
            res.loc[np.isnan(res.loc[:,M_val]),M_val] = 0
            if j != 0:#Skip water compartment
                #pdb.set_trace
                res.loc[(slice(None),slice(None),slice(None)),inp_mass] = dt*res.loc[:,inp_val]
            elif j == 'pond':
                #res.loc[res.ndm,'Mover_p'] = pondastar*res.loc[res.ndm,'Dadvpond']*dt
                #We are going to set the Dadv to the mass flux - should be OK
                res.loc[res.ndm,'Dadvpond'] = res.Mover_p/res.loc[res.ndm,a_val]/dt               
            if sum(res.loc[:,a_val]<0) >0: #If solution gives negative values flag it
                pdb.set_trace()
                #xxx = 'poop'
                #On error, exit.
                print('Negative activity error at '+str(res.time[0]))
                sys.exit('Negative activity error at '+str(res.time[0]))
            
        #xxx = 1
        #'''
        
        return res


    def forward_step_euler(self,ic,params,num_compartments,dt):
        """ Perform a forward calculation step to determine model unsteady-state fugacities
        based on input emissions. Input calcs need to include ic(t+1), DTi(t+1),
        and D_ij(t+1) for each compartment, mass M(n), as well as a column named
        compound. num_compartments (numc) defines the size of the matrix. 
        From Csizar, Diamond and Thibodeaux (2012) DOI 10.1016/j.chemosphere.2011.12.044
        """
        #pdb.set_trace()
        #Set up and solve a single box-model Level IV multimedia model
        res = ic
        numc = num_compartments
        chems = res.index.levels[0]
        numchems = len(chems)
        #Make the matrix. We have numchem x (numc x numc) activities to solve for
        #Initialize transport matrix and RHS vector (inp)
        mat = np.zeros([numchems,numc,numc])
        inp = np.zeros([numchems,numc])
        DT_vals = np.arange(0,numc,numc) # Define where DT values will go
        #FILL THE MATRICEs
        #Set D values and inp values. This has to be in a loop as the number of compartments might change
        j,k = [0,0]
        for j in range(0,numc): #j is the row index
            inp_val = 'inp_' +str(j+1)
            if inp_val not in res.columns: #If no inputs given assume zero
                res.loc[:,inp_val] = 0
            #For the other compartments the input is the source term less the value at time n
            a_val, V_val, Z_val = 'a'+str(j+1) + '_t', 'V' + str(j+1), 'Z' + str(j+1)
            #RHS of the equation is the source plus the mass at time n for compartment j
            inp[:,DT_vals+j] +=  np.array(-dt * res.loc[:,inp_val] - res.loc[:,a_val]\
               *res.loc[:,Z_val]*res.loc[:,V_val]).reshape(numchems,1)
            for k in range(0,numc): #k is the column index
                if (j == k): #Modify DT to reflect the differential equation
                    D_val, D_valm, V_val, Z_val = 'DT' + str(k+1),'DTm' + str(k+1), 'V' + str(k+1), 'Z' + str(k+1)
                    if np.any(res.loc[:,D_val]< 0): #Error Checking
                        pass
                    res.loc[:,D_valm] = dt*res.loc[:,D_val] + res.loc[:,V_val]*res.loc[:,Z_val]
                    mat[:,DT_vals+j,DT_vals+j] = -np.array(res.loc[:,D_valm]).reshape(numchems,1)
                else: #Place the intercompartmental D values
                    D_val = 'D_' +str(k+1)+str(j+1)
                    if np.any(res.loc[:,D_val]< 0):
                        pass
                    mat[:,DT_vals+j,DT_vals+k] = dt * np.array(res.loc[:,D_val]).reshape(numchems,1)
        #Now, we will solve the matrix for each compound simultaneously (each D-matrix and input is stacked by compound)
        matsol = np.linalg.solve(mat,inp)
        #Error checking - Check solutions ~ inputs
        #np.dot(mat[0],matsol[0]) - inp[0]
        #Loop through the compartments and put the output values into our output res dataframe
        #
        for j in range(numc):
            a_val, inp_mass = 'a'+str(j+1) + '_t1','inp_mass'+str(j+1)
            res.loc[:,a_val] = matsol.reshape(numchems,numc)[:,j]
            res.loc[:,inp_mass] = dt*res.loc[:,inp_val]
            if sum(res.loc[:,a_val]<0) >0: #If solution gives negative values flag it
                pdb.set_trace()
        #xxx = 1
        return res
    
    def IVP_matrix(self,ic,num_compartments):
        """ Set up an initial value problem for dM/dt - need to divide through by
        the volume and the Z value to convert to activity on the RHS
        This is for use with the scipy solve_ivp function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        Not sure what the best thing for varying parameters would be 
        
        Input calcs (ic) need to include , DTi(t+1), D_ij(t+1), volumes and Z v
        values for each compartment
        num_compartments (numc) defines the size of the matrix. 
        """
        #pdb.set_trace()
        #Set up and solve a single box-model Level IV multimedia model
        res = ic
        numc = num_compartments
        chems = res.index.levels[0]
        numchems = len(chems)
        #Make the matrix. We have numchem x (numc x numc) activities to solve for
        #Initialize transport matrix and RHS vector (inp)
        mat = np.zeros([numchems,numc,numc])
        DT_vals = np.arange(0,numc,numc) # Define where DT values will go
        #FILL THE MATRICEs
        #Set D values and inp values. This has to be in a loop as the number of compartments might change
        j,k = [0,0]
        for j in range(0,numc): #j is the row index
            for k in range(0,numc): #k is the column index
                if (j == k): #DT values 
                    D_val, V_val, Z_val = 'DT' + str(k+1), 'V' + str(k+1), 'Z' + str(k+1)
                    #mat[:,DT_vals+j,DT_vals+k] = -np.array(res.loc[:,D_val]).reshape(numchems,1) #Testing for mass balance
                    mat[:,DT_vals+j,DT_vals+k] = -np.array(res.loc[:,D_val]/\
                    (res.loc[:,V_val]*res.loc[:,Z_val])).reshape(numchems,1) #Divide by VZ to get the activity
                else: #Place the intercompartmental D values
                    D_val, V_val, Z_val = 'D_' +str(k+1)+str(j+1), 'V' + str(k+1), 'Z' + str(k+1)
                    #mat[:,DT_vals+j,DT_vals+k] = np.array(res.loc[:,D_val]).reshape(numchems,1) #Testing for mass balance
                    mat[:,DT_vals+j,DT_vals+k] = np.array(res.loc[:,D_val]/\
                       (res.loc[:,V_val]*res.loc[:,Z_val])).reshape(numchems,1) #Divide by VZ to get the activity
        return mat
                
          