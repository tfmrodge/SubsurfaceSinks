# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:53:19 2022

@author: trodge01
"""
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from BioretentionBlues import BCBlues
#Inputs
#outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/IDF_defaults.pkl'
outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/IDF_results.pkl'
#outpath = 'D:/GitHub/Vancouver_BC_Modeling/Pickles/IDF_nodrain.pkl'
#pltvar = 'pct_stormsewer'
numc = ['water', 'subsoil','rootbody', 'rootxylem', 'rootcyl','shoots', 'air','pond']
locsumm = pd.read_excel('inputfiles/Pine8th_BC.xlsx',index_col = 0)
chemsumm = pd.read_excel('inputfiles/6PPDQ_CHEMSUMM.xlsx',index_col = 0)
params = pd.read_excel('inputfiles/params_Pine8th.xlsx',index_col = 0)
timeseries = pd.read_excel('inputfiles/timeseries_IDFstorms.xlsx')
pltdata = pd.read_pickle(outpath)

xticks = [0.25,0.5,1,3, 6,12,24]
xticks = [np.log10(xticks),xticks]

yticks = [5,10,25,50,100]
yticks = [np.log10(yticks),yticks]
#pltvars=['RQ_av','LogD','LogI']
pltvars=['pct_stormsewer','LogD','LogI']
interplims = [0.,1]
vlims=[0.15,0.85]
#pdb.set_trace()
cmap = None#sns.cubehelix_palette(start=.75, rot=-.5,light=0.85, as_cmap=True)
#cmap = sns.cubehelix_palette(n_colors = 7,start=1.40, rot=-0.9,gamma = 0.3, hue = 0.9, dark=0.1, light=.95,as_cmap=True,reverse=True)
bc = BCBlues(locsumm,chemsumm,params,timeseries,numc) 
fig,ax = bc.plot_idfs(pltdata,xticks=xticks,yticks=yticks,pltvals=True,pltvars=pltvars,cmap=cmap,vlims=vlims,interplims=interplims)
#for i, txt in enumerate(pltdata.index):
#    ax.annotate(str(pltdata.iloc[i,0]),xy= (pltdata.iloc[i,1],pltdata.iloc[i,2]))
#ax.set_xlabel('Event Duration (hrs)')
#ax.set_ylabel('Intensity (mm/hr)')
#ax.set_xticks(xticks[0])
#ax.setxticklabels(xticks[1])
#ax.set_yticks(yticks[0],labels=yticks[1])

'''
#cmap = sns.cubehelix_palette(n_colors = 7,start=1.40, rot=-0.9,gamma = 0.3, hue = 0.9, dark=0.1, light=.95,as_cmap=True,reverse=True)
#For mass removal (soil vs water)
cmap = sns.diverging_palette(30, 250, l=40,s=80,center="light", as_cmap=True)#sep=1,
#Actual code
outs = pd.read_pickle(outpath)
pltdata = outs.loc[:,[pltvar,'LogD','LogI']]
#pltdata = pltdata.loc[:,[pltvar,'LogD','Intensity']]
fig, ax = plt.subplots(1,1,figsize = (15,10),dpi=300)

pltvars = ['LogD','LogI']
#pltvars = ["LogD",'Intensity']
df = pltdata.pivot(index=pltvars[1],columns=pltvars[0])[pltvar]
df = df.interpolate(axis=0,method='spline',order=2)
#df = df.interpolate(axis=0,method='nearest')

#df = df.interpolate(axis=0,method='quadratic')
df[np.isnan(df)] = 0
df[df>1.] = 1.
df[df<0.] = 0.

#Soil-water differences
pc = ax.contourf(df.columns,df.index,df.values,cmap=cmap,sep=1, vmin=0.15,vmax=0.85,levels=15)
#Other stuff
#pc = ax.contourf(df.columns,df.index,df.values,cmap=cmap,sep=1)
sns.lineplot(x='LogD',y='LogI',data=outs.reset_index(),hue='Frequency',ax=ax,palette=sns.color_palette('Greys')[:len(outs.Frequency.unique())])
ax.set_xlim([-0.75,1.380211])

if pltvals == True:
    for i, txt in enumerate(pltdata.index):
        #xycoords = str([pltdata.iloc[i,1],pltdata.iloc[i,2]])
        #txt = str(pltdata.iloc[i,0])
        ax.annotate(str(pltdata.iloc[i,0]),xy= (pltdata.iloc[i,1],pltdata.iloc[i,2]))
else:
    pass

fig.colorbar(pc)
figpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Manuscript/Figs/Pythonfigs/'
figname = 'IDF_Default'
#fig.savefig(figpath+figname+'.pdf',format='pdf')
'''
'''
#full_series = pd.read_excel('inputfiles/2014_timeseries.xlsx')
#SWMMseries.head()
timeseries = pd.DataFrame(columns = ['dt','elapsedtime','precip','runoff'])
ind = 0
pdb.set_trace()
timeseries.loc[ind,:] = 0.
SWMMseries = full_series#[:5000]
p_shift = 10*60 #Start changing timestep X hrs before precipitation
r_shift = 1*60 #Start changing timestep X hrs after Qin ends
SWMMseries.loc[:,'p_shift'] = SWMMseries.Precip.shift(-15*60)#Start changing X hrs before precipitation
#Chose 15 hrs to give 6,3,2,1,0.5,0.5,0.25,0.25,0.25,0.25,10/60,10/60,10/60,10/60,5/60,5/60,5/60,5/60,1/60
#SWMMseries.loc[:,'r_shift'] = SWMMseries.Runoff.shift(-1*60)#Start changing X hrs after Qin ends
pswitch = False #To ramp up
rswitch = True #To ramp down. Start true.
ramp = np.array([6.0,3.0,2.0,1.0,0.5,0.5,0.25,0.25,0.25,0.25,10/60,10/60,10/60,10/60,5/60,5/60,5/60,5/60,1/60])
rampstep = 0
dt = 6.0 #Hours. Baseline is 6 hours in dry periods, 1 minute in wet periods
t = 0
while t < len(SWMMseries.index):
    #pdb.set_trace()
    if pswitch == True: #Ramp timestep down to 1 minute and stay there until rswitch activates
        rampstep = rampstep + 1
        if rampstep >= len(ramp):#stay at 1/60 if there
            rampstep = len(ramp)-1       
        #Start to ramp up to 6 hrs if the next X minutes all have zero flow & Y mins have 0 precipitation 
        if  (rswitch == False) & (SWMMseries.loc[t:t+r_shift,'Runoff'].sum() == 0)& (SWMMseries.loc[t:t+p_shift,'Precip'].sum() == 0):
            rswitch = True
            pswitch = False
            
    elif rswitch == True:
        rampstep = rampstep - 1
        if rampstep < 0:#stay at 6 hrs if there
            rampstep = 0 
        #Start to ramp down to 1 minute when precipitation appears
        if (pswitch == False) & (SWMMseries.loc[t:t+p_shift,'Precip'].sum() != 0):
            pswitch = True
            rswitch = False
            #rampstep = 0
    #Now, we are going to make the timeseries for the BC Blues model 
    #Each timestep represents the average from t to t + dt (not including the last step)
    #Elapsed time at the beginning of the timestep (hrs)
    #timeseries.loc[ind,'elapsedtime'] += dt
    timeseries.loc[ind+1,'elapsedtime'] = timeseries.loc[ind,'elapsedtime']+dt
    #Then, change dt
    dt = ramp[rampstep]
    #Next, add the accumulated precipitation and runoff across the timestep
    SWMMend = int(dt *60)-1 #Averaging period
    timeseries.loc[ind,'dt'] = dt
    timeseries.loc[ind,'precip'] = np.sum(SWMMseries.loc[t:t+SWMMend,'Precip'])*1/60/dt #mm/hr
    timeseries.loc[ind,'runoff'] = np.sum(SWMMseries.loc[t:t+SWMMend,'Runoff'])*1/60/dt*3.6 #m³/s
    #To finish, we will go to the next t based on dt
    t += int(dt*60)
    ind += 1
    #Ramp up to 1 minute from 6 hours when pswitch triggers
timeseries = timeseries.drop(index=max(timeseries.index))
#timeseries.to_excel('inputfiles/swmmouts_2014.xlsx')

    '''