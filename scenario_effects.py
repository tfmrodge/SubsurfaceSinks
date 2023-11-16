import pandas as pd
import itertools
import pdb

inpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Modeling/Pickles/IDFouts/'
chemsumm = pd.read_excel('inputfiles/Pine8th/EngDesign_CHEMSUMM.xlsx',index_col = 0)
combos = list(itertools.product([0,1],repeat=8))
combos=pd.DataFrame(combos,columns=['fvalve', 'Foc', 'Kinf', 'Dsys', 
                         'Asys', 'Hp', 'hpipe', 'amend'])
scenario_list = pd.DataFrame(index=['fvalve', 'Foc', 'Kinf', 'Dsys', 
                         'Asys', 'Hp', 'hpipe', 'amend','fvalve_Foc','fvalve_Kinf',
                         'fvalve_Dsys','fvalve_Asys','fvalve_Hp','fvalve_amend','Foc_Kinf',
                         'Foc_Dsys','Foc_Asys','Foc_Hp','Foc_hpipe','Foc_amend','Kinf_Dsys',
                         'Kinf_Asys','Kinf_Hp','Kinf_hpipe','Kinf_amend','Dsys_Asys','Dsys_Hp',
                         'Dsys_hpipe','Dsys_amend','Asys_Hp','Asys_hpipe','Asys_amend','Hp_hpipe',
                         'Hp_amend','hpipe_amend'])
#pdb.set_trace()
#Set up output
chems = ['PFOA','Benzotriazole','6PPDQ','TCEP','Phe','BaP']
for compound in chems:
    #Effect on total percent advected
    scenario_list.loc[:,compound+'E_advected'] =0.
    #Effect on percent advected to stormsewer
    scenario_list.loc[:,compound+'E_storm'] =0.
    #Set up temp values
    scenario_list.loc[:,compound+'E_advected1'] =0.
    scenario_list.loc[:,compound+'E_storm1'] =0.
    scenario_list.loc[:,compound+'E_advected0'] =0.
    scenario_list.loc[:,compound+'E_storm0'] =0.
    scenario_list.loc[:,compound+'E_advected11'] =0.
    scenario_list.loc[:,compound+'E_storm11'] =0.
    scenario_list.loc[:,compound+'E_advected10'] =0.
    scenario_list.loc[:,compound+'E_storm10'] =0.
    scenario_list.loc[:,compound+'E_advected01'] =0.
    scenario_list.loc[:,compound+'E_storm01'] =0.
    scenario_list.loc[:,compound+'E_advected00'] =0.
    scenario_list.loc[:,compound+'E_storm00'] =0.
    
scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                 'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}    
for scenario in scenario_dict.keys():
    combos1 = combos[combos.loc[:,scenario]==True]
    combos0 = combos[combos.loc[:,scenario]==False]
    n=0
    scenario_list.loc[:,'n2'] = 0
    scenario_list.loc[:,'n3'] = 0
    for combo in combos1.index:
        scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                         'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}
        for ind, param in enumerate(scenario_dict):
            #pdb.set_trace()
            scenario_dict[param] = bool(combos1.loc[combo,param])
        filtered = [k for k,v in scenario_dict.items() if v == True]
        #testname = 'IDF_'+'_'.join(filtered)+'.pkl'
        testname = 'IDF_EngDesign'+'_'.join(filtered)+'.pkl'
        #testname = 'IDF_EngDesign_lowKn'+'_'.join(filtered)+'.pkl'
        #testname = 'IDF_defaults'
        try:
            pltdf = pd.read_pickle(inpath+testname)
            n+=1
        except FileNotFoundError:
            continue
        #pdb.set_trace()
        for compound in chems:
            pltdata = pltdf.loc[compound,:]
            #Take the updating average for the values (since some variables have 8 and some have 7 combos)
            scenario_list.loc[scenario,compound+'E_advected1'] = scenario_list.loc[scenario,compound+'E_advected1']*(n-1)/n\
                                                                  +pltdata.pct_advected.mean()/n
            scenario_list.loc[scenario,compound+'E_storm1'] = scenario_list.loc[scenario,compound+'E_storm1']*(n-1)/n\
                                                               +pltdata.pct_stormsewer.mean()/n
            for k in filtered:
                if scenario+'_'+k in scenario_list.index:
                    scenario_list.loc[k,'n2'] += 1
                    n2 = scenario_list.loc[k,'n2']
                    
                    scenario_list.loc[scenario+'_'+k,compound+'E_advected11'] = scenario_list.loc[scenario+'_'+k,compound+'E_advected11']*(n2-1)/n2\
                                                                          +pltdata.pct_advected.mean()/n2
                    scenario_list.loc[scenario+'_'+k,compound+'E_storm11'] = scenario_list.loc[scenario+'_'+k,compound+'E_storm11']*(n2-1)/n2\
                                                                       +pltdata.pct_stormsewer.mean()/n2
            notin = [k for k,v in scenario_dict.items() if v == False]
            for k in notin:
                if scenario+'_'+k in scenario_list.index:
                    scenario_list.loc[k,'n3'] += 1
                    n3 = scenario_list.loc[k,'n3']
                    scenario_list.loc[scenario+'_'+k,compound+'E_advected10'] = scenario_list.loc[scenario+'_'+k,compound+'E_advected10']*(n3-1)/n3\
                                                                          +pltdata.pct_advected.mean()/n3
                    scenario_list.loc[scenario+'_'+k,compound+'E_storm10'] = scenario_list.loc[scenario+'_'+k,compound+'E_storm10']*(n3-1)/n3\
                                                                       +pltdata.pct_stormsewer.mean()/n3
            #if 
    n=0
    scenario_list.loc[:,'n2'] = 0
    scenario_list.loc[:,'n3'] = 0
    for combo in combos0.index:
        scenario_dict = {'fvalve': False, 'Foc': False, 'Kinf':False, 'Dsys':False, 
                         'Asys':False, 'Hp':False, 'hpipe':False, 'amend':False}
        for ind, param in enumerate(scenario_dict):
            #pdb.set_trace()
            scenario_dict[param] = bool(combos0.loc[combo,param])
        filtered = [k for k,v in scenario_dict.items() if v == True]
        #testname = 'IDF_'+'_'.join(filtered)+'.pkl'
        testname = 'IDF_EngDesign'+'_'.join(filtered)+'.pkl'
        #testname = 'IDF_EngDesign_lowKn'+'_'.join(filtered)+'.pkl'
        #testname = 'IDF_defaults'
        try:
            pltdf = pd.read_pickle(inpath+testname)
            n+=1
        except FileNotFoundError:
            continue
        for compound in chems:
            pltdata = pltdf.loc[compound,:]
            #Take the updating average for the values (since some variables have 8 and some have 7 combos)
            #pdb.set_trace()
            scenario_list.loc[scenario,compound+'E_advected0'] = scenario_list.loc[scenario,compound+'E_advected0']*(n-1)/n\
                                                                  +pltdata.pct_advected.mean()/n
            scenario_list.loc[scenario,compound+'E_storm0'] = scenario_list.loc[scenario,compound+'E_storm0']*(n-1)/n\
                                                               +pltdata.pct_stormsewer.mean()/n                                                               
            for k in filtered:
                if scenario+'_'+k in scenario_list.index:
                    scenario_list.loc[k,'n2'] += 1
                    n2 = scenario_list.loc[k,'n2']
                    scenario_list.loc[scenario+'_'+k,compound+'E_advected01'] = scenario_list.loc[scenario+'_'+k,compound+'E_advected01']*(n2-1)/n2\
                                                                          +pltdata.pct_advected.mean()/n2
                    scenario_list.loc[scenario+'_'+k,compound+'E_storm01'] = scenario_list.loc[scenario+'_'+k,compound+'E_storm01']*(n2-1)/n2\
                                                                       +pltdata.pct_stormsewer.mean()/n2
            notin = [k for k,v in scenario_dict.items() if v == False]
            for k in notin:
                if scenario+'_'+k in scenario_list.index:
                    scenario_list.loc[k,'n3'] += 1
                    n3 = scenario_list.loc[k,'n3']
                    scenario_list.loc[scenario+'_'+k,compound+'E_advected00'] = scenario_list.loc[scenario+'_'+k,compound+'E_advected00']*(n3-1)/n3\
                                                                          +pltdata.pct_advected.mean()/n3
                    scenario_list.loc[scenario+'_'+k,compound+'E_storm00'] = scenario_list.loc[scenario+'_'+k,compound+'E_storm00']*(n3-1)/n3\
                                                                       +pltdata.pct_stormsewer.mean()/n3
for compound in chems:
        #Main effect is effect with intervention minus effect without - positive equals bad
        scenario_list.loc[:,compound+'E_advected'] =scenario_list.loc[:,compound+'E_advected0']\
                                                            -scenario_list.loc[:,compound+'E_advected1']
        scenario_list.loc[:,compound+'E_storm'] =scenario_list.loc[:,compound+'E_storm0']\
                                                            -scenario_list.loc[:,compound+'E_storm1']
        scenario_list.loc[:,compound+'E_advectedxy'] =0.5*(scenario_list.loc[:,compound+'E_advected11']-scenario_list.loc[:,compound+'E_advected01']\
                                                           -scenario_list.loc[:,compound+'E_advected10']+scenario_list.loc[:,compound+'E_advected00'])                                                    
        scenario_list.loc[:,compound+'E_stormxy'] =0.5*(scenario_list.loc[:,compound+'E_storm11']-scenario_list.loc[:,compound+'E_storm01']\
                                                           -scenario_list.loc[:,compound+'E_storm10']+scenario_list.loc[:,compound+'E_storm00']) 
        scenario_list = scenario_list.drop([compound+'E_advected1',compound+'E_advected0',compound+'E_advected10',compound+'E_advected01',compound+'E_advected00',compound+'E_advected11'],axis=1)
        scenario_list = scenario_list.drop([compound+'E_storm1',compound+'E_storm0',compound+'E_storm10',compound+'E_storm01',compound+'E_storm00',compound+'E_storm11'],axis=1) 
fpath = 'D:/OneDrive - UBC/Postdoc/Active Projects/6PPD/Manuscript/Figs/Pythonfigs/'    
scenario_list.to_csv(fpath+'CombEffect.csv')
#scenario_list