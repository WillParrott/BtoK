import collections
import sys
import h5py
import gvar as gv
import numpy as np
import corrfitter as cf
#import corrbayes
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

plt.rc("font",**{"size":18})
import os.path
import pickle
import datetime

################ F PARAMETERS  #############################
F = collections.OrderedDict()
F['conf'] = 'F'
F['filename'] = 'KBscalarvectortensor_5cfgs_negFalse.gpl'
F['masses'] = ['0.449','0.566','0.683','0.8']
F['twists'] = ['0','0.4281','1.282','2.141','2.570','2.993']
F['mtw'] = [[1,1,1,1,0,0],[1,1,1,1,1,0],[1,1,1,1,1,1],[1,1,1,1,1,1]]
F['m_l'] = '0.0074'
F['m_s'] = '0.0376'
F['Ts'] = [14,17,20]
F['tp'] = 96
F['L'] = 32
F['tmaxesBG'] = [48,48,48,48]
F['tmaxesBNG'] = [48,48,48,48]
F['tmaxesKG'] = [46,45,45,42,42,37]            #48 is upper limit, ie goes up to 47
F['tmaxesKNG'] = [46,45,45,42,42,37] 
F['tminBG'] = 3
F['tminBNG'] = 3
F['tminKG'] = 3
F['tminKNG'] = 3                            # 3 for 5 twists, 5 for first 4 
F['Stmin'] = 2
F['Vtmin'] = 2
F['Ttmin'] = 2
F['an'] = '0.1(1)'
F['SVn'] = '0.00(15)'                        #Prior for SV[n][n]
F['SV0'] = '0.0(4)'                          #Prior for SV_no[0][0] etc
F['VVn'] = '0.00(15)'
F['VV0'] = '0.1(5)'
F['TVn'] = '0.00(15)'
F['TV0'] = '0.1(5)'
F['loosener'] = 0.3                          #Loosener on V_nn[0][0] often 0.5
F['Mloosener'] = 0.05                        #Loosener on ground state 
F['oMloosener'] = 0.2                       #Loosener on oscillating ground state
F['a'] = 0.1715/(1.9006*0.1973)
F['BG-Tag'] = 'B_G5-G5_m{0}'
F['BNG-Tag'] = 'B_G5T-G5T_m{0}'
F['KG-Tag'] = 'K_G5-G5_tw{0}'
F['KNG-Tag'] = 'K_G5-G5X_tw{0}'
F['threePtTag'] = '{0}_T{1}_m{2}_m{3}_m{4}_tw{0}'

                

################ USER INPUTS ################################
#############################################################
DoFit = True
FitAll = False
TestData = False
Fit = F                                               # Choose to fit F, SF or UF
FitMasses = [0,1,2,3]                                 # Choose which masses to fit
FitTwists = [0,1,2,3,4]                               # Choose which twists to fit
FitTs = [0,1]#,2]
FitCorrs = ['BG','BNG','KG','KNG','S','V','T']  # Choose which corrs to fit ['G','NG','D','S','V']
FitAllTwists = True
Chained = False
Marginalised = False
CorrBayes = False
SaveFit = False
svdnoise = False
priornoise = False
ResultPlots = False         # Tell what to plot against, "Q", "N","Log(GBF)", False
AutoSvd = True
recalculateSvd = True
SvdFactor = 1.0                       # Multiplies saved SVD
PriorLoosener = 1.0                    # Multiplies all prior error by loosener
Nmax = 8                               # Number of exp to fit for 2pts in chained, marginalised fit
FitToGBF = True                     # If false fits to Nmax
##############################################################
#PDGGMass =
#PDGNGMass =
#PDGDMass =
middle = 3/8                      #middle in Meff Aeff estimate 3/8 normal
gap = 1/14                        #gap in the Meff Aeff estimate 1/14 works well cannot exceed 1/8
##############################################################
##############################################################

def make_params(FitMasses,FitTwists,FitTs):
    TwoPts = collections.OrderedDict()
    ThreePts = collections.OrderedDict()
    qsqPos = collections.OrderedDict()
    masses = []
    twists = []
    tmaxesBG = []
    tmaxesBNG = []
    tmaxesKG = []
    tmaxesKNG = []
    Ts = []    
    m_s = Fit['m_s']
    m_l = Fit['m_l']
    filename = Fit['filename']          
    for i in FitMasses:
        masses.append(Fit['masses'][i])
        tmaxesBG.append(Fit['tmaxesBG'][i])
        tmaxesBNG.append(Fit['tmaxesBNG'][i])
        for j in FitTwists:
            if FitAllTwists == True:
                qsqPos['m{0}_tw{1}'.format(Fit['masses'][i],Fit['twists'][j])] = 1
            else:    
                qsqPos['m{0}_tw{1}'.format(Fit['masses'][i],Fit['twists'][j])] = Fit['mtw'][i][j]
    for j in FitTwists:
        twists.append(Fit['twists'][j])
        tmaxesKG.append(Fit['tmaxesKG'][j])
        tmaxesKNG.append(Fit['tmaxesKNG'][j])
    for k in FitTs:
        Ts.append(Fit['Ts'][k])
    for twist in Fit['twists']:
        TwoPts['KGtw{0}'.format(twist)] = Fit['KG-Tag'].format(twist)
        TwoPts['KNGtw{0}'.format(twist)] = Fit['KNG-Tag'].format(twist)
    for mass in Fit['masses']:
        TwoPts['BGm{0}'.format(mass)] = Fit['BG-Tag'].format(mass)
        TwoPts['BNGm{0}'.format(mass)] = Fit['BNG-Tag'].format(mass)
        for twist in Fit['twists']:
            for T in Ts:
                ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)] = Fit['threePtTag'].format('scalar',T,m_s,mass,twist)
                ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)] = Fit['threePtTag'].format('vector',T,m_s,mass,twist)
                ThreePts['Tm{0}_tw{1}_T{2}'.format(mass,twist,T)] = Fit['threePtTag'].format('tensor',T,m_s,mass,twist)
                
                
    return(TwoPts,ThreePts,masses,twists,Ts,tmaxesBG,tmaxesBNG,tmaxesKG,tmaxesKNG,qsqPos)



def make_data(filename,N):
    if CorrBayes == True:
        dset = cf.read_dataset(filename)
        Autoprior,new_dset = corrbayes.get_prior(dset,15,N,loosener = 1.0)        
        return(Autoprior,gv.dataset.avg_data(new_dset))
    else:
        Autoprior = 0
        dset = cf.read_dataset(filename)
        for key in dset:
            print(key,np.shape(dset[key]))
        return(Autoprior,gv.dataset.avg_data(dset))



def eff_calc():
    T = Ts[0]   #play with this. Maybe middle T is best?
    tp = Fit['tp']
    #Make this do plots
    M_effs = collections.OrderedDict()
    A_effs = collections.OrderedDict()
    V_effs = collections.OrderedDict()       
    M_eff = collections.OrderedDict()
    A_eff = collections.OrderedDict()
    V_eff = collections.OrderedDict()    
    #plt.figure(1)
    for mass in Fit['masses']:
        M_effs['BGm{0}'.format(mass)] = []
        M_effs['BNGm{0}'.format(mass)] = []
        M_eff['BGm{0}'.format(mass)] = 0
        M_eff['BNGm{0}'.format(mass)] = 0
        A_effs['BGm{0}'.format(mass)] = []
        A_effs['BNGm{0}'.format(mass)] = []
        A_eff['BGm{0}'.format(mass)] = 0
        A_eff['BNGm{0}'.format(mass)] = 0
        #plt.figure(mass)
        for t in range(2,tp-2):
            BG = (data[TwoPts['BGm{0}'.format(mass)]][t-2] + data[TwoPts['BGm{0}'.format(mass)]][t+2])/(2*data[TwoPts['BGm{0}'.format(mass)]][t])
            if BG >= 1:
                M_effs['BGm{0}'.format(mass)].append(gv.arccosh(BG)/2)
            else:
                M_effs['BGm{0}'.format(mass)].append(0)
                
            BNG = (data[TwoPts['BNGm{0}'.format(mass)]][t-2] + data[TwoPts['BNGm{0}'.format(mass)]][t+2])/(2*data[TwoPts['BNGm{0}'.format(mass)]][t])
            if BNG >= 1:
                M_effs['BNGm{0}'.format(mass)].append(gv.arccosh(BNG)/2)
            else:
                M_effs['BNGm{0}'.format(mass)].append(0)            
           
        #plt.errorbar(M_effs['Gm{0}'.format(mass)][:].mean, yerr=M_effV[:].sdev, fmt='ko')
        #plt.errorbar(M_effs['Gm{0}'.format(mass)][:].mean, yerr=M_effS[:].sdev, fmt='ro')
    for twist in Fit['twists']:
        M_effs['KGtw{0}'.format(twist)] = []
        M_eff['KGtw{0}'.format(twist)] = 0
        A_effs['KGtw{0}'.format(twist)] = []
        A_eff['KGtw{0}'.format(twist)] = 0
        M_effs['KNGtw{0}'.format(twist)] = []
        M_eff['KNGtw{0}'.format(twist)] = 0
        A_effs['KNGtw{0}'.format(twist)] = []
        A_eff['KNGtw{0}'.format(twist)] = 0
        for t in range(2,tp-2):
            KG = (data[TwoPts['KGtw{0}'.format(twist)]][t-2] + data[TwoPts['KGtw{0}'.format(twist)]][t+2])/(2*data[TwoPts['KGtw{0}'.format(twist)]][t])
            if KG >= 1:
                M_effs['KGtw{0}'.format(twist)].append(gv.arccosh(KG)/2)
            else:
                M_effs['KGtw{0}'.format(twist)].append(0)
            KNG = (data[TwoPts['KNGtw{0}'.format(twist)]][t-2] + data[TwoPts['KNGtw{0}'.format(twist)]][t+2])/(2*data[TwoPts['KNGtw{0}'.format(twist)]][t])
            if KNG >= 1:
                M_effs['KNGtw{0}'.format(twist)].append(gv.arccosh(KNG)/2)
            else:
                M_effs['KNGtw{0}'.format(twist)].append(0)
    #print(M_effs)
        #plt.errorbar(M_effs['Dtw{0}'.format(twist)][:].mean, yerr=M_effS[:].sdev, fmt='ro')
    #plt.title('M_eff')
    #print('M',M_effs)
    for mass in Fit['masses']:
        denomBG = 0
        denomBNG = 0
        for i in list(range(int(tp*(middle-gap)),int(tp*(middle+gap))))+list(range(int(tp*(1-middle-gap)),int(tp*(1-middle+gap)))):
            M_eff['BGm{0}'.format(mass)] += M_effs['BGm{0}'.format(mass)][i]
            if M_effs['BGm{0}'.format(mass)][i] != 0:
                denomBG += 1
            M_eff['BNGm{0}'.format(mass)] += M_effs['BNGm{0}'.format(mass)][i]
            if M_effs['BNGm{0}'.format(mass)][i] != 0:
                denomBNG += 1
       # if denomG !=0:        
        M_eff['BGm{0}'.format(mass)] = M_eff['BGm{0}'.format(mass)]/denomBG
       # else:
       #     M_eff['Gm{0}'.format(mass)] = PDGGMass*Fit['a']
        M_eff['BNGm{0}'.format(mass)] = M_eff['BNGm{0}'.format(mass)]/denomBNG
    for twist in Fit['twists']:
        denomKG = 0
        denomKNG = 0
        for i in list(range(int(tp*(middle-gap)),int(tp*(middle+gap))))+list(range(int(tp*(1-middle-gap)),int(tp*(1-middle+gap)))):
            M_eff['KGtw{0}'.format(twist)] += M_effs['KGtw{0}'.format(twist)][i]
            if M_effs['KGtw{0}'.format(twist)][i] != 0:
                denomKG +=1
            M_eff['KNGtw{0}'.format(twist)] += M_effs['KNGtw{0}'.format(twist)][i]
            if M_effs['KNGtw{0}'.format(twist)][i] != 0:
                denomKNG +=1
        M_eff['KGtw{0}'.format(twist)] = M_eff['KGtw{0}'.format(twist)]/denomKG
        M_eff['KNGtw{0}'.format(twist)] = M_eff['KNGtw{0}'.format(twist)]/denomKNG
        p = np.sqrt(3)*np.pi*float(twist)/Fit['L']
        M_effTheoryKG = gv.sqrt(M_eff['KGtw0']**2 + p**2)
        M_effTheoryKNG = gv.sqrt(M_eff['KNGtw0']**2 + p**2)        
        if abs(((M_eff['KGtw{0}'.format(twist)]-M_effTheoryKG)/M_effTheoryKG).mean) > 0.1:
            print('Substituted M_effTheory for KG twist',twist, 'Difference:',(M_eff['KGtw{0}'.format(twist)]-M_effTheoryKG)/M_effTheoryKG,'old:',M_eff['KGtw{0}'.format(twist)],'New:',M_effTheoryKG)
            M_eff['KGtw{0}'.format(twist)] = copy.deepcopy(M_effTheoryKG)
        if abs(((M_eff['KNGtw{0}'.format(twist)]-M_effTheoryKNG)/M_effTheoryKNG).mean) > 0.1:
            print('Substituted M_effTheory for KNG twist',twist, 'Difference:',(M_eff['KNGtw{0}'.format(twist)]-M_effTheoryKNG)/M_effTheoryKNG,'old:',M_eff['KNGtw{0}'.format(twist)],'New:',M_effTheoryKNG)
            M_eff['KNGtw{0}'.format(twist)] = copy.deepcopy(M_effTheoryKNG)    
    #print('M_eff',M_eff)        
    #plt.figure(2)
    for mass in Fit['masses']:
        for t in range(1,tp-2):
            numerator = data[TwoPts['BGm{0}'.format(mass)]][t]
            if numerator >= 0:
                A_effs['BGm{0}'.format(mass)].append(gv.sqrt(numerator/(gv.exp(-M_eff['BGm{0}'.format(mass)]*t)+gv.exp(-M_eff['BGm{0}'.format(mass)]*(tp-t)))))
            else:
                A_effs['BGm{0}'.format(mass)].append(0)
            numerator = data[TwoPts['BNGm{0}'.format(mass)]][t]
            if numerator >= 0:
                A_effs['BNGm{0}'.format(mass)].append(gv.sqrt(numerator/(gv.exp(-M_eff['BNGm{0}'.format(mass)]*t)+gv.exp(-M_eff['BNGm{0}'.format(mass)]*(tp-t)))))
            else:
                A_effs['BNGm{0}'.format(mass)].append(0)

                
    for twist in Fit['twists']:
        for t in range(1,tp-2):
            numerator = data[TwoPts['KGtw{0}'.format(twist)]][t]
            if numerator >= 0:
                A_effs['KGtw{0}'.format(twist)].append(gv.sqrt(numerator/(np.exp(-M_eff['KGtw{0}'.format(twist)]*t)+np.exp(-M_eff['KGtw{0}'.format(twist)]*(tp-t)))))
            else:
                A_effs['KGtw{0}'.format(twist)].append(0)
            numerator = data[TwoPts['KNGtw{0}'.format(twist)]][t]
            if numerator >= 0:
                A_effs['KNGtw{0}'.format(twist)].append(gv.sqrt(numerator/(np.exp(-M_eff['KNGtw{0}'.format(twist)]*t)+np.exp(-M_eff['KNGtw{0}'.format(twist)]*(tp-t)))))
            else:
                A_effs['KNGtw{0}'.format(twist)].append(0)
    #print('A',A_effs)          
    for mass in Fit['masses']:
        denomBG = 0
        denomBNG = 0
        for i in list(range(int(tp*(middle-gap)),int(tp*(middle+gap))))+list(range(int(tp*(1-middle-gap)),int(tp*(1-middle+gap)))):
            A_eff['BGm{0}'.format(mass)] += A_effs['BGm{0}'.format(mass)][i]
            if A_effs['BGm{0}'.format(mass)][i] != 0:
                denomBG += 1
            A_eff['BNGm{0}'.format(mass)] += A_effs['BNGm{0}'.format(mass)][i]
            if A_effs['BNGm{0}'.format(mass)][i] != 0:
                denomBNG += 1
        A_eff['BGm{0}'.format(mass)] = A_eff['BGm{0}'.format(mass)]/denomBG
        A_eff['BNGm{0}'.format(mass)] = A_eff['BNGm{0}'.format(mass)]/denomBNG
        if A_eff['BGm{0}'.format(mass)].sdev/A_eff['BGm{0}'.format(mass)].mean >0.5:
            print('BG Aeff mass {1} = {0} error too large so substituted'.format(A_eff['BGm{0}'.format(mass)],mass))
            A_eff['BGm{0}'.format(mass)]= gv.gvar(Fit['an'])
        if A_eff['BNGm{0}'.format(mass)].sdev/A_eff['BNGm{0}'.format(mass)].mean >0.2:
            print('BNG Aeff mass {1} = {0} error too large so substituted'.format(A_eff['BNGm{0}'.format(mass)],mass))
            A_eff['BNGm{0}'.format(mass)]= gv.gvar(Fit['an'])
    for twist in Fit['twists']:
        denomKG = 0
        denomKNG = 0
        for i in list(range(int(tp*(middle-gap)),int(tp*(middle+gap))))+list(range(int(tp*(1-middle-gap)),int(tp*(1-middle+gap)))):       
            A_eff['KGtw{0}'.format(twist)] += A_effs['KGtw{0}'.format(twist)][i]
            if A_effs['KGtw{0}'.format(twist)][i] != 0:
                denomKG += 1
            A_eff['KNGtw{0}'.format(twist)] += A_effs['KNGtw{0}'.format(twist)][i]
            if A_effs['KNGtw{0}'.format(twist)][i] != 0:
                denomKNG += 1
        A_eff['KGtw{0}'.format(twist)] = A_eff['KGtw{0}'.format(twist)]/denomKG
        A_eff['KNGtw{0}'.format(twist)] = A_eff['KNGtw{0}'.format(twist)]/denomKNG
        if A_eff['KGtw{0}'.format(twist)].sdev/A_eff['KGtw{0}'.format(twist)].mean > 0.2:
            print('KG Aeff twist {1} = {0} error too large so substituted'.format(A_eff['KGtw{0}'.format(twist)],twist))
            A_eff['KGtw{0}'.format(twist)] = gv.gvar(Fit['an'])
        if A_eff['KNGtw{0}'.format(twist)].sdev/A_eff['KNGtw{0}'.format(twist)].mean > 0.2:
            print('KNG Aeff twist {1} = {0} error too large so substituted'.format(A_eff['KNGtw{0}'.format(twist)],twist))
            A_eff['KNGtw{0}'.format(twist)] = gv.gvar(Fit['an'])
    #print('A_eff',A_eff)
            #plt.errorbar(t, A_effV[t-1].mean, yerr=A_effV[t-1].sdev, fmt='ko')
        #plt.errorbar(t, A_effS[t-1].mean, yerr=A_effS[t-1].sdev, fmt='ro')
    #plt.title('A_eff')
    #plt.savefig('AeffSf_m{0}_tw{1}'.format(mass,twist))
    #plt.show()
    #print(M_eff)    
    #plt.show()
    if ('S' or 'V' or 'T') in FitCorrs:
        print(FitCorrs)
        for mass in Fit['masses']:
            for twist in Fit['twists']:        
                V_effs['Sm{0}_tw{1}'.format(mass,twist)] = []
                V_effs['Vm{0}_tw{1}'.format(mass,twist)] = []
                V_effs['Tm{0}_tw{1}'.format(mass,twist)] = []
                V_eff['Sm{0}_tw{1}'.format(mass,twist)] = 0
                V_eff['Vm{0}_tw{1}'.format(mass,twist)] = 0
                V_eff['Tm{0}_tw{1}'.format(mass,twist)] = 0
                for t in range(T):
                    V_effs['Sm{0}_tw{1}'.format(mass,twist)].append(A_eff['BGm{0}'.format(mass)]*A_eff['KGtw{0}'.format(twist)]*data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['KGtw{0}'.format(twist)]][t]*data[TwoPts['BGm{0}'.format(mass)]][T-t]))
                    V_effs['Vm{0}_tw{1}'.format(mass,twist)].append(A_eff['BNGm{0}'.format(mass)]*A_eff['KGtw{0}'.format(twist)]*data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['KGtw{0}'.format(twist)]][t]*data[TwoPts['BNGm{0}'.format(mass)]][T-t]))
                    V_effs['Tm{0}_tw{1}'.format(mass,twist)].append(A_eff['BNGm{0}'.format(mass)]*A_eff['KNGtw{0}'.format(twist)]*data[ThreePts['Tm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['KNGtw{0}'.format(twist)]][t]*data[TwoPts['BNGm{0}'.format(mass)]][T-t]))
                

        for mass in Fit['masses']:
            for twist in Fit['twists']:           
                for t in range(int(T/2-T/5),int(T/2+T/5)):
                    V_eff['Sm{0}_tw{1}'.format(mass,twist)]  += (1/4)*V_effs['Sm{0}_tw{1}'.format(mass,twist)][t]
                    V_eff['Vm{0}_tw{1}'.format(mass,twist)]  += (1/4)*V_effs['Vm{0}_tw{1}'.format(mass,twist)][t]
                    V_eff['Tm{0}_tw{1}'.format(mass,twist)]  += (1/4)*V_effs['Tm{0}_tw{1}'.format(mass,twist)][t]
                if V_eff['Sm{0}_tw{1}'.format(mass,twist)] < 0.1 or V_eff['Sm{0}_tw{1}'.format(mass,twist)] > 1.5:
                    V_eff['Sm{0}_tw{1}'.format(mass,twist)] = gv.gvar('0.5(5)')
                if V_eff['Vm{0}_tw{1}'.format(mass,twist)] < 0.1 or V_eff['Vm{0}_tw{1}'.format(mass,twist)] > 1.5:
                    V_eff['Vm{0}_tw{1}'.format(mass,twist)] = gv.gvar('0.5(5)')
                if V_eff['Tm{0}_tw{1}'.format(mass,twist)] < 0.1 or V_eff['Tm{0}_tw{1}'.format(mass,twist)] > 1.5:
                    V_eff['Tm{0}_tw{1}'.format(mass,twist)] = gv.gvar('0.5(5)')
    #print(V_effs)
    #print(V_eff)
    return(M_eff,A_eff,V_eff)





def make_prior(All,N,M_eff,A_eff,V_eff,Autoprior):    
    Lambda = 0.5    ###Set Lambda_QCD in GeV
    an = '{0}({1})'.format(gv.gvar(Fit['an']).mean,PriorLoosener*gv.gvar(Fit['an']).sdev)
    SVn = '{0}({1})'.format(gv.gvar(Fit['SVn']).mean,PriorLoosener*gv.gvar(Fit['SVn']).sdev)
    SV0 = '{0}({1})'.format(gv.gvar(Fit['SV0']).mean,PriorLoosener*gv.gvar(Fit['SV0']).sdev)
    VVn = '{0}({1})'.format(gv.gvar(Fit['VVn']).mean,PriorLoosener*gv.gvar(Fit['VVn']).sdev)
    VV0 = '{0}({1})'.format(gv.gvar(Fit['VV0']).mean,PriorLoosener*gv.gvar(Fit['VV0']).sdev)
    TVn = '{0}({1})'.format(gv.gvar(Fit['TVn']).mean,PriorLoosener*gv.gvar(Fit['TVn']).sdev)
    TV0 = '{0}({1})'.format(gv.gvar(Fit['TV0']).mean,PriorLoosener*gv.gvar(Fit['TV0']).sdev)
    a = Fit['a']
    loosener = Fit['loosener']
    Mloosener = Fit['Mloosener']                 
    oMloosener = Fit['oMloosener']
    prior = gv.BufferDict()
    if CorrBayes == True:
        TwoKeys,ThreeKeys = makeKeys()
        for key in Autoprior:
            if key in TwoKeys:
                prior[key] = Autoprior[key]
                if key in ThreeKeys:
                    prior[key] = Autoprior[key]
    else:        
        if 'KG' in FitCorrs:
            for twist in twists:        
                # Daughter
                prior['log({0}:a)'.format(TwoPts['KGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(TwoPts['KGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                prior['log({0}:a)'.format(TwoPts['KGtw{0}'.format(twist)])][0] = gv.log(gv.gvar(A_eff['KGtw{0}'.format(twist)].mean,PriorLoosener*loosener*A_eff['KGtw{0}'.format(twist)].mean))
                prior['log(dE:{0})'.format(TwoPts['KGtw{0}'.format(twist)])][0] = gv.log(gv.gvar(M_eff['KGtw{0}'.format(twist)].mean,PriorLoosener*Mloosener*M_eff['KGtw{0}'.format(twist)].mean))
            #prior['log(etas:dE)'][1] = gv.log(gv.gvar(EtaE1[str(twist)]))
        
                # Daughter -- oscillating part
                #if twist!='0':                      
                prior['log(o{0}:a)'.format(TwoPts['KGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:o{0})'.format(TwoPts['KGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                #prior['log(o{0}:a)'.format(TwoPts['Dtw{0}'.format(twist)])][0] = gv.log(gv.gvar(A_eff['Dtw{0}'.format(twist)].mean,loosener*A_eff['Dtw{0}'.format(twist)].mean))
                prior['log(dE:o{0})'.format(TwoPts['KGtw{0}'.format(twist)])][0] = gv.log(gv.gvar(M_eff['KGtw{0}'.format(twist)].mean+Lambda*a,PriorLoosener*oMloosener*(M_eff['KGtw{0}'.format(twist)].mean+Lambda*a)))
        if 'KNG' in FitCorrs:
            for twist in twists:        
                # Daughter
                prior['log({0}:a)'.format(TwoPts['KNGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(TwoPts['KNGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                prior['log({0}:a)'.format(TwoPts['KNGtw{0}'.format(twist)])][0] = gv.log(gv.gvar(A_eff['KNGtw{0}'.format(twist)].mean,PriorLoosener*loosener*A_eff['KNGtw{0}'.format(twist)].mean))
                prior['log(dE:{0})'.format(TwoPts['KNGtw{0}'.format(twist)])][0] = gv.log(gv.gvar(M_eff['KNGtw{0}'.format(twist)].mean,PriorLoosener*Mloosener*M_eff['KNGtw{0}'.format(twist)].mean))
            #prior['log(etas:dE)'][1] = gv.log(gv.gvar(EtaE1[str(twist)]))
        
                # Daughter -- oscillating part
                #if twist!='0':                      
                prior['log(o{0}:a)'.format(TwoPts['KNGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:o{0})'.format(TwoPts['KNGtw{0}'.format(twist)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                #prior['log(o{0}:a)'.format(TwoPts['Dtw{0}'.format(twist)])][0] = gv.log(gv.gvar(A_eff['Dtw{0}'.format(twist)].mean,loosener*A_eff['Dtw{0}'.format(twist)].mean))
                prior['log(dE:o{0})'.format(TwoPts['KNGtw{0}'.format(twist)])][0] = gv.log(gv.gvar(M_eff['KNGtw{0}'.format(twist)].mean+Lambda*a,PriorLoosener*oMloosener*(M_eff['KNGtw{0}'.format(twist)].mean+Lambda*a)))
           
        if 'BG' in FitCorrs:
            for mass in masses:
                # Goldstone
                prior['log({0}:a)'.format(TwoPts['BGm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(TwoPts['BGm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                prior['log({0}:a)'.format(TwoPts['BGm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['BGm{0}'.format(mass)].mean,PriorLoosener*loosener*A_eff['BGm{0}'.format(mass)].mean))
                prior['log(dE:{0})'.format(TwoPts['BGm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['BGm{0}'.format(mass)].mean,PriorLoosener*Mloosener*M_eff['BGm{0}'.format(mass)].mean))
        

                # Goldstone -- oscillating part
                prior['log(o{0}:a)'.format(TwoPts['BGm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:o{0})'.format(TwoPts['BGm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                #prior['log(o{0}:a)'.format(TwoPts['Gm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['Gm{0}'.format(mass)].mean,loosener*A_eff['Gm{0}'.format(mass)].mean))
                prior['log(dE:o{0})'.format(TwoPts['BGm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['BGm{0}'.format(mass)].mean+Lambda*a,PriorLoosener*oMloosener*(M_eff['BGm{0}'.format(mass)].mean+Lambda*a)))
        if 'BNG' in FitCorrs:
            for mass in masses:
                # Non-Goldstone
                prior['log({0}:a)'.format(TwoPts['BNGm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(TwoPts['BNGm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                #prior['log({0}:a)'.format(TwoPts['NGm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['NGm{0}'.format(mass)].mean,loosener*A_eff['NGm{0}'.format(mass)].mean))
                prior['log(dE:{0})'.format(TwoPts['BNGm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['BNGm{0}'.format(mass)].mean,PriorLoosener*Mloosener*M_eff['BNGm{0}'.format(mass)].mean))
        

                # Non-Goldstone -- oscillating part
                prior['log(o{0}:a)'.format(TwoPts['BNGm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:o{0})'.format(TwoPts['BNGm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,PriorLoosener*0.5*Lambda*a)]))
                #prior['log(o{0}:a)'.format(TwoPts['NGm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['NGm{0}'.format(mass)].mean,loosener*A_eff['NGm{0}'.format(mass)].mean))
                prior['log(dE:o{0})'.format(TwoPts['BNGm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['BNGm{0}'.format(mass)].mean+Lambda*a,PriorLoosener*oMloosener*(M_eff['BNGm{0}'.format(mass)].mean+Lambda*a)))
    if All == True:
        if 'S' in FitCorrs:
            for mass in masses: 
                for twist in twists:
                    if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                        prior['SVnn_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [SVn]])
                        prior['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V_eff['Sm{0}_tw{1}'.format(mass,twist)].mean,PriorLoosener*loosener*V_eff['Sm{0}_tw{1}'.format(mass,twist)].mean)
                        prior['SVno_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [SVn]])
                        prior['SVno_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(SV0)
                        #if twist != '0':                    
                        prior['SVon_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [SVn]])
                        prior['SVon_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(SV0)
                        prior['SVoo_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [SVn]])
                        prior['SVoo_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(SV0)
        if 'V' in FitCorrs:
            for mass in masses:
                for twist in twists:
                    if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                        prior['VVnn_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [VVn]])
                        prior['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V_eff['Vm{0}_tw{1}'.format(mass,twist)].mean,PriorLoosener*loosener*V_eff['Vm{0}_tw{1}'.format(mass,twist)].mean)
                        prior['VVno_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [VVn]])
                        prior['VVno_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(VV0)
                        #if twist != '0':                    
                        prior['VVon_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [VVn]])
                        prior['VVon_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(VV0)
                        prior['VVoo_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [VVn]])
                        prior['VVoo_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(VV0)
        if 'T' in FitCorrs:
            for mass in masses:
                for twist in twists:
                    if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                        prior['TVnn_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [TVn]])
                        prior['TVnn_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V_eff['Tm{0}_tw{1}'.format(mass,twist)].mean,PriorLoosener*loosener*V_eff['Tm{0}_tw{1}'.format(mass,twist)].mean)
                        prior['TVno_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [TVn]])
                        prior['TVno_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(TV0)
                        #if twist != '0':                    
                        prior['TVon_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [TVn]])
                        prior['TVon_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(TV0)
                        prior['TVoo_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [TVn]])
                        prior['TVoo_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(TV0)
    return(prior)



def make_models():
    print('Masses:',masses,'Twists:',twists, 'Ts:',Ts,'Corrs:',FitCorrs)
    """ Create models to fit data. """
    tminBG = Fit['tminBG']
    tminBNG = Fit['tminBNG']
    tminKG = Fit['tminKG']
    tminKNG = Fit['tminKNG']
    #tmaxG = Fit['tmaxG']
    #tmaxNG = Fit['tmaxNG']
    #tmaxD = Fit['tmaxD']
    Stmin = Fit['Stmin']
    Vtmin = Fit['Vtmin']
    Ttmin = Fit['Ttmin']
    tp = Fit['tp']
    twopts  = []
    Sthreepts = []
    Vthreepts = []
    Tthreepoints = []
    if 'BG' in FitCorrs:
        for i,mass in enumerate(masses):        
            BGCorrelator = copy.deepcopy(TwoPts['BGm{0}'.format(mass)])
            twopts.append(cf.Corr2(datatag=BGCorrelator, tp=tp, tmin=tminBG, tmax=tmaxesBG[i], a=('{0}:a'.format(BGCorrelator), 'o{0}:a'.format(BGCorrelator)), b=('{0}:a'.format(BGCorrelator), 'o{0}:a'.format(BGCorrelator)), dE=('dE:{0}'.format(BGCorrelator), 'dE:o{0}'.format(BGCorrelator)),s=(1.,-1.)))
    if 'BNG' in FitCorrs:
        for i,mass in enumerate(masses):
            BNGCorrelator = copy.deepcopy(TwoPts['BNGm{0}'.format(mass)])
            twopts.append(cf.Corr2(datatag=BNGCorrelator, tp=tp, tmin=tminBNG, tmax=tmaxesBNG[i], a=('{0}:a'.format(BNGCorrelator), 'o{0}:a'.format(BNGCorrelator)), b=('{0}:a'.format(BNGCorrelator), 'o{0}:a'.format(BNGCorrelator)), dE=('dE:{0}'.format(BNGCorrelator), 'dE:o{0}'.format(BNGCorrelator)),s=(1.,-1.)))
    if 'KG' in FitCorrs:
        for i,twist in enumerate(twists):
            KGCorrelator = copy.deepcopy(TwoPts['KGtw{0}'.format(twist)])
            #if twist != '0':                
            twopts.append(cf.Corr2(datatag=KGCorrelator, tp=tp, tmin=tminKG, tmax=tmaxesKG[i],a=('{0}:a'.format(KGCorrelator), 'o{0}:a'.format(KGCorrelator)), b=('{0}:a'.format(KGCorrelator), 'o{0}:a'.format(KGCorrelator)), dE=('dE:{0}'.format(KGCorrelator), 'dE:o{0}'.format(KGCorrelator)),s=(1.,-1.)))
    if 'KNG' in FitCorrs:
        for i,twist in enumerate(twists):
            KNGCorrelator = copy.deepcopy(TwoPts['KNGtw{0}'.format(twist)])
            #if twist != '0':                
            twopts.append(cf.Corr2(datatag=KNGCorrelator, tp=tp, tmin=tminKNG, tmax=tmaxesKNG[i],a=('{0}:a'.format(KNGCorrelator), 'o{0}:a'.format(KNGCorrelator)), b=('{0}:a'.format(KNGCorrelator), 'o{0}:a'.format(KNGCorrelator)), dE=('dE:{0}'.format(KNGCorrelator), 'dE:o{0}'.format(KNGCorrelator)),s=(1.,-1.)))
            
                
    if 'S' in FitCorrs:
        for mass in masses:
            BGCorrelator = copy.deepcopy(TwoPts['BGm{0}'.format(mass)])
            BNGCorrelator = copy.deepcopy(TwoPts['BNGm{0}'.format(mass)])
            for twist in twists:
                KGCorrelator = copy.deepcopy(TwoPts['KGtw{0}'.format(twist)])
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    for T in Ts:
                        #if twist != '0':
                        Sthreepts.append(cf.Corr3(datatag=ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)], T=T, tmin=Stmin,  a=('{0}:a'.format(KGCorrelator), 'o{0}:a'.format(KGCorrelator)), dEa=('dE:{0}'.format(KGCorrelator), 'dE:o{0}'.format(KGCorrelator)), sa=(1,-1), b=('{0}:a'.format(BGCorrelator), 'o{0}:a'.format(BGCorrelator)), dEb=('dE:{0}'.format(BGCorrelator), 'dE:o{0}'.format(BGCorrelator)), sb=(1,-1), Vnn='SVnn_m'+str(mass)+'_tw'+str(twist), Vno='SVno_m'+str(mass)+'_tw'+str(twist), Von='SVon_m'+str(mass)+'_tw'+str(twist), Voo='SVoo_m'+str(mass)+'_tw'+str(twist)))
                        
                        
    if 'V' in FitCorrs:
        for mass in masses:
            BGCorrelator = copy.deepcopy(TwoPts['BGm{0}'.format(mass)])
            BNGCorrelator = copy.deepcopy(TwoPts['BNGm{0}'.format(mass)])
            for twist in twists:
                KGCorrelator = copy.deepcopy(TwoPts['KGtw{0}'.format(twist)])
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    for T in Ts:                    
                        #if twist != '0':
                        Vthreepts.append(cf.Corr3(datatag=ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)], T=T, tmin=Vtmin, a=('{0}:a'.format(KGCorrelator), 'o{0}:a'.format(KGCorrelator)), dEa=('dE:{0}'.format(KGCorrelator), 'dE:o{0}'.format(KGCorrelator)), sa=(1,-1), b=('{0}:a'.format(BNGCorrelator), 'o{0}:a'.format(BNGCorrelator)), dEb=('dE:{0}'.format(BNGCorrelator), 'dE:o{0}'.format(BNGCorrelator)), sb=(1,-1), Vnn='VVnn_m'+str(mass)+'_tw'+str(twist), Vno='VVno_m'+str(mass)+'_tw'+str(twist), Von='VVon_m'+str(mass)+'_tw'+str(twist), Voo='VVoo_m'+str(mass)+'_tw'+str(twist)))

    if 'T' in FitCorrs:
        for mass in masses:
            BGCorrelator = copy.deepcopy(TwoPts['BGm{0}'.format(mass)])
            BNGCorrelator = copy.deepcopy(TwoPts['BNGm{0}'.format(mass)])
            for twist in twists:
                KNGCorrelator = copy.deepcopy(TwoPts['KNGtw{0}'.format(twist)])
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    for T in Ts:                    
                        #if twist != '0':
                        Tthreepts.append(cf.Corr3(datatag=ThreePts['Tm{0}_tw{1}_T{2}'.format(mass,twist,T)], T=T, tmin=Ttmin, a=('{0}:a'.format(KNGCorrelator), 'o{0}:a'.format(KNGCorrelator)), dEa=('dE:{0}'.format(KNGCorrelator), 'dE:o{0}'.format(KNGCorrelator)), sa=(1,-1), b=('{0}:a'.format(BNGCorrelator), 'o{0}:a'.format(BNGCorrelator)), dEb=('dE:{0}'.format(BNGCorrelator), 'dE:o{0}'.format(BNGCorrelator)), sb=(1,-1), Vnn='TVnn_m'+str(mass)+'_tw'+str(twist), Vno='TVno_m'+str(mass)+'_tw'+str(twist), Von='TVon_m'+str(mass)+'_tw'+str(twist), Voo='TVoo_m'+str(mass)+'_tw'+str(twist)))
                        
    if Chained == True:            
        twopts = twopts
        threepts = []
        for element in range(len(Sthreepts)):
            threepts.append(Sthreepts[element])
        for element in range(len(Vthreepts)):
            threepts.append(Vthreepts[element])
        for element in range(len(Tthreepts)):
            threepts.append(Tthreepts[element])
        return(twopts,threepts)
    else:
        twopts.extend(Sthreepts)
        twopts.extend(Vthreepts)
        twopts.extend(Tthreepts)
        return(twopts)

    
    
def modelsandsvd():
    if Chained == True:     
        twopts,threepts = make_models()
        models = [twopts, threepts]
        File2 = 'Ps/Chain2pts{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}.pickle'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['tminG'],Fit['tminNG'],Fit['tminD'],tmaxesG,tmaxesNG,tmaxesD,FitAllTwists)
        File3 = 'Ps/Chain3pts{0}{1}{2}{3}{4}{5}{6}{7}.pickle'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['Stmin'],Fit['Vtmin'],FitAllTwists)
    else:
        models = make_models()   
    print('Models made: ', models)
    File = 'Ps/{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}.pickle'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['Stmin'],Fit['Vtmin'],Fit['tminG'],Fit['tminNG'],Fit['tminD'],tmaxesG,tmaxesNG,tmaxesD,Chained,FitAllTwists)
    if AutoSvd == True:
        if Chained == True:
            #####################################CHAINED########################################
            if os.path.isfile(File2) == True and recalculateSvd == False:
                pickle_off = open(File2,"rb")
                trueSvd = pickle.load(pickle_off)
                svdcut2 = SvdFactor*trueSvd
                print('Used existing svdcut for twopoints {0} times factor {1}:'.format(trueSvd,SvdFactor), svdcut2)
            else:
                print('Calculating svd')
                s = gv.dataset.svd_diagnosis(cf.read_dataset(filename), models=models[0], nbstrap=20)
                s.plot_ratio(show=True)
                var = input("Hit enter to accept svd for twopoints = {0}, or else type svd here:".format(s.svdcut))
                if var == '':
                    trueSvd = s.svdcut
                    svdcut2 = trueSvd*SvdFactor
                    print('Used calculated svdcut for twopoints {0}, times factor {1}:'.format(trueSvd,SvdFactor),svdcut2)
                else:
                    trueSvd = float(var)
                    svdcut2 = SvdFactor*trueSvd
                    print('Used alternative svdcut for twopoints {0}, times factor {1}:'.format(float(var),SvdFactor), svdcut2)                
                pickling_on = open(File2, "wb")
                pickle.dump(trueSvd,pickling_on)
                pickling_on.close()
            if os.path.isfile(File3) == True and recalculateSvd == False:
                pickle_off = open(File3,"rb")
                trueSvd = pickle.load(pickle_off)
                svdcut3 = SvdFactor*trueSvd
                print('Used existing svdcut for threepoints {0} times factor {1}:'.format(trueSvd,SvdFactor), svdcut3)
            else:
                print('Calculating svd')
                s = gv.dataset.svd_diagnosis(cf.read_dataset(filename), models=models[1], nbstrap=20)
                s.plot_ratio(show=True)
                var = input("Hit enter to accept svd for threepoints = {0}, or else type svd here:".format(s.svdcut))
                if var == '':
                    trueSvd = s.svdcut
                    svdcut3 = trueSvd*SvdFactor
                    print('Used calculated svdcut for threepoints {0}, times factor {1}:'.format(trueSvd,SvdFactor),svdcut3)
                else:
                    trueSvd = float(var)
                    svdcut3 = SvdFactor*trueSvd
                    print('Used alternative svdcut for threepoints {0}, times factor {1}:'.format(float(var),SvdFactor), svdcut3)                
                pickling_on = open(File3, "wb")
                pickle.dump(trueSvd,pickling_on)
                pickling_on.close()
            svdcut=[svdcut2,svdcut3]
        #####################################UNCHAINED########################################        
        else:
            if os.path.isfile(File) == True and recalculateSvd == False:
                pickle_off = open(File,"rb")
                trueSvd = pickle.load(pickle_off)
                svdcut = SvdFactor*trueSvd
                print('Used existing svdcut {0} times factor {1}:'.format(trueSvd,SvdFactor), svdcut)
            else:
                print('Calculating svd')
                s = gv.dataset.svd_diagnosis(cf.read_dataset(filename), models=models, nbstrap=20)
                s.plot_ratio(show=True)
                var = input("Hit enter to accept svd = {0}, or else type svd here:".format(s.svdcut))
                if var == '':
                    trueSvd = s.svdcut
                    svdcut = trueSvd*SvdFactor
                    print('Used calculated svdcut {0}, times factor {1}:'.format(trueSvd,SvdFactor),svdcut)
                else:
                    trueSvd = float(var)
                    svdcut = SvdFactor*trueSvd
                    print('Used alternative svdcut {0}, times factor {1}:'.format(float(var),SvdFactor), svdcut)                
                pickling_on = open(File, "wb")
                pickle.dump(trueSvd,pickling_on)
                pickling_on.close()
    else:
        if os.path.isfile(File) == True and recalculateSvd == False:
            pickle_off = open(File,"rb")
            previous = pickle.load(pickle_off)
            var = input('Hit enter to use previously chosen svd {0}, times factor {1} or type new one:'.format(previous,SvdFactor))
            if var == '':
                trueSvd =  previous
                svdcut = SvdFactor*trueSvd
            else:
                trueSvd = float(var)
                svdcut = SvdFactor*trueSvd               
        else:
            var = input('Type new svd:')
            trueSvd = float(var)
            svdcut = SvdFactor*trueSvd
        print('Using svdcut {0}, times factor {1}:'.format(trueSvd,SvdFactor),svdcut)
        pickling_on = open(File, "wb")
        pickle.dump(trueSvd,pickling_on)
        pickling_on.close()
    return(models,svdcut)

        
def main(Autoprior,data): 
    M_eff,A_eff,V_eff = eff_calc()
    TwoKeys,ThreeKeys = makeKeys()
    p0 = collections.OrderedDict()
######################### CHAINED ################################### 
    if Chained == True:
        Nexp = 3
        GBF1 = -1e21
        GBF2 = -1e20
        bothmodels,bothsvds = modelsandsvd()
        models = bothmodels[0]
        svdcut = bothsvds[0]
        Fitter = cf.CorrFitter(models=models, svdcut=svdcut, fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, fast=False, tol=(1e-6,0.0,0.0))
        if FitToGBF == True:
            cond = (lambda: Nexp <= 8) if FitAll else (lambda: GBF2 - GBF1 > 0.01)
            cond2 = (lambda: Nexp <= Nmax) if Marginalised else (lambda: GBF2 - GBF1 > 0.01)
        else:
            cond = (lambda: Nexp <= 8) if FitAll else (lambda: Nexp <= Nmax)
            cond2 = (lambda: Nexp <= 8) if FitAll else (lambda: Nexp <= Nmax)
        while cond() and cond2():           
            fname2 = 'Ps/Chain2pts{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['tminG'],Fit['tminNG'],Fit['tminD'],tmaxesG,tmaxesNG,tmaxesD,FitAllTwists)            
            p0 = load_p0(p0,Nexp,fname2,TwoKeys,[])                    
            GBF1 = copy.deepcopy(GBF2)
            print('Making Prior')
            if CorrBayes == True:
                Autoprior,data = make_data(filename,Nexp)       
            prior = make_prior(False,Nexp,M_eff,A_eff,V_eff,Autoprior)
            print(30 * '=','Chained','Nexp =',Nexp,'Date',datetime.datetime.now())
            fit = Fitter.lsqfit(data=data, prior=prior,  p0=p0, add_svdnoise=svdnoise, add_priornoise=priornoise)            
            #print('Key, Fit, p0, Prior' )
            #for key in p0:
            #    print(key,fit.p[key],p0[key],prior[key])
            GBF2 = fit.logGBF
            if FitToGBF == True:
                cond = (lambda: Nexp <= 8) if FitAll else (lambda: GBF2 - GBF1 > 0.01)
            else:
                cond = (lambda: Nexp <= 8) if FitAll else (lambda: Nexp <= Nmax)
            if fit.Q>=0.05:
                pickling_on = open('{0}{1}.pickle'.format(fname2,Nexp), "wb")
                pickle.dump(fit.pmean,pickling_on)
                pickling_on.close()
            if cond():
                for2results = fit.p
                if FitAll == False:
                    if ResultPlots == 'Q':
                        plots(fit.Q,fit.p,Nexp)
                    if ResultPlots == 'GBF':
                        plots(GBF2,fit.p,Nexp)
                    if ResultPlots == 'N':
                        plots(Nexp,fit.p,fit.Q)
                print(fit)
                #print(fit.format(pstyle=None if Nexp<3 else'v'))
                print('Nexp = ',Nexp)
                print('Q = {0:.2f}'.format(fit.Q))
                print('log(GBF) = {0:.2f}, up {1:.2f}'.format(GBF2,GBF2-GBF1))       
                print('chi2/dof = {0:.2f}'.format(fit.chi2/fit.dof))
                print('dof =', fit.dof)
                print('SVD noise = {0} Prior noise = {1}'.format(svdnoise,priornoise))
                if fit.Q >= 0.05:
                    p0=fit.pmean
                    save_p0(p0,Nexp,fname2,TwoKeys,[])
                    if SaveFit == True:
                        gv.dump(fit.p,'Fits/{5}5_2pts_Q{4:.2f}_Nexp{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof))
                        f = open('Fits/{5}5_2pts_Q{4:.2f}_Nexp{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}.txt'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof), 'w')
                        f.write(fit.format(pstyle=None if Nexp<3 else'v'))
                        f.close()
            else:
                print('log(GBF) had gone down by {2:.2f} from {0:.2f} to {1:.2f}'.format(GBF1,GBF2,GBF1-GBF2))                
            print(100 * '+')
            print(100 * '+')
            Nexp += 1
        if GBF2 - GBF1 <= 0.01 :    
            NMarg = Nexp-2
        else:
            NMarg = Nexp-1
        if FitToGBF == False:
            NMarg = Nexp-1
        print(100*'=')
        print(100*'=','MOVING TO 3 POINTS')
        print(100*'=')
        ###################################THREEPOINTS###############################################
        if Marginalised == False:
            Nexp = 2
        else:
            Nexp = 1
        GBF1 = -1e21
        GBF2 = -1e20
        models = bothmodels[1]
        svdcut = bothsvds[1]
        Fitter = cf.CorrFitter(models=models, svdcut=svdcut, fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, fast=False, tol=(1e-6,0.0,0.0))            
        cond = (lambda: Nexp <= 8) if FitAll else (lambda: GBF2 - GBF1 > 0.01)
        #cond2 = (lambda: Nexp <= 8) if Marginalised else (lambda: GBF2 - GBF1 > 0.01)
        while cond():# and cond2():           
            fname3 = 'Ps/Chain3pts{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['Stmin'],Fit['Vtmin'],FitAllTwists,Marginalised)
            if Marginalised == False:
                p0 = load_p0(p0,Nexp,fname3,TwoKeys,ThreeKeys)
            else:
                p0 = load_p0(p0,Nexp,fname3,TwoKeys,ThreeKeys)   
            GBF1 = copy.deepcopy(GBF2)
            print('Making Prior')
            if CorrBayes == True:
                if Marginalised == False:
                     Autoprior,data = make_data(filename,Nexp)
                else:
                    Autoprior,data = make_data(filename,NMarg)
            if Marginalised == False:
                prior = make_prior(True,Nexp,M_eff,A_eff,V_eff,Autoprior)
            else:
                prior = make_prior(True,NMarg,M_eff,A_eff,V_eff,Autoprior)
            if Marginalised == False: 
                for key in for2results:
                    for n in range(Nexp):
                        if n < len(for2results[key]):
                            prior[key][n] = for2results[key][n]
            else:                                                    
                for key in for2results:
                    for n in range(NMarg):
                        if n < len(for2results[key]):
                            prior[key][n] = for2results[key][n]
            if Marginalised == False:
                print(30 * '=','Chained','Nexp =',Nexp,'Date',datetime.datetime.now())
            else:
                print(30 * '=','Chained and Marginalised','NMarg =',NMarg, 'nterm =','({0},{0})'.format(Nexp),'Date',datetime.datetime.now())
            if Marginalised == False:
                fit = Fitter.lsqfit(data=data, prior=prior,  p0=p0, add_svdnoise=svdnoise, add_priornoise=priornoise)
            else:
                fit = Fitter.lsqfit(data=data, prior=prior,  p0=p0, nterm=(Nexp,Nexp), add_svdnoise=svdnoise, add_priornoise=priornoise)
            GBF2 = fit.logGBF
            cond = (lambda: Nexp <= 8) if FitAll else (lambda: GBF2 - GBF1 > 0.01)
            if fit.Q>=0.05:
                pickling_on = open('{0}{1}.pickle'.format(fname3,Nexp), "wb")
                pickle.dump(fit.pmean,pickling_on)
                pickling_on.close()
            if cond():# and cond2():
                for3results = fit.p
                if FitAll == False:
                    if ResultPlots == 'Q':
                        plots(fit.Q,fit.p,Nexp)
                    if ResultPlots == 'GBF':
                        plots(GBF2,fit.p,Nexp)
                    if ResultPlots == 'N':
                        plots(Nexp,fit.p,fit.Q)
                print(fit)
                #print(fit.format(pstyle=None if Nexp<3 else'v'))
                if Marginalised == False:
                    print('Nexp = ',Nexp)
                else:
                    print('NMarg = ',NMarg, 'nterm =', '({0},{0})'.format(Nexp))
                print('Q = {0:.2f}'.format(fit.Q))
                print('log(GBF) = {0:.2f}, up {1:.2f}'.format(GBF2,GBF2-GBF1))       
                print('chi2/dof = {0:.2f}'.format(fit.chi2/fit.dof))
                print('dof =', fit.dof)
                print('SVD noise = {0} Prior noise = {1}'.format(svdnoise,priornoise))
                if fit.Q >= 0.05:
                    p0=fit.pmean
                    if Marginalised == False:
                        save_p0(p0,Nexp,fname3,TwoKeys,ThreeKeys)
                    #else:
                    #    save_p0(p0,NMarg,fname3,TwoKeys,ThreeKeys)      #Don't save gloabal elements if margianlised
                    if SaveFit == True:
                        gv.dump(fit.p,'Fits/{5}5_3pts_Q{4:.2f}_Nexp{0}_NMarg{7}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}_pl{8}_svdfac{9}'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof,NMarg,PriorLoosener,SvdFactor))
                        f = open('Fits/{5}5_3pts_Q{4:.2f}_Nexp{0}_NMarg{7}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}_pl{8}_svdfac{9}.txt'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof,NMarg,PriorLoosener,SvdFactor), 'w')
                        f.write(fit.format(pstyle=None if Nexp<3 else'v'))
                        f.close()
                        
            else:
                print('log(GBF) had gone down by {2:.2f} from {0:.2f} to {1:.2f}'.format(GBF1,GBF2,GBF1-GBF2))                
            print(100 * '+')
            print(100 * '+')
            Nexp += 1    
        #print_results(for2results)
        print_results(for3results)

                
                
            
########################## Unchained ######################################                
    else:
        #print('Initial p0', p0)        
        Nexp = 3
        if ('S' or 'V') in FitCorrs:
            Nexp = 2
        GBF1 = -1e21
        GBF2 = -1e20
        models,svdcut = modelsandsvd()
        Fitter = cf.CorrFitter(models=models, svdcut=svdcut, fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, fast=False, tol=(1e-6,0.0,0.0))
        if FitToGBF == True:
            cond = (lambda: Nexp <= 8) if FitAll else (lambda: GBF2 - GBF1 > 0.01)
        else:
            cond = (lambda: Nexp <= 8) if FitAll else (lambda: Nexp <= Nmax)
        while cond():           
            fname = 'Ps/{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['Stmin'],Fit['Vtmin'],Fit['tminG'],Fit['tminNG'],Fit['tminD'],tmaxesG,tmaxesNG,tmaxesD,Chained,FitAllTwists)            
            p0 = load_p0(p0,Nexp,fname,TwoKeys,ThreeKeys)                    
            GBF1 = copy.deepcopy(GBF2)
            print('Making Prior')
            if CorrBayes == True:
                Autoprior,data = make_data(filename,Nexp)       
            prior = make_prior(True,Nexp,M_eff,A_eff,V_eff,Autoprior)
            print(30 * '=','Unchained-Unmarginalised','Nexp =',Nexp,'Date',datetime.datetime.now())            
            fit = Fitter.lsqfit(data=data, prior=prior,  p0=p0, add_svdnoise=svdnoise, add_priornoise=priornoise)            
            GBF2 = fit.logGBF
            if FitToGBF == True:
                cond = (lambda: Nexp <= 8) if FitAll else (lambda: GBF2 - GBF1 > 0.01)
            else:
                cond = (lambda: Nexp <= 8) if FitAll else (lambda: Nexp <= Nmax)
            if fit.Q>=0.05:
                pickling_on = open('{0}{1}.pickle'.format(fname,Nexp), "wb")
                pickle.dump(fit.pmean,pickling_on)
                pickling_on.close()
            if cond():
                forresults = fit.p
                if FitAll == False:
                    if ResultPlots == 'Q':
                        plots(fit.Q,fit.p,Nexp)
                    if ResultPlots == 'GBF':
                        plots(GBF2,fit.p,Nexp)
                    if ResultPlots == 'N':
                        plots(Nexp,fit.p,fit.Q)
                print(fit)
                #print(fit.format(pstyle=None if Nexp<3 else'v'))
                print('Nexp = ',Nexp)
                print('Q = {0:.2f}'.format(fit.Q))
                print('log(GBF) = {0:.2f}, up {1:.2f}'.format(GBF2,GBF2-GBF1))       
                print('chi2/dof = {0:.2f}'.format(fit.chi2/fit.dof))
                print('dof =', fit.dof)
                print('SVD noise = {0} Prior noise = {1}'.format(svdnoise,priornoise))
                if fit.Q >= 0.05:
                    p0=fit.pmean
                    save_p0(p0,Nexp,fname,TwoKeys,ThreeKeys)
                    if SaveFit == True:
                        gv.dump(fit.p,'Fits/{5}5_Q{4:.2f}_Nexp{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof))
                        f = open('Fits/{5}5_Q{4:.2f}_Nexp{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}.txt'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof), 'w')
                        f.write(fit.format(pstyle=None if Nexp<3 else'v'))
                        f.close()
            else:
                print('log(GBF) had gone down by {2:.2f} from {0:.2f} to {1:.2f}'.format(GBF1,GBF2,GBF1-GBF2))                
            print(100 * '+')
            print(100 * '+')
            Nexp += 1           
        print_results(forresults)
        if len(FitCorrs)==1:
            fit.show_plots(view='ratio')
    return()






def plots(Q,p,Nexp):
    if ResultPlots == 'Q':
        xlab = 'Q'
        lab = 'N'       
    if ResultPlots == 'GBF':
        xlab = 'Log(GBF)'
        lab = 'N'
    if ResultPlots == 'N':
        xlab = 'N'
        lab = 'Q'
    for twist in twists:
        if 'D' in FitCorrs:
            result = p['dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)])][0]
            y = result.mean
            err = result.sdev    
            plt.figure(twist)
            plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('{0} = {1:.2f}'.format(lab,Nexp)))
            plt.legend()
            plt.xlabel('{0}'.format(xlab))
            plt.ylabel('dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)]))
    for mass in masses:
        if 'G' in FitCorrs:
            result = p['dE:{0}'.format(TwoPts['Gm{0}'.format(mass)])][0]
            y = result.mean
            err = result.sdev    
            plt.figure(mass)
            plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('G {0} = {1:.2f}'.format(lab,Nexp)))
            plt.legend()
            plt.xlabel('{0}'.format(xlab))
            plt.ylabel('dE:{0}'.format(mass))
        if 'NG' in FitCorrs:
            result = p['dE:{0}'.format(TwoPts['NGm{0}'.format(mass)])][0]
            y = result.mean
            err = result.sdev    
            plt.figure(mass)
            plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('NG {0} = {1:.2f}'.format(lab,Nexp)))
            plt.legend()
            plt.xlabel('{0}'.format(xlab))
            plt.ylabel('dE:{0}'.format(mass))
            for twist in twists:
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    if 'S' in FitCorrs:                    
                        result = p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
                        y = result.mean
                        err = result.sdev    
                        plt.figure('SVnn_m{0}_tw{1}'.format(mass,twist))
                        plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('{0} = {1:.2f}'.format(lab,Nexp)))
                        plt.legend()
                        plt.xlabel('{0}'.format(xlab))
                        plt.ylabel('SVnn_m{0}_tw{1}'.format(mass,twist))
                    if 'V' in FitCorrs:
                        result = p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
                        y = result.mean
                        err = result.sdev    
                        plt.figure('VVnn_m{0}_tw{1}'.format(mass,twist))
                        plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('{0} = {1:.2f}'.format(lab,Nexp)))
                        plt.legend()
                        plt.xlabel('{0}'.format(xlab))
                        plt.ylabel('VVnn_m{0}_tw{1}'.format(mass,twist))
    return()


def test_data():
    tp = Fit['tp']
    if 'D' in FitCorrs:
        for i,twist in enumerate(twists):
            plt.figure('log{0}'.format(twist))
            for t in range(1,int(tp/2+1)):                
                plt.errorbar(t, gv.log((data[TwoPts['Dtw{0}'.format(twist)]][t]+data[TwoPts['Dtw{0}'.format(twist)]][tp-t])/2).mean, yerr=gv.log((data[TwoPts['Dtw{0}'.format(twist)]][t]+data[TwoPts['Dtw{0}'.format(twist)]][tp-t])/2).sdev, fmt='ko')
            plt.title('D Twist = {0}'.format(twist))
            plt.xlabel('t',fontsize=20)
            plt.ylabel('log(2-point)',fontsize=20)
        #plt.legend()
            #lim = plt.ylim()
            #plt.plot([Fit['tminD'],Fit['tminD']],[lim[0],lim[1]],'k-',linewidth=2.5)
            #plt.plot([Fit['tmaxesD'][i],Fit['tmaxesD'][i]],[lim[0],lim[1]],'k-',linewidth=2.5)
            #plt.plot([int(3*tp/8-tp*gap),int(3*tp/8-tp*gap)],[lim[0],lim[1]],'k--',linewidth=2.5)
            #plt.plot([int(3*tp/8+tp*gap),int(3*tp/8+tp*gap)],[lim[0],lim[1]],'k--',linewidth=2.5)       
    if 'G' in FitCorrs:                
        for i,mass in enumerate(masses):
            plt.figure('log{0}'.format(mass))
            for t in range(1,int(tp/2+1)):                
                plt.errorbar(t, gv.log((data[TwoPts['Gm{0}'.format(mass)]][t]+data[TwoPts['Gm{0}'.format(mass)]][tp-t])/2).mean, yerr=gv.log((data[TwoPts['Gm{0}'.format(mass)]][t]+data[TwoPts['Gm{0}'.format(mass)]][tp-t])/2).sdev, fmt='ko')
            plt.title('G NG Mass = {0}'.format(mass))
            plt.xlabel('t',fontsize=20)
        #plt.legend()
            #lim = plt.ylim()
            #plt.plot([Fit['tminG'],Fit['tminG']],[lim[0],lim[1]],'k-',linewidth=2.5)
            #plt.plot([Fit['tmaxesG'][i],Fit['tmaxesG'][i]],[lim[0],lim[1]],'k-',linewidth=2.5)
            #plt.plot([int(3*tp/8-tp*gap),int(3*tp/8-tp*gap)],[lim[0],lim[1]],'k--',linewidth=2.5)
            #plt.plot([int(3*tp/8+tp*gap),int(3*tp/8+tp*gap)],[lim[0],lim[1]],'k--',linewidth=2.5)
    if 'NG' in FitCorrs:                
        for i,mass in enumerate(masses):
            plt.figure('log{0}'.format(mass))
            for t in range(1,int(tp/2+1)):                
                plt.errorbar(t, gv.log((data[TwoPts['NGm{0}'.format(mass)]][t]+data[TwoPts['NGm{0}'.format(mass)]][tp-t])/2).mean, yerr=gv.log((data[TwoPts['NGm{0}'.format(mass)]][t]+data[TwoPts['NGm{0}'.format(mass)]][tp-t])/2).sdev, fmt='ro')
            plt.title('G NG Mass = {0}'.format(mass))
            plt.xlabel('t',fontsize=20)
            plt.ylabel('log(2-point)',fontsize=20)
            #lim = plt.ylim()
            #plt.plot([Fit['tminNG'],Fit['tminNG']],[lim[0],lim[1]],'r-',linewidth=2.5)
            #plt.plot([Fit['tmaxesNG'][i],Fit['tmaxesNG'][i]],[lim[0],lim[1]],'r-',linewidth=2.5)
            #plt.plot([int(3*tp/8-tp*gap),int(3*tp/8-tp*gap)],[lim[0],lim[1]],'r--',linewidth=2.5)
            #plt.plot([int(3*tp/8+tp*gap),int(3*tp/8+tp*gap)],[lim[0],lim[1]],'r--',linewidth=2.5)
    if 'D' in FitCorrs:
        for i,twist in enumerate(twists):
            plt.figure(twist)
            for t in range(1,int(tp/2+1)):                
                plt.errorbar(t, ((data[TwoPts['Dtw{0}'.format(twist)]][t]+data[TwoPts['Dtw{0}'.format(twist)]][tp-t])/2).mean, yerr=((data[TwoPts['Dtw{0}'.format(twist)]][t]+data[TwoPts['Dtw{0}'.format(twist)]][tp-t])/2).sdev, fmt='ko')
            plt.title('D Twist = {0}'.format(twist))
            plt.xlabel('t',fontsize=20)
            plt.ylabel('2-point',fontsize=20)
        #plt.legend()
            lim = plt.ylim()
            plt.plot([Fit['tminD'],Fit['tminD']],[lim[0],lim[1]],'k-',linewidth=2.5)
            plt.plot([Fit['tmaxesD'][i],Fit['tmaxesD'][i]],[lim[0],lim[1]],'k-',linewidth=2.5)
            plt.plot([int(tp*(middle-gap)),int(tp*(middle-gap))],[lim[0],lim[1]],'k--',linewidth=2.5)
            plt.plot([int(tp*(middle+gap)),int(tp*(middle+gap))],[lim[0],lim[1]],'k--',linewidth=2.5)
            plt.plot([1,int(tp/2)],[0,0],'b--')
    if 'G' in FitCorrs:                
        for i,mass in enumerate(masses):
            plt.figure(mass)
            for t in range(1,int(tp/2+1)):                
                plt.errorbar(t, ((data[TwoPts['Gm{0}'.format(mass)]][t]+data[TwoPts['Gm{0}'.format(mass)]][tp-t])/2).mean, yerr=((data[TwoPts['Gm{0}'.format(mass)]][t]+data[TwoPts['Gm{0}'.format(mass)]][tp-t])/2).sdev, fmt='ko')
            plt.title('G NG Mass = {0}'.format(mass))
            plt.xlabel('t',fontsize=20)
            plt.ylabel('2-point',fontsize=20)
        #plt.legend()
            lim = plt.ylim()
            plt.plot([Fit['tminG'],Fit['tminG']],[lim[0],lim[1]],'k-',linewidth=2.5)
            plt.plot([Fit['tmaxesG'][i],Fit['tmaxesG'][i]],[lim[0],lim[1]],'k-',linewidth=2.5)
            plt.plot([int(tp*(middle-gap)),int(tp*(middle-gap))],[lim[0],lim[1]],'k--',linewidth=2.5)
            plt.plot([int(tp*(middle+gap)),int(tp*(middle+gap))],[lim[0],lim[1]],'k--',linewidth=2.5)
            plt.plot([1,int(tp/2)],[0,0],'b--')
    if 'NG' in FitCorrs:                
        for i,mass in enumerate(masses):
            plt.figure(mass)
            for t in range(1,int(tp/2+1)):                
                plt.errorbar(t, ((data[TwoPts['NGm{0}'.format(mass)]][t]+data[TwoPts['NGm{0}'.format(mass)]][tp-t])/2).mean, yerr=((data[TwoPts['NGm{0}'.format(mass)]][t]+data[TwoPts['NGm{0}'.format(mass)]][tp-t])/2).sdev, fmt='ro')
            plt.title('G NG Mass = {0}'.format(mass))
            plt.xlabel('t',fontsize=20)
            plt.ylabel('2-point',fontsize=20)
            lim = plt.ylim()
            plt.plot([Fit['tminNG'],Fit['tminNG']],[lim[0],lim[1]],'r-',linewidth=2.5)
            plt.plot([Fit['tmaxesNG'][i],Fit['tmaxesNG'][i]],[lim[0],lim[1]],'r-',linewidth=2.5)
            plt.plot([int(tp*(middle-gap)),int(tp*(middle-gap))],[lim[0],lim[1]],'r--',linewidth=2.5)
            plt.plot([int(tp*(middle+gap)),int(tp*(middle+gap))],[lim[0],lim[1]],'r--',linewidth=2.5)
            plt.plot([1,int(tp/2)],[0,0],'b--')

    colours = ['r','k','b']
    if 'S' in FitCorrs:        
        for mass in masses:
            for twist in twists:                    
                plt.figure('S{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, gv.log(data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).mean, yerr=gv.log(data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).sdev,  fmt='{0}o'.format(colours[i]))
                plt.title('S Mass = {0}, Twist = {1}'.format(mass, twist))
                plt.xlabel('t',fontsize=20)
                plt.ylabel('log(3-point)',fontsize=20)
                #plt.legend()

                plt.figure('SRat{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, (data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['Gm{0}'.format(mass)]][T-t])).mean, yerr=(data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['Gm{0}'.format(mass)]][T-t])).sdev, fmt='{0}o'.format(colours[i]))
                plt.title('S Ratio Mass = {0}, Twist = {1}'.format(mass, twist))
                plt.xlabel('t',fontsize=20)
                #plt.legend()
        
    if 'V' in FitCorrs:
        for mass in masses:
            for twist in twists:                    
                plt.figure('V{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, gv.log(data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).mean, yerr=gv.log(data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).sdev, fmt='{0}o'.format(colours[i]))
                plt.title('V Mass = {0}, Twist = {1}'.format(mass, twist))
                plt.xlabel('t',fontsize=20)
                plt.ylabel('log(3-point)',fontsize=20)
                #plt.legend()

                plt.figure('VRat{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, (data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['NGm{0}'.format(mass)]][T-t])).mean, yerr=(data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['NGm{0}'.format(mass)]][T-t])).sdev, fmt='{0}o'.format(colours[i]))
                plt.title('V Ratio Mass = {0}, Twist = {1}'.format(mass, twist))
                plt.xlabel('t',fontsize=20)
                #plt.legend()
    plt.show()
    return()


def makeKeys():
    TwoKeys = []
    ThreeKeys = []
    if 'D' in FitCorrs:
        for twist in twists:
            TwoKeys.append('log({0}:a)'.format(TwoPts['Dtw{0}'.format(twist)]))
            TwoKeys.append('log(dE:{0})'.format(TwoPts['Dtw{0}'.format(twist)]))
            if twist != '0':            
                TwoKeys.append('log(dE:o{0})'.format(TwoPts['Dtw{0}'.format(twist)]))
                TwoKeys.append('log(o{0}:a)'.format(TwoPts['Dtw{0}'.format(twist)]))
    if 'G' in FitCorrs:
        for mass in masses:
            TwoKeys.append('log({0}:a)'.format(TwoPts['Gm{0}'.format(mass)]))
            TwoKeys.append('log(o{0}:a)'.format(TwoPts['Gm{0}'.format(mass)]))
            TwoKeys.append('log(dE:{0})'.format(TwoPts['Gm{0}'.format(mass)]))
            TwoKeys.append('log(dE:o{0})'.format(TwoPts['Gm{0}'.format(mass)]))
    if 'NG' in FitCorrs:
        for mass in masses:
            TwoKeys.append('log({0}:a)'.format(TwoPts['NGm{0}'.format(mass)]))
            TwoKeys.append('log(o{0}:a)'.format(TwoPts['NGm{0}'.format(mass)]))
            TwoKeys.append('log(dE:{0})'.format(TwoPts['NGm{0}'.format(mass)]))
            TwoKeys.append('log(dE:o{0})'.format(TwoPts['NGm{0}'.format(mass)]))
    if 'S' in FitCorrs:
        for mass in masses:
            for twist in twists:
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    ThreeKeys.append('SVnn_m{0}_tw{1}'.format(mass,twist))
                    ThreeKeys.append('SVno_m{0}_tw{1}'.format(mass,twist))
                    if twist != '0':
                        ThreeKeys.append('SVon_m{0}_tw{1}'.format(mass,twist))
                        ThreeKeys.append('SVoo_m{0}_tw{1}'.format(mass,twist))
    if 'V' in FitCorrs:
        for mass in masses:
            for twist in twists:
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    ThreeKeys.append('VVnn_m{0}_tw{1}'.format(mass,twist))
                    ThreeKeys.append('VVno_m{0}_tw{1}'.format(mass,twist))
                    if twist != '0':
                        ThreeKeys.append('VVon_m{0}_tw{1}'.format(mass,twist))
                        ThreeKeys.append('VVoo_m{0}_tw{1}'.format(mass,twist))
    #print(TwoKeys,ThreeKeys)
    return(TwoKeys,ThreeKeys)



def save_p0(p0,Nexp,fname,TwoKeys,ThreeKeys):
    if os.path.exists('Ps/{0}.pickle'.format(Fit['conf'])):
        rglobalpickle = open('Ps/{0}.pickle'.format(Fit['conf']), "rb")
        p1 = pickle.load(rglobalpickle)                    
        for key in TwoKeys:                        
            if key in p1.keys():
                if len(p0[key]) >= len(p1[key]):      
                    p1.pop(key,None)
                    p1[key]=p0[key]
                    print('Replaced element of global p0:', key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0:',key)
        for key in ThreeKeys:
            if key in p1.keys():
                if np.shape(p0[key])[0] >= np.shape(p1[key])[0]:      
                    p1.pop(key,None)
                    p1[key]=p0[key]
                    print('Replaced element of global p0:', key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0:',key)
        wglobalpickle = open('Ps/{0}.pickle'.format(Fit['conf']), "wb")
        pickle.dump(p1,wglobalpickle)
        wglobalpickle.close()
    else:
        p2 = collections.OrderedDict()
        for key in TwoKeys:                        
            p2[key] = copy.deepcopy(p0[key])
        for key in ThreeKeys:                        
            p2[key] = copy.deepcopy(p0[key])

        wglobalpickle = open('Ps/{0}.pickle'.format(Fit['conf']), "wb")
        pickle.dump(p2,wglobalpickle)
        wglobalpickle.close()
#################################### p0 Nexp ###########################             
    if os.path.exists('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp)):
        rglobalpickle = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp), "rb")
        p1 = pickle.load(rglobalpickle)                    
        for key in TwoKeys:                        
            if key in p1.keys():                      
                p1.pop(key,None)
                p1[key]=p0[key]
                #print('Replaced element of global p0 Nexp={0}:'.format(Nexp), key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0 Nexp={0}:'.format(Nexp),key)
        for key in ThreeKeys:
            if key in p1.keys():                      
                p1.pop(key,None)
                p1[key]=p0[key]
                #print('Replaced element of global p0 Nexp={0}:'.format(Nexp), key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0 Nexp={0}:'.format(Nexp),key)
        wglobalpickle = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp), "wb")
        pickle.dump(p1,wglobalpickle)
        wglobalpickle.close()
    else:
        p2 = collections.OrderedDict()
        for key in TwoKeys:                        
            p2[key] = copy.deepcopy(p0[key])
        for key in ThreeKeys:                        
            p2[key] = copy.deepcopy(p0[key])
        wglobalpickle = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp), "wb")
        pickle.dump(p2,wglobalpickle)
        wglobalpickle.close()
        
    return()




def load_p0(p0,Nexp,fname,TwoKeys,ThreeKeys):
    elements1 = []
    elements2 = []
    if os.path.isfile('{0}{1}.pickle'.format(fname,Nexp)):
        pickle_off = open('{0}{1}.pickle'.format(fname,Nexp),"rb")
        p0 = pickle.load(pickle_off)
        print('Using existing p0 for Nexp')                
    elif os.path.isfile('{0}{1}.pickle'.format(fname,Nexp+1)):
        pickle_off = open('{0}{1}.pickle'.format(fname,Nexp+1),"rb")
        p1 = pickle.load(pickle_off)
        for key in TwoKeys:
            if key in p1.keys():
                p0.pop(key,None)
                p0[key] = p1[key][:-1]
        print('Using existing p0 for Nexp+1')
    
    elif os.path.exists('Ps/{0}.pickle'.format(Fit['conf'])):
        pickle_off = open('Ps/{0}.pickle'.format(Fit['conf']),"rb")
        p1 = pickle.load(pickle_off)
        for key in TwoKeys:
            if key in p1.keys():                    
                if len(p1[key]) >= Nexp:
                    elements1.append(key)
                    p0.pop(key,None)                            
                    p0[key] = np.zeros((Nexp))                            
                    for n in range(Nexp):
                        p0[key][n] = p1[key][n]                            
        for key in ThreeKeys:
            if key in p1.keys():                    
                if np.shape(p1[key])[0] >= Nexp:
                    p0.pop(key,None)
                    p0[key]=np.zeros((Nexp,Nexp))
                    elements1.append(key)
                    for n in range(Nexp):
                        for m in range(Nexp):
                            p0[key][n][m]=p1[key][n][m]
        if os.path.exists('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp)):
            pickle_off = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp),"rb")
            p2 = pickle.load(pickle_off)
            for key in TwoKeys:
                if key in p2.keys():
                    p0.pop(key,None)
                    p0[key] = p2[key]                    
                    elements2.append(key)         
            for key in ThreeKeys:
                if key in p2.keys():
                   p0.pop(key,None)
                   p0[key]=p2[key]
                   elements2.append(key)
        for element in elements1:
            if element in elements2:
                print('Using element of global p0 Nexp={0}:'.format(Nexp),element)
            else:
                print('Using element of global p0:',element)
    return(p0)




def print_results(p):
    if 'D' in FitCorrs:
        for twist in twists:
            print('D    tw {0:<16}: {1:<12} Error: {2:.3f}%'.format(twist,p['dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)])][0],100*p['dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)])][0].sdev/p['dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)])][0].mean))
    if 'G' in FitCorrs:
        for mass in masses:
            print('G    m  {0:<16}: {1:<12} Error: {2:.3f}%'.format(mass,p['dE:{0}'.format(TwoPts['Gm{0}'.format(mass)])][0],100*p['dE:{0}'.format(TwoPts['Gm{0}'.format(mass)])][0].sdev/p['dE:{0}'.format(TwoPts['Gm{0}'.format(mass)])][0].mean))
    if 'NG' in FitCorrs:
        for mass in masses:
            print('NG   m  {0:<16}: {1:<12} Error: {2:.3f}%'.format(mass,p['dE:{0}'.format(TwoPts['NGm{0}'.format(mass)])][0],100*p['dE:{0}'.format(TwoPts['NGm{0}'.format(mass)])][0].sdev/p['dE:{0}'.format(TwoPts['NGm{0}'.format(mass)])][0].mean))
    if 'S' in FitCorrs:
        for mass in masses:
            for twist in twists:
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    print('SVnn m  {0:<5} tw {1:<7}: {2:<12} Error: {3:.3f}%'. format(mass, twist,p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0],100*p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0].sdev/p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0].mean))
    if 'V' in FitCorrs:
        for mass in masses:
            for twist in twists:
                if qsqPos['m{0}_tw{1}'.format(mass,twist)] == 1:
                    print('VVnn m  {0:<5} tw {1:<7}: {2:<12} Error: {3:.3f}%'. format(mass, twist,p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0],100*p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0].sdev/p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0].mean))

    return()


    
if DoFit == True:
    filename = Fit['filename']
    Autoprior,data = make_data(filename,Nmax)
    gotnan = any(bool(a[~np.isfinite([p.mean for p in a])].size) for a in data.values())
    print('Nan or inf in data: ',gotnan)
    if FitAll == True:
        FitTs = [0]
        FitCorrs = ['G','NG','D']#,'S','V']
        for i in range(4):
            FitMasses = [i]
            for j in range(5):                 
                FitTwists = [j]
                TwoPts,ThreePts,masses,twists,Ts,tmaxesBG,tmaxesBNG,tmaxesKG,tmaxesKNG,qsqPos = make_params(FitMasses, FitTwists,FitTs)
                main(Autoprior,data)
                            
    else:        
        TwoPts,ThreePts,masses,twists,Ts,tmaxesBG,tmaxesBNG,tmaxesKG,tmaxesKNG,qsqPos = make_params(FitMasses,FitTwists,FitTs)
        if TestData == True:
            test_data()
        main(Autoprior,data)
        plt.show()
