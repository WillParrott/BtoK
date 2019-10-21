import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

plt.rc("font",**{"size":18})
import collections
import copy
import os.path
import pickle
from collections import defaultdict
################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['filename'] = 'Fits/F5_3pts_Q1.00_Nexp1_Stmin2_Vtmin1_svd0.00157_chi0.360'
F['Masses'] = ['0.449','0.566','0.683','0.8']
F['Twists'] = ['0','0.4281','1.282','2.141','2.570','2.993']
F['m_l'] = '0.0074'
F['m_s'] = '0.0376'
F['m_ssea'] = 0.037
F['m_lsea'] = 0.0074
F['Ts'] = [14,17,20]
F['tp'] = 96
F['L'] = 32
F['w0/a'] = '1.9006(20)'
F['BG-Tag'] = 'B_G5-G5_m{0}'
F['BNG-Tag'] = 'B_G5T-G5T_m{0}'
F['KG-Tag'] = 'K_G5-G5_tw{0}'
F['KNG-Tag'] = 'K_G5-G5X_tw{0}'
F['threePtTag'] = '{0}_T{1}_m{2}_m{3}_m{4}_tw{5}'


##################### USER INPUTS ##########################################
Masses = collections.OrderedDict()
Twists = collections.OrderedDict()
############################################################################

Fits = [F]#,SF]#,UF]                                         # Choose to fit F, SF or UF
Masses['F'] = [0,1,2,3]                                     # Choose which masses to fit
Twists['F'] = [0,1,2,3,4]#,5]
Masses['SF'] = [0,1,2,3]
Twists['SF'] = [0,1,2,3,4]
Masses['UF'] = [0,1,2,3]
Twists['UF'] = [0,1,2,3,4]
AddRho = True
svdnoise = False
priornoise = False
FitNegQsq = True
Pri = '0.00(1.00)'
DoFit = True
N = 3
SHOWPLOTS = True
############################################################################
############################################################################
   
def make_params():
    w0 = gv.gvar('0.1715(9)')  #fm
    hbar = gv.gvar('6.58211928(15)')
    c = 2.99792458
    #for Fit in [F,S,UF]:
    #    Fit['a'] = w0/((hbar*c*1e-2)*gv.gvar(Fit['w0/a']))
    for Fit in Fits:
        Fit['a'] = w0/((hbar*c*1e-2)*gv.gvar(Fit['w0/a']))
        Fit['masses'] = []
        Fit['twists'] = []
        Fit['momenta'] = []
        Fit['Delta'] = 0
        for i in Masses[Fit['conf']]:
            Fit['masses'].append(Fit['Masses'][i])
        for j in Twists[Fit['conf']]:
            Fit['twists'].append(Fit['Twists'][j])
        for twist in Fit['twists']:
            Fit['momenta'].append(np.sqrt(3)*np.pi*float(twist)/Fit['L'])
        for Twist in Fit['Twists']:
            Fit['KGtw{0}'.format(Twist)] = Fit['KG-Tag'].format(Twist)
            Fit['KNGtw{0}'.format(Twist)] = Fit['KNG-Tag'].format(Twist)
        for Mass in Fit['Masses']:
            Fit['BGm{0}'.format(Mass)] = Fit['BG-Tag'].format(Mass)
            Fit['BNGm{0}'.format(Mass)] = Fit['BNG-Tag'].format(Mass)
                   
    return()


def get_results(Fit):
    Vnn = collections.OrderedDict()
    p = gv.load(Fit['filename'])    
    for i in range(len(Fit['twists'])):
        Fit['M_KG_tw{0}'.format(Fit['twists'][i])] =  gv.sqrt(p['dE:{0}'.format(Fit['KGtw{0}'.format(Fit['twists'][i])])][0]**2 - Fit['momenta'][i]**2)
        Fit['E_KG_tw{0}'.format(Fit['twists'][i])] = p['dE:{0}'.format(Fit['KGtw{0}'.format(Fit['twists'][i])])][0]
        Fit['M_KNG_tw{0}'.format(Fit['twists'][i])] =  gv.sqrt(p['dE:{0}'.format(Fit['KNGtw{0}'.format(Fit['twists'][i])])][0]**2 - Fit['momenta'][i]**2)
        Fit['E_KNG_tw{0}'.format(Fit['twists'][i])] = p['dE:{0}'.format(Fit['KNGtw{0}'.format(Fit['twists'][i])])][0]
    for mass in Fit['masses']:
        Fit['M_BG_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['BGm{0}'.format(mass)])][0]
        Fit['M_BGo_m{0}'.format(mass)] = p['dE:o{0}'.format(Fit['BGm{0}'.format(mass)])][0]
        Fit['M_BNG_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['BNGm{0}'.format(mass)])][0]
        Fit['M_BNGo_m{0}'.format(mass)] = p['dE:o{0}'.format(Fit['BNGm{0}'.format(mass)])][0]
        #print(gv.evalcorr([Fit['M_NG_m{0}'.format(mass)],Fit['M_G_m{0}'.format(mass)]]))    
        for twist in Fit['twists']:
            if 'SVnn_m{0}_tw{1}'.format(mass,twist) in p:
                Fit['Sm{0}_tw{1}'.format(mass,twist)] = 2*2*gv.sqrt(Fit['E_KG_tw{0}'.format(twist)]*Fit['M_BG_m{0}'.format(mass)])*p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
                Fit['Vm{0}_tw{1}'.format(mass,twist)] = 2*2*gv.sqrt(Fit['E_KG_tw{0}'.format(twist)]*Fit['M_BG_m{0}'.format(mass)])*p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
                Fit['Tm{0}_tw{1}'.format(mass,twist)] = 2*2*gv.sqrt(Fit['E_KG_tw{0}'.format(twist)]*Fit['M_BG_m{0}'.format(mass)])*p['TVnn_m{0}_tw{1}'.format(mass,twist)][0][0]  ### Is this correct???
    return()


def make_fs(Fit):
    plt.figure(9)
    make_params()    
    print('Calc for', Fit['filename'] )
    get_results(Fit)
    F_0 = collections.OrderedDict()
    F_plus = collections.OrderedDict()
    F_T = collections.OrderedDict()
    qSq = collections.OrderedDict()
    Z = collections.OrderedDict()
    Sca = collections.OrderedDict()
    Vec = collections.OrderedDict()
    Ten = collections.OrderedDict()
    for mass in Fit['masses']:
        F_0[mass] = collections.OrderedDict()
        F_plus[mass] = collections.OrderedDict()
        F_T[mass] = collections.OrderedDict()
        qSq[mass] = collections.OrderedDict()
        Z[mass] = collections.OrderedDict()
        Sca[mass] = collections.OrderedDict()
        Vec[mass] = collections.OrderedDict()
        Ten[mass] = collections.OrderedDict()
        Z_v = (float(mass) - float(Fit['m_s']))*Fit['Sm{0}_tw0'.format(mass)]/((Fit['M_BG_m{0}'.format(mass)] - Fit['M_KG_tw0'])*Fit['Vm{0}_tw0'.format(mass)])
        plt.errorbar((Fit['a']**2).mean,Z_v.mean,xerr=(Fit['a']**2).sdev,yerr=Z_v.sdev,label=mass)
        for twist in Fit['twists']:
            if 'Sm{0}_tw{1}'.format(mass,twist) in Fit:
                delta = (float(mass) - float(Fit['m_s']))*(Fit['M_BG_m{0}'.format(mass)]-Fit['E_KG_tw{0}'.format(twist)])
                qsq = Fit['M_BG_m{0}'.format(mass)]**2 + Fit['M_KG_tw{0}'.format(twist)]**2 - 2*Fit['M_BG_m{0}'.format(mass)]*Fit['E_KG_tw{0}'.format(twist)]
                t = (Fit['M_BG_m{0}'.format(mass)] + Fit['M_KG_tw{0}'.format(twist)])**2
                z = (gv.sqrt(t-qsq)-gv.sqrt(t))/(gv.sqrt(t-qsq)+gv.sqrt(t)) 
                if FitNegQsq == False:
                    if qsq.mean >= 0:
                        F0 = (float(mass) - float(Fit['m_s']))*(1/(Fit['M_BG_m{0}'.format(mass)]**2 - Fit['M_KG_tw{0}'.format(twist)]**2))*Fit['Sm{0}_tw{1}'.format(mass,twist)]
                        FT = Fit['Sm{0}_tw{1}'.format(mass,twist)]*(Fit['M_BG_m{0}'.format(mass)]+Fit['M_KG_m{0}'.format(mass)])/(2*Fit['M_BG_m{0}'.format(mass)]*float(twist))        # Have we used correct masses?
                        F_0[mass][twist] = F0
                        F_T[mass][twist] = FT
                        qSq[mass][twist] = qsq                    
                        Z[mass][twist] = z
                        Sca[mass][twist] = Fit['Sm{0}_tw{1}'.format(mass,twist)]
                        Vec[mass][twist] = Fit['Vm{0}_tw{1}'.format(mass,twist)]
                        Ten[mass][twist] = Fit['Tm{0}_tw{1}'.format(mass,twist)]
                        A = Fit['M_BG_m{0}'.format(mass)] + Fit['E_KG_tw{0}'.format(twist)]
                        B = (Fit['M_BG_m{0}'.format(mass)]**2 - Fit['M_KG_tw{0}'.format(twist)]**2)*(Fit['M_BG_m{0}'.format(mass)] - Fit['E_KG_tw{0}'.format(twist)])/qsq           
                        if twist != '0':
                            F_plus[mass][twist] = (1/(A-B))*(Z_v*Fit['Vm{0}_tw{1}'.format(mass,twist)] - B*F0)       
                elif FitNegQsq == True:
                    F0 = (float(mass) - float(Fit['m_s']))*(1/(Fit['M_BG_m{0}'.format(mass)]**2 - Fit['M_KG_tw{0}'.format(twist)]**2))*Fit['Sm{0}_tw{1}'.format(mass,twist)]
                    FT = Fit['Sm{0}_tw{1}'.format(mass,twist)]*(Fit['M_BG_m{0}'.format(mass)]+Fit['M_KG_m{0}'.format(mass)])/(2*Fit['M_BG_m{0}'.format(mass)]*float(twist))
                    F_0[mass][twist] = F0
                    F_T[mass][twist] = FT  
                    qSq[mass][twist] = qsq                    
                    Z[mass][twist] = z
                    Sca[mass][twist] = Fit['Sm{0}_tw{1}'.format(mass,twist)]
                    Vec[mass][twist] = Fit['Vm{0}_tw{1}'.format(mass,twist)]
                    Ten[mass][twist] = Fit['Tm{0}_tw{1}'.format(mass,twist)]
                    A = Fit['M_BG_m{0}'.format(mass)] + Fit['E_KG_tw{0}'.format(twist)]
                    B = (Fit['M_BG_m{0}'.format(mass)]**2 - Fit['M_KG_tw{0}'.format(twist)]**2)*(Fit['M_BG_m{0}'.format(mass)] - Fit['E_KG_tw{0}'.format(twist)])/qsq           
                    if twist != '0':
                        F_plus[mass][twist] = (1/(A-B))*(Z_v*Fit['Vm{0}_tw{1}'.format(mass,twist)] - B*F0)
   # plt.legend()
    plt.xlabel('$a^2$')
    plt.ylabel('$Z_v$')
    return(F_0,F_plus,qSq,Z)



def main():
    Metasphys = gv.gvar('0.6885(22)')
    slratio = gv.gvar('27.18(10)')
    make_params()
    f = gv.BufferDict()
    #fplus = gv.BufferDict()
    z = gv.BufferDict()
    prior = gv.BufferDict()    
    mh0val = gv.BufferDict()   
    MetacF = gv.gvar('1.366850(90)')/F['a']
    MetacSF = gv.gvar('0.896686(23)')/SF['a']
    MetacUF = gv.gvar('0.666754(39)')/UF['a']       #Check this
    MetacPhys = gv.gvar('2.9863(27)')
    x = MetacPhys*gv.gvar('0.1438(4)')/2   #GeV
    datatags = []
    prior['Metacphys'] = MetacPhys
    LQCD = collections.OrderedDict()
    for Fit in Fits:
        LQCD['{0}'.format(Fit['conf'])] = 0.5*(Fit['a'].mean)
        #print(gv.evalcorr([prior['LQCD_{0}'.format(Fit['conf'])],Fit['a']]))
        F_0,F_plus,qSq,Z = make_fs(Fit)
        if Fit == F:
            prior['Metac_{0}'.format(Fit['conf'])] = MetacF          
        elif Fit == SF:
            prior['Metac_{0}'.format(Fit['conf'])] = MetacSF
        elif Fit == UF:
            prior['Metac_{0}'.format(Fit['conf'])] = MetacUF
        ms0val = float(Fit['m_s'])
        Metas = Fit['M_D_tw{0}'.format(0)]/Fit['a']
        prior['mstuned_{0}'.format(Fit['conf'])] = ms0val*(Metasphys/Metas)**2
        mltuned = prior['mstuned_{0}'.format(Fit['conf'])]/slratio 
        prior['MHc_{0}'.format(Fit['conf'])] = Fit['M_G_m{0}'.format(Fit['Masses'][0])]      
        prior['mstuned_{0}'.format(Fit['conf'])] = ms0val*(Metasphys/Metas)**2
        ms0 = Fit['m_ssea']
        ml0 = Fit['m_lsea']
        prior['deltas_{0}'.format(Fit['conf'])] = ms0-prior['mstuned_{0}'.format(Fit['conf'])]     
        prior['deltasval_{0}'.format(Fit['conf'])] = ms0val-prior['mstuned_{0}'.format(Fit['conf'])]
        prior['deltal_{0}'.format(Fit['conf'])] = ml0-mltuned
        for mass in Fit['masses']:
            prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)] = Fit['M_Go_m{0}'.format(mass)]
            prior['MHh_{0}_m{1}'.format(Fit['conf'],mass)] = Fit['M_G_m{0}'.format(mass)]
            prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)] = prior['MHh_{0}_m{1}'.format(Fit['conf'],mass)] + x*Fit['a']/prior['MHh_{0}_m{1}'.format(Fit['conf'],mass)]
            
            
            mh0val['{0}_m{1}'.format(Fit['conf'],mass)] = float(mass)
            for twist in Fit['twists']:
                if twist in qSq[mass]:                    
                    datatag =  '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                    datatags.append(datatag)
                    prior['Eetas_{0}_tw{1}'.format(Fit['conf'],twist)] = Fit['E_D_tw{0}'.format(twist)]
                    prior['z_{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)] = Z[mass][twist]
                    prior['qsq_{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)] = qSq[mass][twist]                    
                    f['0{0}'.format(datatag)] = F_0[mass][twist]
                    if twist !='0':
                        f['plus{0}'.format(datatag)] = F_plus[mass][twist]                                            
    if AddRho == True:
        prior['0rho'] =gv.gvar(N*[Pri])
        prior['plusrho'] =gv.gvar(N*[Pri])
        prior['plusrho'][0] = prior['0rho'][0]       
    prior['0d'] = gv.gvar(3*[3*[3*[N*[Pri]]]])
    prior['0cl'] = gv.gvar(N*[Pri])
    prior['0cs'] = gv.gvar(N*[Pri])
    prior['0cc'] = gv.gvar(N*[Pri])
    prior['0csval'] = gv.gvar(N*[Pri])
    prior['plusd'] = gv.gvar(3*[3*[3*[N*[Pri]]]])
    for i in range(3):
        prior['plusd'][i][0][0][0] = prior['0d'][i][0][0][0]
    prior['pluscl'] = gv.gvar(N*[Pri])
    prior['pluscs'] = gv.gvar(N*[Pri])
    prior['pluscc'] = gv.gvar(N*[Pri])
    prior['pluscsval'] = gv.gvar(N*[Pri])
    
    #np.save('Extraps/Datatags', datatags)   

    #plots(prior,f,Metasphys)
    if DoFit == True:
        def fcn(p):
            models = {}
            #print(datatags)
            for datatag in datatags:
                fit = datatag.split('_')[0]
                #fitdict = locals()[datatag.split('_')[0]]
                mass = datatag.split('_')[1].strip('m')
                twist = datatag.split('_')[2].strip('tw')
                if '0{0}'.format(datatag) in f:
                    models['0{0}'.format(datatag)] =  0
                if 'plus{0}'.format(datatag) in f:
                    models['plus{0}'.format(datatag)] =  0
                for n in range(N):
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                if AddRho == True: 
                                    if '0{0}'.format(datatag) in f:
                                        models['0{0}'.format(datatag)] += (1/(1-(p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/(p['MHs0_{0}_m{1}'.format(fit,mass)])**2))) * (p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)])**n * (1 + p['0rho'][n]*gv.log(p['MHh_{0}_m{1}'.format(fit,mass)]/p['MHc_{0}'.format(fit)])) * (1 + (p['0csval'][n]*p['deltasval_{0}'.format(fit)] + p['0cs'][n]*p['deltas_{0}'.format(fit)] + 2*p['0cl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['0cc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['0d'][i][j][k][n] * (LQCD['{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                    if 'plus{0}'.format(datatag) in f:                                       
                                        models['plus{0}'.format(datatag)] += (1/(1-p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/p['MHsstar_{0}_m{1}'.format(fit,mass)]**2)) * ( p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**n - (n/N) * (-1)**(n-N) * p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**N)* (1 + p['plusrho'][n]*gv.log(p['MHh_{0}_m{1}'.format(fit,mass)]/p['MHc_{0}'.format(fit)])) * (1 + (p['pluscsval'][n]*p['deltasval_{0}'.format(fit)] + p['pluscs'][n]*p['deltas_{0}'.format(fit)] + 2*p['pluscl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['pluscc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['plusd'][i][j][k][n] * (LQCD['{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                else:
                                    if '0{0}'.format(datatag) in f:
                                        models['0{0}'.format(datatag)] += (1/(1-(p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/(p['MHs0_{0}_m{1}'.format(fit,mass)])**2))) * (p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)])**n * (1 + (p['0csval'][n]*p['deltasval_{0}'.format(fit)] + p['0cs'][n]*p['deltas_{0}'.format(fit)] + 2*p['0cl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['0cc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['0d'][i][j][k][n] * (LQCD['{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                    if 'plus{0}'.format(datatag) in f:                                       
                                        models['plus{0}'.format(datatag)] += (1/(1-p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/p['MHsstar_{0}_m{1}'.format(fit,mass)]**2)) * ( p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**n - (n/N) * (-1)**(n-N) * p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**N) * (1 + (p['pluscsval'][n]*p['deltasval_{0}'.format(fit)] + p['pluscs'][n]*p['deltas_{0}'.format(fit)] + 2*p['pluscl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['pluscc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['plusd'][i][j][k][n] * (LQCD['{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                                                                                
                            
            return(models)
    
        if os.path.exists('Extraps/{0}{1}.pickle'.format(AddRho,N)):
            pickle_off = open('Extraps/{0}{1}.pickle'.format(AddRho,N),"rb")
            p0 = pickle.load(pickle_off)
        else:
            p0 = None
        #s = gv.dataset.svd_diagnosis(f)
        #s.plot_ratio(show=True)
        fit = lsqfit.nonlinear_fit(data=f, prior=prior, p0=p0, fcn=fcn, svdcut=1e-15 ,add_svdnoise=svdnoise, add_priornoise=priornoise,fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, tol=(1e-6,0.0,0.0) )
        gv.dump(fit.p,'Extraps/{0}{1}chi{2:.3f}'.format(AddRho,N,fit.chi2/fit.dof))
        print(fit.format(maxline=True))        
        #print(fit)
        savefile = open('Extraps/{0}{1}chi{2:.3f}.txt'.format(AddRho,N,fit.chi2/fit.dof),'w')
        savefile.write(fit.format(pstyle='v'))
        savefile.close()
        pickle_on = open('Extraps/{0}{1}.pickle'.format(AddRho,N),"wb")
        pickle.dump(fit.pmean,pickle_on)
        pickle_on.close
        plots(prior,f,Metasphys,fit.p,x)
    return()
    
     
       
def plots(prior,f,Metasphys,p,x):
    Z,Zmax,Qsq,MBsphys,MBsstarphys,F0mean,F0upp,F0low,Fplusmean,Fplusupp,Fpluslow,F0meanpole,F0upppole,F0lowpole,Fplusmeanpole,Fplusupppole,Fpluslowpole = plot_results(Metasphys,p,x,prior,f)
    cols = ['b','g','r','c']#,'m','y','k','purple']
    plt.figure(1)
    if F in Fits:   
        Fit = F
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]        
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
                if '0{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='o',ms=12, mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if SF in Fits:
        Fit = SF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]        
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
                if '0{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='--')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',fmt='^',ms=12,label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if UF in Fits:        
        Fit = UF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]        
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
                if '0{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='-.')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',fmt='*',ms=12,label=('{0}_m{1}'.format(Fit['conf'],mass)))
    plt.plot(Z,F0mean, color='b')
    #plt.plot(Z,F0upp, color='r')
    plt.fill_between(Z,F0low,F0upp, color='b',alpha=0.4)
    plt.legend()
    plt.xlabel('z',fontsize=30)
    plt.ylabel(r'$(1-\frac{q^2}{M^2_{H_{s}^0}})f_0$',fontsize=30)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    #plt.savefig('Extraps/f0pole')
    plt.figure(2)
    if F in Fits:
        Fit = F
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                if 'plus{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='o',ms=12,mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if SF in Fits:        
        Fit = SF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                if 'plus{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='--')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='^',ms=12,mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if UF in Fits:        
        Fit = UF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                if 'plus{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='-.')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='*',ms=12,mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    plt.plot(Z,Fplusmean, color='r')
    #plt.plot(Z,Fplusupp, color='g')
    plt.fill_between(Z,Fpluslow,Fplusupp, color='r',alpha=0.4)
    plt.legend()
    plt.xlabel('z',fontsize=30)
    plt.ylabel(r'$(1-\frac{q^2}{M^2_{H_{s}^*}})f_{plus}$',fontsize=30)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
    #plt.savefig('Extraps/fpluspole')



    plt.figure(7)
    if F in Fits:
        Fit = F
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]        
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
                if '0{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append(f['0{0}'.format(datatag)].mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append(f['0{0}'.format(datatag)].sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='o', mfc='none',ms=12,label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if SF in Fits:        
        Fit = SF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]        
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
                if '0{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append(f['0{0}'.format(datatag)].mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append(f['0{0}'.format(datatag)].sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='--')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',fmt='^',ms=12,label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if UF in Fits:        
        Fit = UF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]        
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
                if '0{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append(f['0{0}'.format(datatag)].mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append(f['0{0}'.format(datatag)].sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='-.')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',fmt='*',ms=12,label=('{0}_m{1}'.format(Fit['conf'],mass)))
    plt.plot(Z,F0meanpole, color='b')
    #plt.plot(Z,F0upp, color='r')
    plt.fill_between(Z,F0lowpole,F0upppole, color='b',alpha=0.4)
    plt.legend()
    plt.xlabel('z',fontsize=30)
    plt.ylabel(r'$f_0$',fontsize=30)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.axes().set_xlim([Zmax,0])
    #plt.savefig('Extraps/f0')


    
    plt.figure(8)
    if F in Fits:
        Fit = F
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                if 'plus{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append(f['plus{0}'.format(datatag)].mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append(f['plus{0}'.format(datatag)].sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='o',ms=12,mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if SF in Fits:        
        Fit = SF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                if 'plus{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append(f['plus{0}'.format(datatag)].mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append(f['plus{0}'.format(datatag)].sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='--')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='^',ms=12,mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    if UF in Fits:        
        Fit = UF
        for i , mass in enumerate(Fit['masses']):
            xmean = [] 
            xerr = [] 
            ymean = [] 
            yerr = []
            col = cols[i]
            for twist in Fit['twists']:
                datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                if 'plus{0}'.format(datatag) in f:
                    xmean.append(prior['z_{0}'.format(datatag)].mean)
                    ymean.append(f['plus{0}'.format(datatag)].mean)
                    xerr.append(prior['z_{0}'.format(datatag)].sdev)
                    yerr.append(f['plus{0}'.format(datatag)].sdev)
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',linestyle='-.')
            plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='*',ms=12,mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    plt.plot(Z,Fplusmeanpole, color='r')
    #plt.plot(Z,Fplusupp, color='g')
    plt.fill_between(Z,Fpluslowpole,Fplusupppole, color='r',alpha=0.4)
    plt.legend()
    plt.xlabel('z',fontsize=30)
    plt.ylabel(r'$f_{+}$',fontsize=30)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    #plt.axes().set_xlim([Zmax,0])
    #plt.savefig('Extraps/fplus')
    return()




def plot_results(Metasphys,p,x,prior,f):
    nopts = 100
    a = collections.OrderedDict()
    a['0'] = [0]*N
    a['plus'] = [0]*N
    Del = gv.gvar('0.312(15)')
    filename = 'Extraps/Testchi0.142'
    datatag0 = 'F_m0.499_tw0'
    plusdatatag = 'F_m0.499_tw0.4281'
    Metab = gv.gvar('9.3987(20)')
    MBsphys = gv.gvar('5.36688(17)')
    MDsphys = gv.gvar('1.96834(7)')
    MBs0 = gv.gvar('5.36688(17)') + Del  # Get this properly later
    #MBs0 = gv.gvar('5.36688(17)') + gv.gvar('0.3471(73)')
    MBsstarphys = gv.gvar('5.4158(15)')
    #p = gv.load(filename)
    F0meanpole = np.zeros((nopts))
    Fplusmeanpole = np.zeros((nopts))
    F0upppole = np.zeros((nopts))
    F0lowpole = np.zeros((nopts))
    Fplusupppole = np.zeros((nopts))
    Fpluslowpole = np.zeros((nopts))
    F0mean = np.zeros((nopts))
    Fplusmean = np.zeros((nopts))
    F0upp= np.zeros((nopts))
    F0low= np.zeros((nopts))
    Fplusupp = np.zeros((nopts))
    Fpluslow = np.zeros((nopts))
    Z = []
    Zmean = []
    #qsqmax = 26
    qsqmax = (MBsphys-Metasphys)**2
    tplus = (MBsphys+Metasphys)**2
    zmax = ((gv.sqrt(tplus-qsqmax)-gv.sqrt(tplus))/(gv.sqrt(tplus-qsqmax)+gv.sqrt(tplus))).mean
    Zmax=zmax
    qsq = np.linspace(0,qsqmax.mean,nopts)
    #print('qsq',qsq)
    #p['LQCD'] = 0.5
    #if F in Fits:
    #    p['LQCD'] = p['LQCD_F']/F['a']
    #elif SF in Fits:
    #    p['LQCD'] = p['LQCD_SF']/SF['a']
    #elif UF in Fits:
    #    p['LQCD'] = p['LQCD_UF']/UF['a']
    #print(p['LQCD'])    
    #print(gv.evalcorr([p['LQCD'],F['a']]))
    p['LQCD'] = 0.5
    plt.figure(3)
    
    for j in range(len(qsq)):
        f0physpole = 0
        f0phys = 0
        fplusphyspole = 0
        fplusphys = 0
        if qsq[j] == 0.0:
            Z.append(0)
            Zmean.append(0)
        else:
            Z.append((gv.sqrt(tplus-qsq[j])-gv.sqrt(tplus))/(gv.sqrt(tplus-qsq[j])+gv.sqrt(tplus)))
            Zmean.append(Z[j].mean)       
        for n in range(N):
            a['0'][n] = 0
            a['plus'][n] = 0
            for i in range(3):
                if AddRho == True:
                    a['0'][n] += p['0d'][i][0][0][n] * (p['LQCD']/MBsphys)**i * (1 + p['0rho'][n] * gv.log(MBsphys/MDsphys) )
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/MBsphys)**i * (1 + p['plusrho'][n] * gv.log(MBsphys/MDsphys))
                else:
                    a['0'][n] += p['0d'][i][0][0][n] * (p['LQCD']/MBsphys)**i
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/MBsphys)**i 
            if n == 0:
                f0physpole += (1/(1 - qsq[j]/(MBs0**2))) * a['0'][n]
                f0phys +=  a['0'][n]
                fplusphyspole += (1/(1 - qsq[j]/(MBsstarphys**2))) * a['plus'][n]
                fplusphys +=  a['plus'][n]
            else:
                f0physpole += (1/(1 - qsq[j]/(MBs0**2))) * Z[j]**n * a['0'][n]
                f0phys +=  Z[j]**n * a['0'][n]
                fplusphyspole += (1/(1 - qsq[j]/(MBsstarphys**2))) * (Z[j]**n - (n/N) * (-1)**(n-N) * Z[j]**N) * a['plus'][n]
                fplusphys += (Z[j]**n - (n/N) * (-1)**(n-N) * Z[j]**N) * a['plus'][n]
        if qsq[j] == 0.0:
            print('f_0(0):',f0physpole)
            print('f_+(0)/f_0(0):',fplusphyspole/f0physpole)
            if AddRho:               
                inputs = {'d0000':prior['0d'][0][0][0][0],'d1000':prior['0d'][1][0][0][0],'d1000':prior['0d'][1][0][0][0],'d2000':prior['0d'][2][0][0][0],'d0001':prior['0d'][0][0][0][1],'d1001':prior['0d'][1][0][0][1],'d1001':prior['0d'][1][0][0][1],'d2001':prior['0d'][2][0][0][1],'d0002':prior['0d'][0][0][0][2],'d1002':prior['0d'][1][0][0][2],'d1002':prior['0d'][1][0][0][2],'d2002':prior['0d'][2][0][0][2],'rho0':prior['0rho'][0],'rho1':prior['0rho'][1],'rho2':prior['0rho'][2],'data':f}
                
            else:
                inputs = {'d0000':prior['0d'][0][0][0][0],'d1000':prior['0d'][1][0][0][0],'d1000':prior['0d'][1][0][0][0],'d2000':prior['0d'][2][0][0][0],'d0001':prior['0d'][0][0][0][1],'d1001':prior['0d'][1][0][0][1],'d1001':prior['0d'][1][0][0][1],'d2001':prior['0d'][2][0][0][1],'d0002':prior['0d'][0][0][0][2],'d1002':prior['0d'][1][0][0][2],'d1002':prior['0d'][1][0][0][2],'d2002':prior['0d'][2][0][0][2],'data':f}
            outputs = {'f_0(0)':f0physpole}
            print(gv.fmt_errorbudget(outputs=outputs, inputs=inputs))
        F0mean[j] = f0phys.mean
        F0upp[j] = f0phys.mean + f0phys.sdev
        F0low[j] = f0phys.mean - f0phys.sdev
                
        F0meanpole[j] = f0physpole.mean
        F0upppole[j] = f0physpole.mean + f0physpole.sdev
        F0lowpole[j] = f0physpole.mean - f0physpole.sdev

        Fplusmean[j] = fplusphys.mean
        Fplusupp[j] = fplusphys.mean + fplusphys.sdev
        Fpluslow[j] = fplusphys.mean - fplusphys.sdev
                
        Fplusmeanpole[j] = fplusphyspole.mean
        Fplusupppole[j] = fplusphyspole.mean + fplusphyspole.sdev
        Fpluslowpole[j] = fplusphyspole.mean - fplusphyspole.sdev
    plt.plot(Zmean,F0meanpole, color='b',label='$f_0$')
    #plt.plot(Z,F0mean, color='b',linestyle='--',label='$f_0$ no pole')
    plt.fill_between(Zmean,F0lowpole,F0upppole, color='b',alpha=0.4)
    plt.xlabel('z',fontsize=30)
    #plt.ylabel('f',fontsize=20)
    plt.plot(Zmean,Fplusmeanpole, color='r',label='$f_+$')
    #plt.plot(Z,Fplusmean, color='r',linestyle='--',label='$f_+$ no pole')
    plt.fill_between(Zmean,Fpluslowpole,Fplusupppole, color='r', alpha=0.4)
    plt.text(-0.20,1.0,'$f_0(z)$',fontsize=30)
    plt.text(-0.07,0.9,'$f_+(z)$',fontsize=30)
    #plt.ylabel('f',fontsize=20)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(0.05))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([zmax,0])
    #plt.axes().set_ylim([0,3.3])

    plt.figure(4)
    plt.plot(qsq,F0meanpole, color='b',label='$f_0$')
    #plt.plot(qsqmean,F0mean, color='b',linestyle='--',label='$f_0$ no pole')
    plt.fill_between(qsq,F0lowpole,F0upppole, color='b',alpha=0.4)
    plt.xlabel('$q^2[GeV^2]$',fontsize=30)
    #plt.ylabel('f',fontsize=20)
    plt.plot(qsq,Fplusmeanpole, color='r',label='$f_+$')
    #plt.plot(qsqmean,Fplusmean, color='r',linestyle='--',label='$f_+$ no pole')
    plt.fill_between(qsq,Fpluslowpole,Fplusupppole, color='r',alpha=0.4)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([0,qsqmax.mean])
    #plt.axes().set_ylim([0,3.3])    
    plt.text(19,1.0,'$f_0(q^2)$',fontsize=30)
    plt.text(10,1.0,'$f_+(q^2)$',fontsize=30)

    ####################################################### M_h plots ###############
    lower = MDsphys.mean
    upper = MBsphys.mean
    Mh = np.linspace(lower,upper,nopts)
    a = collections.OrderedDict()
    a['0'] = [0]*N
    a['plus'] = [0]*N
    fplusqmaxmean = np.zeros((nopts))
    fplusqmaxupp = np.zeros((nopts))
    fplusqmaxlow = np.zeros((nopts))
    f0qmaxmean = np.zeros((nopts))
    f0qmaxupp = np.zeros((nopts))
    f0qmaxlow = np.zeros((nopts))
    f0q0mean = np.zeros((nopts))
    f0q0upp = np.zeros((nopts))
    f0q0low = np.zeros((nopts))
    fplusq0mean = np.zeros((nopts))
    fplusq0upp = np.zeros((nopts))
    fplusq0low = np.zeros((nopts))
    invbetamean = np.zeros((nopts))
    invbetaupp = np.zeros((nopts))
    invbetalow = np.zeros((nopts))
    deltamean = np.zeros((nopts))
    deltaupp =np.zeros((nopts))
    deltalow =np.zeros((nopts))
    for j in range(len(Mh)):
        tpl =(Mh[j]+Metasphys)**2
        MHs0 = Mh[j] + Del
        MHsstar = Mh[j] + x/Mh[j]
        qsqmax = (Mh[j]-Metasphys)**2
        zmax = ((gv.sqrt(tpl-qsqmax)-gv.sqrt(tpl))/(gv.sqrt(tpl-qsqmax)+gv.sqrt(tpl)))
        f0qmax = 0
        fplusqmax = 0
        for n in range(N):
            a['0'][n] = 0
            a['plus'][n] = 0
            for i in range(3):
                if AddRho == True:
                    a['0'][n] +=  p['0d'][i][0][0][n] * (p['LQCD']/Mh[j])**i *(1 + p['0rho'][n] * gv.log(Mh[j]/MDsphys))
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/Mh[j])**i * (1 + p['plusrho'][n] * gv.log(Mh[j]/MDsphys))
                else:
                    a['0'][n] +=  p['0d'][i][0][0][n] * (p['LQCD']/Mh[j])**i 
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/Mh[j])**i 

            if n==0:
                f0qmax += (1/(1 - qsqmax/(MHs0**2))) * a['0'][n]
                fplusqmax += (1/(1 - qsqmax/(MHsstar**2))) * a['plus'][n]
            else:
                f0qmax += (1/(1 - qsqmax/(MHs0**2))) * zmax**n * a['0'][n]
                fplusqmax += (1/(1 - qsqmax/(MHsstar**2))) * (zmax**n - (n/N) * (-1)**(n-N) * zmax**N) * a['plus'][n]
        f0q0 =  a['0'][0]
        fplusq0 = a['plus'][0]
        #print(gv.evalcorr([Metasphys,Metasphys]))
        #print(gv.evalcorr([Metasphys**2-Mh[j]**2,tpl]))
        
        
        #invbeta = (Metasphys**2-Mh[j]**2)/(fplusq0*2*tpl) * ( a['0'][0]/(MHs0**2) + a['0'][1])
        #delta = 1-((Metasphys**2-Mh[j]**2)/fplusq0 * (1/(2*tpl)) * (a['plus'][0]/MHsstar**2 - a['0'][0]/MHs0**2 + a['plus'][1] - a['0'][1]))
        invbeta = (Mh[j]**2-Metasphys**2) * ( 1/(MHsstar**2) - a['plus'][1]/(4*tpl*a['plus'][0]))
        delta = 1-((Mh[j]**2-Metasphys**2) * (1/MHsstar**2 - 1/MHs0**2 - a['plus'][1]/(4*tpl*a['plus'][0]) + a['0'][1]/(4*tpl*a['0'][0])))
        if Mh[j] == MBsphys.mean:
            print('f+ gradient =', invbeta*a['plus'][0]/(Mh[j]**2-Metasphys**2))
            print('f+ gradient- f0 gradient =', (1-delta)*a['plus'][0]/(Mh[j]**2-Metasphys**2))
            
        f0qmaxmean[j] = f0qmax.mean
        f0qmaxupp[j] = f0qmax.mean + f0qmax.sdev
        f0qmaxlow[j] = f0qmax.mean - f0qmax.sdev

        fplusqmaxmean[j] = fplusqmax.mean
        fplusqmaxupp[j] = fplusqmax.mean + fplusqmax.sdev
        fplusqmaxlow[j] = fplusqmax.mean - fplusqmax.sdev
                
        f0q0mean[j] = f0q0.mean
        f0q0upp[j] = f0q0.mean + f0q0.sdev
        f0q0low[j] = f0q0.mean - f0q0.sdev

        fplusq0mean[j] = fplusq0.mean
        fplusq0upp[j] = fplusq0.mean + fplusq0.sdev
        fplusq0low[j] = fplusq0.mean - fplusq0.sdev
        
        invbetamean[j] = invbeta.mean
        invbetaupp[j] = invbeta.mean + invbeta.sdev
        invbetalow[j] = invbeta.mean - invbeta.sdev

        deltamean[j] = delta.mean
        deltaupp[j] = delta.mean + delta.sdev
        deltalow[j] = delta.mean - delta.sdev
        
    plt.figure(5)
    plt.plot(Mh,f0qmaxmean, color='b',label='$f_0(q^2_{max})$')
    plt.fill_between(Mh,f0qmaxupp,f0qmaxlow, color='b',alpha=0.4)
    plt.plot(Mh,fplusqmaxmean, color='r',label='$f_+(q^2_{max})$')
    plt.fill_between(Mh,fplusqmaxupp,fplusqmaxlow, color='r',alpha=0.4)
    plt.plot(Mh,f0q0mean, color='k',label='$f_0(0)$')
    plt.fill_between(Mh,f0q0upp,f0q0low, color='k',alpha=0.4)
    #plt.plot(Mh,fplusq0mean, color='purple',label='$f_+(0)$')
    #plt.fill_between(Mh,fplusq0upp,fplusq0low, color='purple',alpha=0.6)
    plt.xlabel(r'$M_{H_s}[GeV]$',fontsize=30)
    #plt.ylabel('$f$',fontsize=20)
    plt.text(2.2,2.0,'$f_+(q^2_{max})$',fontsize=30)
    plt.text(4.2,1.1,'$f_0(q^2_{max})$',fontsize=30)
    plt.text(2.2,0.4,'$f_0(0)$',fontsize=30)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.plot([lower,lower],[-10,10],'k--',lw=1)
    plt.text(lower+0.01,0.1,'$M_{D_s}$',fontsize=20)
    plt.plot([upper,upper],[-10,10],'k--',lw=1)
    plt.text(upper+0.01,0.1,'$M_{B_s}$',fontsize=20)
    #plt.axes().set_xlim([lower,upper])
    plt.axes().set_ylim([0,3.3]) 

    plt.figure(6)

    plt.plot(Mh,invbetamean, color='b')
    plt.fill_between(Mh,invbetaupp,invbetalow, color='b',alpha=0.4)
    plt.plot(Mh,deltamean, color='r',label='$\delta$')
    plt.fill_between(Mh,deltaupp,deltalow, color='r',alpha=0.4)
    plt.xlabel(r'$M_{H_s}[GeV]$',fontsize=30)
    #plt.ylabel('$f$',fontsize=20)
    plt.text(3,1.7,r'$\beta^{-1}$',fontsize=30)
    plt.text(4,0.3,r'$\delta$',fontsize=30)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.plot([lower,lower],[-10,10],'k--',lw=1)
    plt.text(lower+0.01,-0.7,'$M_{D_s}$',fontsize=20)
    plt.plot([upper,upper],[-10,10],'k--',lw=1)
    plt.text(upper+0.01,-0.7,'$M_{B_s}$',fontsize=20)
    #plt.axes().set_xlim([lower,upper])
    plt.axes().set_ylim([-0.8,2.5]) 
    

    
    return(Zmean,Zmax,qsq,MBsphys,MBsstarphys,F0mean,F0upp,F0low,Fplusmean,Fplusupp,Fpluslow,F0meanpole,F0upppole,F0lowpole,Fplusmeanpole,Fplusupppole,Fpluslowpole)

#plot_results()


aDelta = collections.OrderedDict()
amb = collections.OrderedDict()

afm = ['0.1583(13)','0.1595(14)','0.1247(10)','0.1264(11)','0.0878(7)']
amb['1'] = [3.4,3.4,3.6,3.6]
amb['2'] = 3.4
amb['3'] = 2.8
amb['4'] = 2.8
amb['5'] = 1.95
aDelta['1'] = ['0.310(11)','0.317(11)','0.300(15)','0.315(11)']
aDelta['2'] = '0.299(17)'
aDelta['3'] = '0.215(17)'
aDelta['4'] = '0.253(8)'
aDelta['5'] = '0.1708(48)'


def convert_Gev(a):
    hbar = '6.58211928(15)'
    c = 2.99792458
    aGev = gv.gvar(a)/(gv.gvar(hbar)*c*1e-2)
    return(aGev)

def findDelta():
    x = []
    y = []
    mean =[]
    upper= []
    lower = []
    colours =['r','g','g','b','k','purple']
    labels = ['Very Coarse', 'Coarse','Coarse', 'Fine','Fine','Superfine']
    plt.figure()
    #Delta = collections.OrderedDict()
    for c , Fit in enumerate(Fits):        
        p = gv.load(Fit['filename'])
        for mass in Fit['masses']:
            Delta = (p['dE:o{0}'.format(Fit['Gm{0}'.format(mass)])][0]-p['dE:{0}'.format(Fit['Gm{0}'.format(mass)])][0])/Fit['a']
            x.append((float(mass)/Fit['a']))
            y.append(Delta)
            
            plt.errorbar((float(mass)/Fit['a']).mean, Delta.mean, yerr=Delta.sdev,fmt='o', mfc='none', color=colours[c+4],label=(labels[c+4]))
    plt.xlabel('Heavy Mass (GeV)')
    plt.ylabel('Delta (GeV)')

    
    for i in range(4):
        a = convert_Gev(afm[1])
        delta = gv.gvar(aDelta['1'][i])/a
        mb = amb['1'][i]/a
        plt.errorbar((mb).mean, delta.mean, xerr=(mb).sdev, yerr=delta.sdev, fmt='o', mfc='none',color='r',label=(labels[0]))
        #print('mb(Gev)', a, 'Delta (Gev)', delta)
    for i in range(4):
        
        a = convert_Gev(afm[i+1])
        delta = gv.gvar(aDelta['{0}'.format(i+2)])/a
        mb = amb['{0}'.format(i+2)]/a
        plt.errorbar((mb).mean, delta.mean, xerr=(mb).sdev, yerr=delta.sdev,color=colours[i],fmt='o', mfc='none',label=(labels[i]))        
    plt.legend()

    prior = gv.BufferDict()
    prior['x'] = x
    prior['a'] = gv.gvar('-0.01(1)')
    prior['b'] = gv.gvar('0.36(2)')
 
    def func(p):
        return(p['a']*p['x']+p['b'])
    fit = lsqfit.nonlinear_fit(prior=prior, data=y, fcn=func)
    print(fit)
    p = fit.p
    p.pop('x',None)
    p['x'] = np.linspace(0.7,5,100)
    for i in range(100):
        mean.append(func(p)[i].mean)
        upper.append(func(p)[i].mean+func(p)[i].sdev)
        lower.append(func(p)[i].mean-func(p)[i].sdev)
    plt.plot(p['x'], mean, color='k',linestyle='--')
    p.pop('x',None)
    p['x'] = 4.18
    print(func(p))
    #plt.fill_between(p['x'],lower,upper, color='k',alpha=0.4)
    #print(func(p))
    plt.show()
    return()


#AddRho = True
#main()
#AddRho = False
main()

if SHOWPLOTS:
    plt.show()

#findDelta()

