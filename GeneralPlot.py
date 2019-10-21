import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import collections
Masses = collections.OrderedDict()
Twists = collections.OrderedDict()
################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['filename'] = 'Fits/F5_3pts_Q1.00_Nexp1_NMarg0.7334202569879142_Stmin2_Vtmin2_Ttmin2_svd0.02261_chi0.733_pl1.0_svdfac1.0'
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
############################################################################

Fits = [F]#,SF]                                         # Choose to fit F, SF or UF
Masses['F'] = [0,1,2]#0,1,2,3]                                     # Choose which masses to fit
Twists['F'] = [0,1,2]#0,1,2,3,4]#,5]
Masses['SF'] = [0,1,2,3]
Twists['SF'] = [0,1,2,3,4]
GeV = False  # Cannot change at the current time
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



def plot_f(Fit,F_0,F_plus,F_T,qSq,Z,Sca,Vec,Ten):
    
    plt.figure(1)
    for k in range(len(Fit['masses'])):
        z = []
        zerr = []
        f = []
        ferr = []
        for i in range(len(qSq[Fit['masses'][k]])):
            z.append(Z[Fit['masses'][k]][i].mean)
            f.append(F_0[Fit['masses'][k]][i].mean)
            zerr.append(Z[Fit['masses'][k]][i].sdev)
            ferr.append(F_0[Fit['masses'][k]][i].sdev)        
        plt.errorbar(z,f,xerr=[zerr,zerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel('$f_0$')


    plt.figure(2)
    for k in range(len(Fit['masses'])):
        q = []
        qerr = []
        f = []
        ferr = []
        for i in range(len(qSq[Fit['masses'][k]])):
            q.append(qSq[Fit['masses'][k]][i].mean)
            f.append(F_0[Fit['masses'][k]][i].mean)
            qerr.append(qSq[Fit['masses'][k]][i].sdev)
            ferr.append(F_0[Fit['masses'][k]][i].sdev)        
        plt.errorbar(q,f,xerr=[qerr,qerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()
    if GeV == True:
        plt.xlabel('$q^2 (GeV)^2$')
    else:
        plt.xlabel('$q^2$ (lattice units)')        
    plt.ylabel('$f_0$')


    plt.figure(3)           
    for k in range(len(Fit['masses'])):
        z = []
        zerr = []
        f = []
        ferr = []
        for i in range(1,len(qSq[Fit['masses'][k]])):            
                z.append(Z[Fit['masses'][k]][i].mean)
                f.append(F_plus[Fit['masses'][k]][i-1].mean)
                zerr.append(Z[Fit['masses'][k]][i].sdev)
                ferr.append(F_plus[Fit['masses'][k]][i-1].sdev)        
        plt.errorbar(z,f,xerr=[zerr,zerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()    
    plt.xlabel('$z$')
    plt.ylabel('$f_+$')
    #plt.xlim(right=1.1*qSq[masses[k]][0].mean)


    plt.figure(4)           
    for k in range(len(Fit['masses'])):
        q = []
        qerr = []
        f = []
        ferr = []
        for i in range(1,len(qSq[Fit['masses'][k]])):            
                q.append(qSq[Fit['masses'][k]][i].mean)
                f.append(F_plus[Fit['masses'][k]][i-1].mean)
                qerr.append(qSq[Fit['masses'][k]][i].sdev)
                ferr.append(F_plus[Fit['masses'][k]][i-1].sdev)        
        plt.errorbar(q,f,xerr=[qerr,qerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()
    
    if GeV == True:
        plt.xlabel('$q^2 (GeV)^2$')
    else:
        plt.xlabel('$q^2$ (lattice units)')
    plt.ylabel('$f_+$')

    plt.figure(5)           
    for k in range(len(Fit['masses'])):
        z = []
        zerr = []
        f = []
        ferr = []
        for i in range(1,len(qSq[Fit['masses'][k]])):            
                z.append(Z[Fit['masses'][k]][i].mean)
                f.append(F_T[Fit['masses'][k]][i-1].mean)
                zerr.append(Z[Fit['masses'][k]][i].sdev)
                ferr.append(F_T[Fit['masses'][k]][i-1].sdev)        
        plt.errorbar(z,f,xerr=[zerr,zerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()    
    plt.xlabel('$z$')
    plt.ylabel('$f_T$')
    #plt.xlim(right=1.1*qSq[masses[k]][0].mean)


    plt.figure(4)           
    for k in range(len(Fit['masses'])):
        q = []
        qerr = []
        f = []
        ferr = []
        for i in range(1,len(qSq[Fit['masses'][k]])):            
                q.append(qSq[Fit['masses'][k]][i].mean)
                f.append(F_T[Fit['masses'][k]][i-1].mean)
                qerr.append(qSq[Fit['masses'][k]][i].sdev)
                ferr.append(F_T[Fit['masses'][k]][i-1].sdev)        
        plt.errorbar(q,f,xerr=[qerr,qerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()
    if GeV == True:
        plt.xlabel('$q^2 (GeV)^2$')
    else:
        plt.xlabel('$q^2$ (lattice units)')
    plt.ylabel('$f_T$')
    return()





def main():    
    make_params()
    for Fit in Fits:                       
        print('Plot for', Fit['filename'] )
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
        plot_f(Fit,F_0,F_plus,qSq,Z,Sca,Vec)
    plt.show()
    return()


main() 

    


#plot_f()


def speedtest():
    make_params()
    plt.figure()
    for Fit in Fits:        
        get_results(Fit)        
        csq = collections.OrderedDict()
        for i in range(len(Fit['twists'])):
            csq['{0}tw_{1}'.format(Fit['conf'],Fit['twists'][i])] = Fit['E_KG_tw{0}'.format(Fit['twists'][i])]**2/(Fit['momenta'][i]**2+Fit['E_KG_tw{0}'.format(Fit['twists'][0])]**2)
            plt.errorbar(float(Fit['twists'][i]),csq['{0}tw_{1}'.format(Fit['conf'],Fit['twists'][i])].mean, yerr = csq['{0}tw_{1}'.format(Fit['conf'],Fit['twists'][i])].sdev, label=('{0}tw_{1}'.format(Fit['conf'],Fit['twists'][i])),  fmt='o', mfc='none')
    plt.plot([0,4],[1,1],'k--',lw=1)
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('c')
    plt.show()
    return(csq)
speedtest()


def findDelta():
    plt.figure()
    #Delta = collections.OrderedDict()
    for Fit in Fits:        
        p = gv.load(Fit['filename'])
        for mass in Fit['masses']:
            Delta = (p['dE:o{0}'.format(Fit['BGm{0}'.format(mass)])][0]-p['dE:{0}'.format(Fit['BGm{0}'.format(mass)])][0])/Fit['a']
            print(Fit['conf'],mass, Delta)
            
            plt.errorbar(float(mass), Delta.mean, yerr=Delta.sdev,fmt='o', mfc='none', label=Fit['conf'])
    plt.xlabel('Heavy Mass')
    plt.ylabel('Delta (GeV)')
    plt.show()
    return()

findDelta()
