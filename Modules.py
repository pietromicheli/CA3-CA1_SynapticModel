import Cleft_compartment
import Spine_compartment
from utility                    import *
from matplotlib                 import pyplot as plt
from pysb.simulator             import ScipyOdeSimulator
from scipy.integrate            import odeint
import pylab                    as pl
import numpy                    as np

Vr = -65# mV
Ve = 0 # mV
Rs = 500*1E6 # Ohm, *1
g_AMPA = 15*1E-12 # S, *1
g_NMDA = 40*1E-12 # S, *1
g_NMDA_Ca = -1/500 # -0.002uM/ms*mV, *2
# Mg = 1 # Mg concentration M

Ca_spine_init = 0. # 0 uM
e_Ca = 130 # mV

R = 8.31434 # J mol−1 K−1
T = 299.5 # K
F = 96486.7104 # Faraday constant, C/mol
z = +2
A = 6.022E23
e = 1.6E-19
Q = 6.2415E18 # number of charges in 1 Coulomb

# time constants
tau_Ca = 20*1E-3 # s (20 ms)
I_f = 0.75 # *2
I_s = 0.25 # *2
tau_f_bpap = 3*1E-3 # s, *2
tau_s_bpap = 25*1E-3 # s, *2
tau_rise_bpap = 0.5*1E-3 # s, *2
V_max_bpap = 67 # mV, *2
k_M = 0.093  # 0.093 1/mV, *2


def Stimulation_Pattern_Design (PARAMETERS, plot=False):

    '''
        Given the simulation parameters dictionary,
        plot the stimulation pattern that has been designed.

    '''

    n_step = int(PARAMETERS['T_total']/PARAMETERS['dt'])
    TIME = pl.linspace(0, PARAMETERS['T_total'], num=n_step)

    PRE_ = np.zeros(n_step)
    p1_PRE = 1/PARAMETERS['f1_PRE']
    p2_PRE = 1/PARAMETERS['f2_PRE']
    n2_PRE = 1+int(PARAMETERS['T_total']/p2_PRE)

    glu_t = PARAMETERS['Glu_time']
    glu_n = int(glu_t/PARAMETERS['dt'])

    PRE_t = []
    T_pre = PARAMETERS['T_PRE']
    for i in range(n2_PRE):
        t = T_pre
        for j in range(PARAMETERS['n1_PRE']):
            if t > PARAMETERS['T_total'] : break
            PRE_t.append(t)
            t += p1_PRE
        T_pre += p2_PRE

    for t in PRE_t:
        i = np.where(TIME>=t)
        PRE_[i[0][0]:i[0][0]+glu_n] = 1
    PRE_[-1] = 0

    POST_ = np.zeros(n_step)
    p1_POST = 1/PARAMETERS['f1_POST']
    p2_POST = 1/PARAMETERS['f2_POST']
    n2_POST = 1+int(PARAMETERS['T_total']/p2_POST)

    POST_t = []
    T_post = PARAMETERS['T_POST']
    for i in range(n2_POST):
        t = T_post
        for j in range(PARAMETERS['n1_POST']):
            if t > PARAMETERS['T_total'] : break
            POST_t.append(t)
            t += p1_POST
        T_post += p2_POST

    for t in POST_t:
        i = np.where(TIME>=t)
        POST_[i[0][0]]=1.3

    if plot:
        plt.figure('Stimulation Protocol',(9,5))
        plt.plot(TIME,PRE_,ds='steps',label='Pre-', c='black')
        plt.plot(TIME,POST_,ds='steps',label='Post-', c='red')
        plt.xlabel("Time (s)")
        ax = plt.gca()
        ax.yaxis.set_visible(False)
        ax.set_aspect('auto', anchor='SW')
        ax.fill(TIME,PRE_,'k')
        plt.legend(loc='upper right')
        plt.show()

    results = {'PRE':PRE_,'POST':POST_}

    return results

def Receptors_Gating_Simulation (PARAMETERS, plot=False):

    T_total = PARAMETERS['T_total']
    T_PRE = PARAMETERS['T_PRE']
    dt = PARAMETERS['dt']
    n_step = int(T_total/dt)
    TIME = pl.linspace(0, T_total, num=n_step)

    # PRE-SYNAPTIC STIMULATION PARAMETERS
    p1_PRE = 1/PARAMETERS['f1_PRE']
    p2_PRE = 1/PARAMETERS['f2_PRE']
    n1_PRE = PARAMETERS['n1_PRE']
    n2_PRE = 1+int(T_total/p2_PRE)

    Glu = PARAMETERS['Glu']
    Glu_time = PARAMETERS['Glu_time']

    AMPA_tot = 1
    NMDA_tot = 1

    print('\nSimulating Synaptic Cleft Compartment ...', end =" ")

    time_pre = TIME[int(T_PRE/dt):]
    mymodel = Cleft_compartment.network(Glu, Glu_time, n1_PRE, n2_PRE, p1_PRE, p2_PRE,PARAMETERS['pKd'])

    #N.B set 'atol' and 'rtol' parameters >0 for a more generous handling of integation's error
    atol = 1E-6
    rtol = 1E-5

    try:
        sim = ScipyOdeSimulator(mymodel, integrator='vode', compiler='cython', integrator_options={'atol': atol, 'rtol': rtol}, verbose=False)
    except:
        print('\nWARNING: Generated code is too complex to be executed with Cython compiler. Python compiler will be used istead.')
        sim = ScipyOdeSimulator(mymodel, integrator='vode', compiler='python', integrator_options={'atol': atol, 'rtol': rtol}, verbose=False)

    simres = sim.run(tspan=time_pre)
    print('Done!')

    yout = simres.all

    # add zero-values time points at the begin of the simulated trajectories in order to set their sizes equal to TIME array's one
    zeros = np.zeros(len(TIME[:int(T_PRE/dt)]))

    P_AMPA = np.concatenate((zeros,yout['obs_AMPA_OA2']))
    O_AMPA = P_AMPA*PARAMETERS['AMPA_tot']
    D1_AMPA = np.concatenate((zeros,yout['obs_AMPA_D1']))
    D2_AMPA = np.concatenate((zeros,yout['obs_AMPA_D2']))
    D3_AMPA = np.concatenate((zeros,yout['obs_AMPA_D3']))

    D_AMPA_total = []
    for d1,d2,d3 in zip(D1_AMPA,D2_AMPA,D3_AMPA):
        D_AMPA_total.append(d1+d2+d3)
    D_AMPA = [r/AMPA_tot for r in D_AMPA_total]

    P_NMDA = np.concatenate((zeros,yout['obs_NMDA_O1']))
    O_NMDA = P_NMDA*PARAMETERS['NMDA_tot']
    D1_NMDA = np.concatenate((zeros,yout['obs_NMDA_D1']))
    D2_NMDA = np.concatenate((zeros,yout['obs_NMDA_D2']))

    D_NMDA_total = []
    for d1,d2 in zip(D1_NMDA,D2_NMDA):
        D_NMDA_total.append(d1+d2)
    D_NMDA = [r/NMDA_tot for r in D_NMDA_total]


    results = {
                'O_AMPA':O_AMPA,
                'O_NMDA':O_NMDA,
                'P_AMPA': O_AMPA,
                'P_NMDA': O_NMDA,
                'D_AMPA': D_AMPA,
                'D_NMDA': D_NMDA,
                'D1_NMDA': D1_NMDA,
                'D2_NMDA': D2_NMDA
              }


    if plot:

        plt.figure('Receptors Gating',(9,5))
        plt.plot(TIME,O_AMPA,c='black',label='AMPA')
        plt.plot(TIME,O_NMDA,c='gray',label='NMDA')
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Receptors")
        plt.legend()
        plt.show()

    return results

def EPSCs_EPSPs_Calculation (PARAMETERS,O_AMPA,O_NMDA, plot=False):

    T_total = PARAMETERS['T_total']
    n_step = int(PARAMETERS['T_total']/PARAMETERS['dt'])

    T_PRE = PARAMETERS['T_PRE']

    T_POST = PARAMETERS['T_POST']
    f1_POST = PARAMETERS['f1_POST']
    p1_POST = 1/PARAMETERS['f1_POST']
    n1_POST = PARAMETERS['n1_POST']
    p2_POST = 1/PARAMETERS['f2_POST']
    n2_POST = 1+int(PARAMETERS['T_total']/p2_POST)

    Mg = PARAMETERS['Mg']
    Vr = PARAMETERS['Vr']

    TIME = pl.linspace(0, T_total, num=n_step)

    EPSC_AMPA = []
    EPSP_AMPA = []
    EPSC_NMDA = []
    EPSP_NMDA = []
    Ca_spine = []
    bAP = []
    Vm = []
    EPSC_tot=[]
    EPSP_total=[]
    Mg_block = []

    def intergrate_Ca(Ca,t,I):
        tau = tau_Ca
        dCadt = I*1E3-Ca/tau_Ca
        return dCadt

    # Create the time vector for the bAP stimulation pattern
    bAPs_timevector = []
    if f1_POST > 0:
        T = T_POST
        for i in range(n2_POST):
            t = T
            for j in range(n1_POST):
                if t > T_total : break
                bAPs_timevector.append(t)
                t += p1_POST
            T += p2_POST
    else :
        bAPs_timevector.append(-1)


    print('\nEPSCs/EPSPs Calculation is running ...')

    i=0
    j=0
    sample_rate = 1 # set a step value to use for sample the trajectories of AMPA and NMDA coming from RGS module
    printProgressBar(0, len(TIME[::int(sample_rate)]), prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i,t in enumerate(TIME[::int(sample_rate)]):
        k = i*int(sample_rate)

        # bAP_
        if t < T_POST:
            bAP_=0
            bAP.append(Vr)
        else:
            t_post = bAPs_timevector[j]
            if len(bAPs_timevector) > j+1:
                if t >= bAPs_timevector[j+1]:
                    j+=1

            bAP_ = V_max_bpap * \
                   ( I_f*np.exp(-(t-t_post)/tau_f_bpap) + \
                   I_s*np.exp(-(t-t_post)/tau_s_bpap))
                   # * (1-np.exp(-(t-t_post)/tau_rise_bpap)) # rise phase
            bAP.append(Vr+bAP_)

        if t < T_PRE:

            # We assume that at t=0 Vm is at resting potential and the net current flow is 0 since the glutamate has not still diffused
            EPSC_AMPA.append(0)
            EPSC_NMDA.append(0)
            EPSC_tot.append(0)

            Ca_spine.append(Ca_spine_init)

            EPSP_AMPA.append(Vr)
            EPSP_NMDA.append(Vr)
            Vm.append(Vr)

        else:

            O_ampa = O_AMPA[k]
            O_nmda = O_NMDA[k]
            P_ampa = O_AMPA[k]/PARAMETERS['AMPA_tot']
            P_nmda = O_NMDA[k]/PARAMETERS['NMDA_tot']

            Is_ampa = O_ampa*g_AMPA*(Vm[i-1]-Ve) *1E-3
            Is_nmda = O_nmda*g_NMDA*(Vm[i-1]-Ve)*B *1E-3
            Is_nmda_Ca = O_nmda*g_NMDA_Ca*(Vm[i-1]-e_Ca)*B
            Is_tot = Is_ampa+Is_nmda

            epsp_ampa = Rs*(-Is_ampa) *1E3
            epsp_nmda = Rs*(-Is_nmda) *1E3

            Vm_tot = Vr+epsp_ampa+epsp_nmda+bAP_

            # integrate the Ca2+ variation in the spine
            ts = [TIME[k-1], t]
            Ca_0 = Ca_spine[i-1]
            Ca = odeint(intergrate_Ca,Ca_0,ts,args=(Is_nmda_Ca,))

            EPSC_AMPA.append(Is_ampa)
            EPSC_NMDA.append(Is_nmda)
            EPSC_tot.append(Is_tot)
            Ca_spine.append(Ca[1])

            EPSP_AMPA.append(Vr+epsp_ampa)
            EPSP_NMDA.append(Vr+epsp_nmda)

            Vm.append(Vm_tot)

        # Mg blocking function
        B = 1/(1+np.exp(-k_M*Vm[i])*(Mg/3.57))
        Mg_block.append(B)

        i+=1
        if i % 100 == 0:
            # print progress bar
            printProgressBar(i/100, len(TIME[::int(sample_rate)][::100]), prefix = 'Progress:', suffix = 'Complete', length = 50)

    print('Done!')


    EPSC_AMPA = [i*1E12 for i in EPSC_AMPA] # converts from A to pA
    EPSC_NMDA = [i*1E12 for i in EPSC_NMDA] # converts from A to pA
    EPSC_tot = [i*1E12 for i in EPSC_tot] # converts from A to pA


    results = {
                'EPSC_AMPA':EPSC_AMPA,
                'EPSP_AMPA':EPSP_AMPA,
                'EPSC_NMDA':EPSC_NMDA,
                'EPSP_NMDA':EPSP_NMDA,
                'Ca_spine': Ca_spine,
                'bAP':bAP,
                'Vm':Vm,
                'Mg_block':Mg_block
             }

    if plot :

        plt.figure('EPSPs-ESPCs',(9,13))
        #
        plt.subplot(4,1,2)
        plt.plot(TIME, EPSP_AMPA, label="EPSP_AMPA", c='black')
        plt.plot(TIME, EPSP_NMDA, label="EPSP_NMDA", c='gray')
        plt.plot(TIME, bAP, label="bAP")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("mV")
        #
        plt.subplot(4,1,3)
        plt.plot(TIME, Vm, label="Vm", c='red')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("mV")
        #
        plt.subplot(4,1,1)
        plt.plot(TIME, EPSC_AMPA, label="EPSC_AMPA",c='black')
        plt.plot(TIME, EPSC_NMDA, label="EPSC_NMDA",c='gray')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("pA")
        #
        plt.subplot(4,1,4)
        plt.plot(TIME, Ca_spine, label=" Spine Ca$^{2+}$", c='navy')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("\u03BCM")
        #
        plt.show()

    return results

def CaMKII_Activation_Simulation (PARAMETERS, Ca_spine, plot=False):

    print('\nSimulating Spine Compartment ...')

    n_step = int(PARAMETERS['T_total']/PARAMETERS['dt'])
    TIME = pl.linspace(0, PARAMETERS['T_total'], num=n_step)

    init_cond = {'CaM_init': PARAMETERS['CaM_init'] , 'CaMKII_init': PARAMETERS['CaMKII_init'], 'Ca_init': Ca_spine[0],}

    mymodel = Spine_compartment.network(init_cond)

    atol = 1E-6
    rtol = 1E-5
    sim = ScipyOdeSimulator(mymodel, integrator='vode', compiler='cython', integrator_options={'atol': atol, 'rtol': rtol}, verbose=False)

    sample_rate = int(1E3) # set a step value to use for sample the trajectory of Ca_spine coming from CPC module
    # Ca_spine = [i*1E3 for i in Ca_spine] #test
    Ca_spine_ = Ca_spine[0::sample_rate]
    TIME_ = TIME[0::sample_rate]

    simres = sim.run(tspan=[TIME_[0],TIME_[1]])
    full_trajectories = simres.species

    printProgressBar(0, len(Ca_spine_[1:-1]), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i, Ca in enumerate(Ca_spine_[1:-1]):

        initials = simres.species[-1]
        initials[2] = Ca
        simres = sim.run(tspan=[TIME_[i],TIME_[i+1]], initials=initials, param_values={'Ca_init':Ca},num_processors=1)
        full_trajectories = np.concatenate((full_trajectories,np.array([simres.species[-1]])),axis=0)

        printProgressBar(i+1, len(Ca_spine_[1:-1]), prefix = 'Progress:', suffix = 'Complete', length = 50)

    print('Done!')


    ### Plot Trajectories ###

    # create a dict specie -> index
    species_indices = {}
    for i,specie in enumerate(mymodel.species):
        species_indices.update({str(specie) : i})

    # specify here the species you want to plot: specie : []
    query_species = {
                    "CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1')":[],
                    "CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1')":[],
                    "CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1')":[],
                    "CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1')":[],
                    "CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1')":[],
                    "CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1')":[],
                    "CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1')":[],
                    "Ca()":[]
                    }

    for specie in query_species:
        i = species_indices[specie]
        values = []
        for iteration in full_trajectories:
            values.append(iteration[i])
        query_species.update({specie:values})


    # Sum up all the phosphorilated species of CAMKII
    query_species.update({'pCamKII_total': np.sum([query_species["CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1')"],\
                                                                   query_species["CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1')"],\
                                                                   query_species["CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1')"],\
                                                                   query_species["CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1')"],\
                                                                   query_species["CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1')"],\
                                                                   query_species["CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1')"],\
                                                                   query_species["CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1')"]],\
                                                                   axis=0)})

    if plot:
        plt.figure('CaM/CaMKII Activation Siumlation', (9,10))
        plt.subplot(2,1,1)
        plt.plot(TIME_, query_species["Ca()"], label="Ca_in", c='black')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("\u03BCM")
        #
        plt.subplot(2,1,2)
        plt.plot(TIME_, query_species["pCamKII_total"], label="Active CaMKII", c='purple')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("\u03BCM")
        #
        plt.show()

    return query_species
