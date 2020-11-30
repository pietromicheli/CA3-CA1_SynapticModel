from pysb import *
from pysb.macros import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
from sympy import Piecewise
import math


def network(glu_conc, glu_exposure, n1, n2, p1, p2, pKd):

    Model()

# MONOMERS

    Monomer('NMDA', ['R_b1','R_b2','R_b3','R_b4','R_s'],
                 {'R_s':['R_r', 'R_U_Gly', 'R_U_Glu','R_M_Gly','R_M_Glu','R_C3','R_C2','R_C1','R_D1','R_D2','R_O1']})

    Monomer('AMPA', ['R_b1','R_b2','R_s'],
                 {'R_s':['R_C', 'R_CA1','R_CA2','R_D1','R_D2','R_D3','R_OA2']})

    Monomer('Gly', ['Gly_b','Gly_s'], {'Gly_s':['free','bound','dummy']})
    Monomer('Glu', ['Glu_b','Glu_s'], {'Glu_s':['free','bound','dummy']})

    Monomer('time')

# INITIAL

    Initial(NMDA(R_b1=None,R_b2=None,R_b3=30,R_b4=40,R_s='R_U_Glu')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), Parameter('R_U_Glu_init', 1))   ### initial concentration of resting receptors (2 Gly-bound)
    Initial(AMPA(R_b1=None,R_b2=None,R_s='R_C'), Parameter('AMPA_C_init', 1))



# PARAMETERS

    Parameter('time_increment', 1)
    Observable('time_count', time())

    #Simulation parameters
    Parameter('n1', n1)
    Parameter('n2', n2)
    Parameter('p1', p1)
    Parameter('p2', p2)

    Parameter('glu_cargo', glu_conc) #constant value of Glu cargo released in the synaptic cleft at each stimulation
    Parameter('glu_release', 1E6)

## NMDA

    ''' NMDA RECEPTOR :

        The rates constant courrently used refers to the average kinetic gating model for
        NMDA isoforms GluN1/GluN2B presented in the article:

        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2849062/#bib31

    '''

    #Glu,Gly binding rates

    Parameter('Kd', 10**(-pKd) *1E6) # uM

    #Recalculate K_off rate for glutamate binding, according to the mutant Kd
    Parameter('NMDA_kGlu1_on', 12)# 1/(uM*s)
    Parameter('NMDA_kGlu1_off', 15)# 1/s
    Parameter('NMDA_kGlu2_on', 6)# 1/(uM*s)
    Parameter('NMDA_kGlu2_off', 30)# 1/s
    Expression('Kd_wt', (NMDA_kGlu1_off+NMDA_kGlu2_off)/(NMDA_kGlu1_on+NMDA_kGlu2_on))

    Expression('NMDA_kGlu1_off_mut', NMDA_kGlu1_off+(Kd-Kd_wt)*NMDA_kGlu1_on)# 1/s
    Expression('NMDA_kGlu2_off_mut', NMDA_kGlu2_off+(Kd-Kd_wt)*NMDA_kGlu2_on)# 1/s
    Expression('NMDA_kGlu1_on_mut', NMDA_kGlu1_on)# 1/s
    Expression('NMDA_kGlu2_on_mut', NMDA_kGlu2_on)# 1/s


    # NMDA states transitions kinetik constants
    # on rates
    Parameter('kC3_C2_on', 204) # 1/s
    Parameter('kC3_D1_on', 3.5) # 1/s
    Parameter('kC2_D2_on', 7.9) # 1/s
    Parameter('kC2_C1_on', 484) # 1/s
    Parameter('kC1_O1_on', 1266) # 1/s
    # off rates
    Parameter('kC3_C2_off', 134) # 1/s
    Parameter('kC3_D1_off', 0.7) # 1/s
    Parameter('kC2_D2_off', 23) # 1/s
    Parameter('kC2_C1_off', 2138) # 1/s
    Parameter('kC1_O1_off', 274) # 1/s

## NMDA

    ''' AMPA RECEPTOR :

        The rates constant courrently used refers to the average kinetic gating model for
        AMPA isoforms GluN1/GluN2B presented in the article:

        https://www.jneurosci.org/content/20/6/2166

    '''

    #Glu,Gly binding rates

    Parameter('AMPA_kGlu1_on', 1.8)# 1/(uM*s) -> 1.8*1E6 1/(M*s)
    Parameter('AMPA_kGlu1_off', 2.4E3)# 1/s
    Parameter('AMPA_kGlu2_on', 10)# 1/(uM*s) -> 1.0*1E7 1/(M*s)
    Parameter('AMPA_kGlu2_off', 1E4)# 1/s

    # AMPA states transitions kinetik constants
    # on rates
    Parameter('kCA2_OA2_on', 1.6E4) # 1/s
    Parameter('kCA1_D1_on', 7.0E2) # 1/s
    Parameter('kCA2_D2_on', 1.0E2) # 1/s
    Parameter('kOA2_D3_on', 3.0E2) # 1/s
    Parameter('kD1_D2_on', 1E1) # 1/(uM*s) -> 1.0*1E7 1/(M*s)
    Parameter('kD2_D3_on', 1.6E4) # 1/s
    # off rates
    Parameter('kCA2_OA2_off', 5.0E3) # 1/s
    Parameter('kCA1_D1_off', 1.5E2) # 1/s
    Parameter('kCA2_D2_off', 2.1) # 1/s
    Parameter('kOA2_D3_off', 1.5E1) # 1/s
    Parameter('kD1_D2_off', 1E3) # 1/s
    Parameter('kD2_D3_off', 1.2E4) # 1/s



# Glutamate Pulses Modellization

    # Set monomers, parameters and reaction for the release of each cargo of Glu :
    # for each stimuli the release of a new Glu cargo from the vescicles is simulated

    for i in range(n2):

        t_i0 = i*p2
        t_i1 = (i+1)*p2

        for j in range(n1):

            t_j0 = t_i0 + j*p1
            t_j1 = t_j0 + p1

            name = 'Glu_cargo_'+str(i)+str(j)
            Monomer(name)
            Initial(eval(name)(), glu_cargo)
            Expression(name+'_rate_on', ((time_count >= t_j0))*glu_release)
            Rule(name+'_release', eval(name)() >> Glu(Glu_b=None, Glu_s='free'), eval(name+'_rate_on'))

            if j < n1-1 :
                Expression(name+'_clearance', (time_count>t_j0+glu_exposure)*(time_count<t_j1)*1E8)
            else:
                Expression(name+'_clearance', (time_count>t_j0+glu_exposure)*(time_count<t_i1)*1E8)

            Rule(name+'_clearing', Glu(Glu_b=None, Glu_s='free') >> None, eval(name+'_clearance'))

# RULES

    Rule('time_progression', None >> time(), time_increment)

    ############################################################################

    ''' NMDA Gaiting Reaction Mechanism '''

    Rule('NMDA_2Gly_2Gly1Glu',\
    NMDA(R_b1=None,R_b2=None,R_b3=30,R_b4=40,R_s='R_U_Glu')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') + Glu(Glu_b=None,Glu_s='free')| \
    NMDA(R_b1=10,R_b2=None,R_b3=30,R_b4=40,R_s='R_M_Glu')%Glu(Glu_b=10,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    NMDA_kGlu1_on_mut, NMDA_kGlu1_off_mut)

    Rule('NMDA_2Gly1Glu_C3',\
    NMDA(R_b1=10,R_b2=None,R_b3=30,R_b4=40,R_s='R_M_Glu')%Glu(Glu_b=10,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') + Glu(Glu_b=None,Glu_s='free')| \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    NMDA_kGlu1_on_mut, NMDA_kGlu2_off_mut)

    Rule('NMDA_C3_C2', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >>\
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC3_C2_on)
    Rule('NMDA_C2_C3', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC3_C2_off)

    Rule('NMDA_C3_D1', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC3_D1_on)
    Rule('NMDA_D1_C3', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC3_D1_off)

    Rule('NMDA_C2_D2', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC2_D2_on)
    Rule('NMDA_D2_C2', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC2_D2_off)

    Rule('NMDA_C2_C1', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC2_C1_on)
    Rule('NMDA_C1_C2', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC2_C1_off)

    Rule('NMDA_C1_O1', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_O1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC1_O1_on)
    Rule('NMDA_O1_C1', \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_O1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free') >> \
    NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'), \
    kC1_O1_off)

    ############################################################################

    ''' AMPA Gaiting Reaction Mechanism

            C == CA1 == CA2 == OA2
                  ||    ||     ||
                  D1 == D2  == D3
    '''

    Rule('AMPA_C_CA1',\
    AMPA(R_b1=None,R_b2=None,R_s='R_C')+ Glu(Glu_b=None,Glu_s='free') | \
    AMPA(R_b1=10,R_b2=None,R_s='R_CA1')%Glu(Glu_b=10,Glu_s='free'), \
    AMPA_kGlu1_on,AMPA_kGlu1_off)


    Rule('AMPA_CA1_CA2',\
    AMPA(R_b1=10,R_b2=None,R_s='R_CA1')%Glu(Glu_b=10,Glu_s='free') + Glu(Glu_b=None,Glu_s='free') | \
    AMPA(R_b1=10,R_b2=20,R_s='R_CA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    AMPA_kGlu2_on,AMPA_kGlu2_off)


    Rule('AMPA_CA2_OA2',\
    AMPA(R_b1=10,R_b2=20,R_s='R_CA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_OA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kCA2_OA2_on)
    Rule('AMPA_OA2_CA2',\
    AMPA(R_b1=10,R_b2=20,R_s='R_OA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_CA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kCA2_OA2_off)

    Rule('AMPA_CA1_D1',\
    AMPA(R_b1=10,R_b2=None,R_s='R_CA1')%Glu(Glu_b=10,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=None,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free'), \
    kCA1_D1_on)
    Rule('AMPA_D1_CA1',\
    AMPA(R_b1=10,R_b2=None,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=None,R_s='R_CA1')%Glu(Glu_b=10,Glu_s='free'), \
    kCA1_D1_off)

    Rule('AMPA_CA2_D2',\
    AMPA(R_b1=10,R_b2=20,R_s='R_CA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kCA2_D2_on)
    Rule('AMPA_D2_CA2',\
    AMPA(R_b1=10,R_b2=20,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_CA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kCA2_D2_off)

    Rule('AMPA_OA2_D3',\
    AMPA(R_b1=10,R_b2=20,R_s='R_OA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_D3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kOA2_D3_on)
    Rule('AMPA_D3_OA2',\
    AMPA(R_b1=10,R_b2=20,R_s='R_D3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_OA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kOA2_D3_off)

    Rule('AMPA_D1_D2',\
    AMPA(R_b1=10,R_b2=None,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free') + Glu(Glu_b=None,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kD1_D2_on)
    Rule('AMPA_D2_D1',\
    AMPA(R_b1=10,R_b2=20,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=None,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free') + Glu(Glu_b=None,Glu_s='free'), \
    kD1_D2_off)

    Rule('AMPA_D2_D3',\
    AMPA(R_b1=10,R_b2=20,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_D3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kD2_D3_on)
    Rule('AMPA_D3_D2',\
    AMPA(R_b1=10,R_b2=20,R_s='R_D3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free') >> \
    AMPA(R_b1=10,R_b2=20,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'), \
    kD2_D3_off)



# OBSERVABLES

 ## Glu
    Observable('obs_Glu', Glu(Glu_b=None, Glu_s='free'))

 ## NMDA
    Observable('obs_NMDA_O1', NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_O1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'))
    Observable('obs_NMDA_C3', NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_C3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'))
    Observable('obs_NMDA_D1', NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'))
    Observable('obs_NMDA_D2', NMDA(R_b1=10,R_b2=20,R_b3=30,R_b4=40,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free')%Gly(Gly_b=30,Gly_s='free')%Gly(Gly_b=40,Gly_s='free'))

 ## AMPA
    Observable('obs_AMPA_OA2', AMPA(R_b1=10,R_b2=20,R_s='R_OA2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'))
    Observable('obs_AMPA_D1', AMPA(R_b1=10,R_b2=20,R_s='R_D1')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'))
    Observable('obs_AMPA_D2', AMPA(R_b1=10,R_b2=20,R_s='R_D2')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'))
    Observable('obs_AMPA_D3', AMPA(R_b1=10,R_b2=20,R_s='R_D3')%Glu(Glu_b=10,Glu_s='free')%Glu(Glu_b=20,Glu_s='free'))


    return model
