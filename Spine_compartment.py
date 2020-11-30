from sympy import *
from pysb import *
from pysb.macros import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np

import math

CaMKII_states = [
                 #CaMKII-CaM complexes
                 'APO','CaMKII_CaM0','CaMKII_CaM1C','CaMKII_CaM2C','CaMKII_CaM1N','CaMKII_CaM2N',\
                 'CaMKII_CaM1C1N','CaMKII_CaM1C2N','CaMKII_CaM2C1N','CaMKII_CaM4',\

                 #CaMKII-CaMKII complexes
                 'CaMKII_CaM0_CaMKII_CaM0','CaMKII_CaM0_CaMKII_CaM1C','CaMKII_CaM0_CaMKII_CaM2C',\
                 'CaMKII_CaM0_CaMKII_CaM1N','CaMKII_CaM0_CaMKII_CaM2N','CaMKII_CaM0_CaMKII_CaM1C1N',\
                 'CaMKII_CaM0_CaMKII_CaM2C1N','CaMKII_CaM0_CaMKII_CaM1C2N','CaMKII_CaM0_CaMKII_CaM4',\

                 'CaMKII_CaM1C_CaMKII_CaM0','CaMKII_CaM1C_CaMKII_CaM1C','CaMKII_CaM1C_CaMKII_CaM2C',\
                 'CaMKII_CaM1C_CaMKII_CaM1N','CaMKII_CaM1C_CaMKII_CaM2N','CaMKII_CaM1C_CaMKII_CaM1C1N',\
                 'CaMKII_CaM1C_CaMKII_CaM2C1N','CaMKII_CaM1C_CaMKII_CaM1C2N','CaMKII_CaM1C_CaMKII_CaM4',\

                 'CaMKII_CaM2C_CaMKII_CaM0','CaMKII_CaM2C_CaMKII_CaM1C','CaMKII_CaM2C_CaMKII_CaM2C',\
                 'CaMKII_CaM2C_CaMKII_CaM1N','CaMKII_CaM2C_CaMKII_CaM2N','CaMKII_CaM2C_CaMKII_CaM1C1N',\
                 'CaMKII_CaM2C_CaMKII_CaM2C1N','CaMKII_CaM2C_CaMKII_CaM1C2N','CaMKII_CaM2C_CaMKII_CaM4',\

                 'CaMKII_CaM1N_CaMKII_CaM0','CaMKII_CaM1N_CaMKII_CaM1C','CaMKII_CaM1N_CaMKII_CaM2C',\
                 'CaMKII_CaM1N_CaMKII_CaM1N','CaMKII_CaM1N_CaMKII_CaM2N','CaMKII_CaM1N_CaMKII_CaM1C1N',\
                 'CaMKII_CaM1N_CaMKII_CaM2C1N','CaMKII_CaM1N_CaMKII_CaM1C2N','CaMKII_CaM1N_CaMKII_CaM4',\

                 'CaMKII_CaM2N_CaMKII_CaM0','CaMKII_CaM2N_CaMKII_CaM1C','CaMKII_CaM2N_CaMKII_CaM2C',\
                 'CaMKII_CaM2N_CaMKII_CaM1N','CaMKII_CaM2N_CaMKII_CaM2N','CaMKII_CaM2N_CaMKII_CaM1C1N',\
                 'CaMKII_CaM2N_CaMKII_CaM2C1N','CaMKII_CaM2N_CaMKII_CaM1C2N','CaMKII_CaM2N_CaMKII_CaM4',\

                 'CaMKII_CaM1C1N_CaMKII_CaM0','CaMKII_CaM1C1N_CaMKII_CaM1C','CaMKII_CaM1C1N_CaMKII_CaM2C',\
                 'CaMKII_CaM1C1N_CaMKII_CaM1N','CaMKII_CaM1C1N_CaMKII_CaM2N','CaMKII_CaM1C1N_CaMKII_CaM1C1N',\
                 'CaMKII_CaM1C1N_CaMKII_CaM2C1N','CaMKII_CaM1C1N_CaMKII_CaM1C2N','CaMKII_CaM1C1N_CaMKII_CaM4',\

                 'CaMKII_CaM2C1N_CaMKII_CaM0','CaMKII_CaM2C1N_CaMKII_CaM1C','CaMKII_CaM2C1N_CaMKII_CaM2C',\
                 'CaMKII_CaM2C1N_CaMKII_CaM1N','CaMKII_CaM2C1N_CaMKII_CaM2N','CaMKII_CaM2C1N_CaMKII_CaM1C1N',\
                 'CaMKII_CaM2C1N_CaMKII_CaM2C1N','CaMKII_CaM2C1N_CaMKII_CaM1C2N','CaMKII_CaM2C1N_CaMKII_CaM4',\

                 'CaMKII_CaM1C2N_CaMKII_CaM0','CaMKII_CaM1C2N_CaMKII_CaM1C','CaMKII_CaM1C2N_CaMKII_CaM2C',\
                 'CaMKII_CaM1C2N_CaMKII_CaM1N','CaMKII_CaM1C2N_CaMKII_CaM2N','CaMKII_CaM1C2N_CaMKII_CaM1C1N',\
                 'CaMKII_CaM1C2N_CaMKII_CaM2C1N','CaMKII_CaM1C2N_CaMKII_CaM1C2N','CaMKII_CaM1C2N_CaMKII_CaM4',\

                 'CaMKII_CaM4_CaMKII_CaM0','CaMKII_CaM4_CaMKII_CaM1C','CaMKII_CaM4_CaMKII_CaM2C',\
                 'CaMKII_CaM4_CaMKII_CaM1N','CaMKII_CaM4_CaMKII_CaM2N','CaMKII_CaM4_CaMKII_CaM1C1N',\
                 'CaMKII_CaM4_CaMKII_CaM2C1N','CaMKII_CaM4_CaMKII_CaM1C2N','CaMKII_CaM4_CaMKII_CaM4',\
                 ]


def network(init_cond):

    Model()

    #MONOMERS

    Monomer('CaM', ['CaM_b1','CaM_s'], {'CaM_s':['CaM0','CaM1C','CaM2C','CaM1N','CaM2N','CaM1C1N','CaM1C2N','CaM2C1N','CaM4']})
    #N.B : 'CaMKII' is a single monomeric subunit of the whole dodecameric CaMKII structure.
    Monomer('CaMKII', ['CaMKII_b1','CaMKII_s','CaMKII_p'],
           {'CaMKII_s': CaMKII_states,'CaMKII_p':['p0','p1']})
    Monomer('Ca')



    #INITIAL CONDITIONS
    Initial(CaM(CaM_b1=None, CaM_s='CaM0'), Parameter('CaM_init', init_cond['CaM_init'])) # uM
    Initial(CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0'), Parameter('CaMKII_init', init_cond['CaMKII_init'])) # uM
    Initial(Ca(), Parameter('Ca_init', init_cond['Ca_init'])) # uM

    #PARAMETERS

    # counter
    # Parameter('counter_speed', 1)
    # Observable('time', counter())

    #Ca binding CaM
    Parameter('CaM_1C_on', 4.000) # 1/(uM*s)
    Parameter('CaM_1C_off', 40.000) # 1/s
    Parameter('CaM_2C_on', 10.000) # 1/(uM*s)
    Parameter('CaM_2C_off', 9.250) # 1/s
    Parameter('CaM_1N_on', 100.000) # 1/(uM*s)
    Parameter('CaM_1N_off', 2500.000) # 1/s
    Parameter('CaM_2N_on', 150.000) # 1/(uM*s)
    Parameter('CaM_2N_off', 750.000) # 1/s
    #CaMKII dimerization
    Parameter('CaMKII2_on', 50) # 1/(uM*s)
    Parameter('CaMKII2_off', 60) # 1/s
    Parameter('CaMKII_pCaMKII_on', 50) # 1/(uM*s)
    Parameter('CaMKII_pCaMKII_off', 60) # 1/s
    #CaM binding CaMKII
    Parameter('CaMKII_CaM0_on', 0.0038) # 1/(uM*s)
    Parameter('CaMKII_CaM0_off', 5.5) # 1/s
    Parameter('CaMKII_CaM1C1N_on', 3.3) # 1/(uM*s)
    Parameter('CaMKII_CaM1C1N_off', 3.4) # 1/s
    Parameter('CaMKII_CaM1C2N_on', 1.9) # 1/(uM*s)
    Parameter('CaMKII_CaM1C2N_off', 1.9) # 1/s
    Parameter('CaMKII_CaM1C_on', 0.059) # 1/(uM*s)
    Parameter('CaMKII_CaM1C_off', 6.8) # 1/s
    Parameter('CaMKII_CaM1N_on', 0.022) # 1/(uM*s)
    Parameter('CaMKII_CaM1N_off', 3.1) # 1/s
    Parameter('CaMKII_CaM2C1N_on', 5.2) # 1/(uM*s)
    Parameter('CaMKII_CaM2C1N_off', 3.8) # 1/s
    Parameter('CaMKII_CaM2C_on', 0.92) # 1/(uM*s)
    Parameter('CaMKII_CaM2C_off', 6.8) # 1/s
    Parameter('CaMKII_CaM2N_on', 0.1) # 1/(uM*s)
    Parameter('CaMKII_CaM2N_off', 1.7) # 1/s
    Parameter('CaMKII_CaM4_on', 30) # 1/(uM*s)
    Parameter('CaMKII_CaM4_off', 1.7) # 1/s
    #Ca binding CaM-CaMKII complex
    Parameter('CaMKII_CaM_1C_on', 44) # 1/(uM*s)
    Parameter('CaMKII_CaM_1C_off', 33) # 1/s
    Parameter('CaMKII_CaM_1N_on', 75) # 1/(uM*s)
    Parameter('CaMKII_CaM_1N_off', 300) # 1/s
    Parameter('CaMKII_CaM_2C_on', 44) # 1/(uM*s)
    Parameter('CaMKII_CaM_2C_off', 2.7) # 1/s
    Parameter('CaMKII_CaM_2N_on', 76) # 1/(uM*s)
    Parameter('CaMKII_CaM_2N_off', 33) # 1/s
    #CaMKII autophosphorylation
    Parameter('pCaMKII_CaM0', 0) # 1/s
    Parameter('pCaMKII_CaM1C', 0.032) # 1/s
    Parameter('pCaMKII_CaM1C1N', 0.094) # 1/s
    Parameter('pCaMKII_CaM1C2N', 0.154) # 1/s
    Parameter('pCaMKII_CaM1N', 0.060) # 1/s
    Parameter('pCaMKII_CaM2C', 0.064) # 1/s
    Parameter('pCaMKII_CaM2C1N', 0.124) # 1/s
    Parameter('pCaMKII_CaM2N', 0.120) # 1/s
    Parameter('pCaMKII_CaM4', 0.960) # 1/s
    # CaMKII + PPI
    Parameter('pCaMKII_PPI_on', 3.0) # 1/(uM*s)
    Parameter('pCaMKII_PPI_off', 0.5) # 1/s
    Parameter('pCaMKII_dephosph', 2.0) # 1/s
    #RULES

    # let Counter flow
    # Rule('counter_increment', None >> counter(), counter_speed)

    ###### Working only for Simulate_alfa !!!
    # Ca inflow
    # Parameter('Ca_inflow_k', 0)
    # Rule('Ca_inflow', None >> Ca(), Ca_inflow_k)
    # # Ca outflow
    # Parameter('Ca_outflow_k', 1/0.02)
    # Rule('Ca_outflow', Ca() >> None, Ca_outflow_k)
    ######

    # Ca binding CaM (reactions 1-24)

    Rule('CaM0_Ca_C', CaM(CaM_b1=None, CaM_s='CaM0') + Ca()  | CaM(CaM_b1=None, CaM_s='CaM1C') , CaM_1C_on, CaM_1C_off)
    Rule('CaM1C_Ca_C', CaM(CaM_b1=None, CaM_s='CaM1C') + Ca() | CaM(CaM_b1=None, CaM_s='CaM2C') , CaM_2C_on, CaM_2C_off)
    Rule('CaM0_Ca_N', CaM(CaM_b1=None, CaM_s='CaM0') + Ca() | CaM(CaM_b1=None, CaM_s='CaM1N') , CaM_1N_on, CaM_1N_off)
    Rule('CaM1N_Ca_N', CaM(CaM_b1=None, CaM_s='CaM1N') + Ca() | CaM(CaM_b1=None, CaM_s='CaM2N') , CaM_2N_on, CaM_2N_off)
    Rule('CaM1C_Ca_N', CaM(CaM_b1=None, CaM_s='CaM1C') + Ca() | CaM(CaM_b1=None, CaM_s='CaM1C1N') , CaM_1N_on, CaM_1N_off)
    Rule('CaM1C1N_Ca_N', CaM(CaM_b1=None, CaM_s='CaM1C1N') + Ca() | CaM(CaM_b1=None, CaM_s='CaM1C2N') , CaM_2N_on, CaM_2N_off)
    Rule('CaM2C_Ca_N', CaM(CaM_b1=None, CaM_s='CaM2C') + Ca() | CaM(CaM_b1=None, CaM_s='CaM2C1N') , CaM_1N_on, CaM_1N_off)
    Rule('CaM2C1N_Ca_N', CaM(CaM_b1=None, CaM_s='CaM2C1N') + Ca() | CaM(CaM_b1=None, CaM_s='CaM4') , CaM_2N_on, CaM_2N_off)
    Rule('CaM1N_Ca_C', CaM(CaM_b1=None, CaM_s='CaM1N') + Ca() | CaM(CaM_b1=None, CaM_s='CaM1C1N') , CaM_1C_on, CaM_1C_off)
    Rule('CaM1C1N_Ca_C', CaM(CaM_b1=None, CaM_s='CaM1C1N') + Ca() | CaM(CaM_b1=None, CaM_s='CaM2C1N') , CaM_2C_on, CaM_2C_off)
    Rule('CaM2N_Ca_C', CaM(CaM_b1=None, CaM_s='CaM2N') + Ca() | CaM(CaM_b1=None, CaM_s='CaM1C2N') , CaM_1C_on, CaM_1C_off)
    Rule('CaM1C2N_Ca_C', CaM(CaM_b1=None, CaM_s='CaM1C2N') + Ca() | CaM(CaM_b1=None, CaM_s='CaM4') , CaM_2C_on, CaM_2C_off)

    #CaM binding CaMKII (reactions 49-66)

    Rule('CaM0_CaMKII', CaM(CaM_b1=None, CaM_s='CaM0') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0'), CaMKII_CaM0_on, CaMKII_CaM0_off)
    Rule('CaM1C_CaMKII', CaM(CaM_b1=None, CaM_s='CaM1C') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0'), CaMKII_CaM1C_on, CaMKII_CaM1C_off)
    Rule('CaM2C_CaMKII', CaM(CaM_b1=None, CaM_s='CaM2C') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'), CaMKII_CaM2C_on, CaMKII_CaM2C_off)
    Rule('CaM1N_CaMKII', CaM(CaM_b1=None, CaM_s='CaM1N') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0'), CaMKII_CaM1N_on, CaMKII_CaM1N_off)
    Rule('CaM2N_CaMKII', CaM(CaM_b1=None, CaM_s='CaM2N') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0'), CaMKII_CaM2N_on, CaMKII_CaM2N_off)
    Rule('CaM1C1N_CaMKII', CaM(CaM_b1=None, CaM_s='CaM1C1N') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0'), CaMKII_CaM1C1N_on, CaMKII_CaM1C1N_off)
    Rule('CaM1C2N_CaMKII', CaM(CaM_b1=None, CaM_s='CaM1C2N') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII_CaM1C2N_on, CaMKII_CaM1C2N_off)
    Rule('CaM2C1N_CaMKII', CaM(CaM_b1=None, CaM_s='CaM2C1N') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII_CaM2C1N_on, CaMKII_CaM2C1N_off)
    Rule('CaM4_CaMKII', CaM(CaM_b1=None, CaM_s='CaM4') + CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), CaMKII_CaM4_on, CaMKII_CaM4_off)

    #Ca binding CaM-CaMKII dimers (reactions 25-48)

    Rule('CaMKII_CaM0_Ca_C',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') , CaMKII_CaM_1C_on, CaMKII_CaM_1C_off)
    Rule('CaMKII_CaM1C_Ca_C',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') , CaMKII_CaM_2C_on, CaMKII_CaM_2C_off)
    Rule('CaMKII_CaM0_Ca_N',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') , CaMKII_CaM_1N_on, CaMKII_CaM_1N_off)
    Rule('CaMKII_CaM1N_Ca_N',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') , CaMKII_CaM_2N_on, CaMKII_CaM_2N_off)
    Rule('CaMKII_CaM1C_Ca_N',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') , CaMKII_CaM_1N_on, CaMKII_CaM_1N_off)
    Rule('CaMKII_CaM1N_Ca_C',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') , CaMKII_CaM_1C_on, CaMKII_CaM_1C_off)
    Rule('CaMKII_CaM2C_Ca_N',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') , CaMKII_CaM_1N_on, CaMKII_CaM_1N_off)
    Rule('CaMKII_CaM2N_Ca_C',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') , CaMKII_CaM_1C_on, CaMKII_CaM_1C_off)
    Rule('CaMKII_CaM1C1N_Ca_C',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') , CaMKII_CaM_2C_on, CaMKII_CaM_2C_off)
    Rule('CaMKII_CaM1C1N_Ca_N',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') , CaMKII_CaM_2N_on, CaMKII_CaM_2N_off)
    Rule('CaMKII_CaM2C1N_Ca_N',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') , CaMKII_CaM_2N_on, CaMKII_CaM_2N_off)
    Rule('CaMKII_CaM1C2N_Ca_C',CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + Ca() | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') , CaMKII_CaM_2C_on, CaMKII_CaM_2C_off)

    #CaM-CaMKII dimers + CaM-CaMKII dimers complexation (reactions 67-156)

    Rule('CaMKII_CaM0_CaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM0', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM0_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM1C_CaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C_CaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C_CaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C_CaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C_CaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM2C_CaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C_CaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C_CaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C_CaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM1N_CaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1N_CaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1N_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1N_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM2N_CaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2N_CaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2N_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2N_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2N_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM1C1N_CaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM2C1N_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2C1N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C1N_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM2C1N_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM1C2N_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C2N', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)
    Rule('CaMKII_CaM1C2N_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    Rule('CaMKII_CaM4_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM4', CaMKII_p='p0'), CaMKII2_on, CaMKII2_off)

    #CaM-CaMKII%CaM-CaMKII complexes autophosphorylation (reactions 157-237)

    Rule('CaMKII_CaM0_CaMKII_CaM1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM0_CaMKII_CaM2N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM0_CaMKII_CaM1C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM1C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM0_CaMKII_CaM1C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM1C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM0_CaMKII_CaM1C2N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM1C2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM0_CaMKII_CaM2C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM2C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM0_CaMKII_CaM2C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM0_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM0)
    Rule('CaMKII_CaM0_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM1N_CaMKII_CaM2N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C2N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM1C2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM1N_CaMKII_CaM2C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM2C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM1N_CaMKII_CaM2C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM1N_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM2N_CaMKII_CaM1C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM2N_CaMKII_CaM1C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM2N_CaMKII_CaM1C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM2N_CaMKII_CaM1C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM2N_CaMKII_CaM1C2N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM2N_CaMKII_CaM1C2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2N_CaMKII_CaM2C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM2N_CaMKII_CaM2C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM2N_CaMKII_CaM2C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM2N_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM2N_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM2N_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM1C_CaMKII_CaM1C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C_CaMKII_CaM1C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C_CaMKII_CaM1C2N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C_CaMKII_CaM1C2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM1C_CaMKII_CaM2C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C_CaMKII_CaM2C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM1C_CaMKII_CaM2C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM1C_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM1C1N_CaMKII_CaM1C2N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM1C2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM2C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM2C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM2C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM1C2N_CaMKII_CaM2C_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM1C2N_CaMKII_CaM2C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM1C2N_CaMKII_CaM2C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C2N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM1C2N_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C2N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM1C2N_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM1C2N_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM2C_CaMKII_CaM2C1N_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C1N', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM2C_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM2C1N_CaMKII_CaM4_p1', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM2C1N_CaMKII_CaM4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM4', CaMKII_p='p0')  >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    Rule('CaMKII_CaM0_CaMKII_CaM0_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM0', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1'), pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_CaMKII_CaM1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1'), pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_CaMKII_CaM2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1'), pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_CaMKII_CaM1C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1'), pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_CaMKII_CaM1C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1'), pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_CaMKII_CaM1C2N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C2N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1'), pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_CaMKII_CaM2C_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'), pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_CaMKII_CaM2C1N_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2C1N', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'), pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_CaMKII_CaMC4_p2', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM4', CaMKII_p='p0') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'), pCaMKII_CaM4)

    # #CaM-CaMKII dimers + pCaM-CaMKII phosphorylated dimers complexation (reactions 238-399)

    Rule('CaMKII_CaM0_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM0', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM0', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1C2N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM2C', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM2C1N', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    Rule('CaMKII_CaM0_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1N_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2N_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)
    Rule('CaMKII_CaM4_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') | CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM4', CaMKII_p='p1'), CaMKII_pCaMKII_on, CaMKII_pCaMKII_off)

    #pCaM-CaMKII%CaM-CaMKII complexes autophosphorylation (reactions 400-480)
    Rule('CaMKII_CaM0_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM0_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM0', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM1C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM1C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM1C2N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM1C2N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM2C_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM2C', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM2C1N_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM2C1N', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') ,pCaMKII_CaM4)

    Rule('CaMKII_CaM0_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM0', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM0)
    Rule('CaMKII_CaM1N_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM1N)
    Rule('CaMKII_CaM2N_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM2N)
    Rule('CaMKII_CaM1C_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM1C)
    Rule('CaMKII_CaM1C1N_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM1C1N)
    Rule('CaMKII_CaM1C2N_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM1C2N)
    Rule('CaMKII_CaM2C_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM2C)
    Rule('CaMKII_CaM2C1N_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM2C1N)
    Rule('CaMKII_CaM4_pCaMKII_CaM4_autophospho', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4_CaMKII_CaM4', CaMKII_p='p1') >> CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') + CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1') ,pCaMKII_CaM4)

    #OBSERVABLES
    
    Observable('obs_CaM0', CaM(CaM_b1=None, CaM_s='CaM0'))
    Observable('obs_CaM1C', CaM(CaM_b1=None, CaM_s='CaM1C'))
    Observable('obs_CaM1N', CaM(CaM_b1=None, CaM_s='CaM1N'))
    Observable('obs_CaM2C', CaM(CaM_b1=None, CaM_s='CaM2C'))
    Observable('obs_CaM2N', CaM(CaM_b1=None, CaM_s='CaM2N'))
    Observable('obs_CaM1C1N', CaM(CaM_b1=None, CaM_s='CaM1C1N'))
    Observable('obs_CaM1C2N', CaM(CaM_b1=None, CaM_s='CaM1C2N'))
    Observable('obs_CaM2C1N', CaM(CaM_b1=None, CaM_s='CaM2C1N'))
    Observable('obs_CaM4', CaM(CaM_b1=None, CaM_s='CaM4'))

    Observable('obs_CaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p0'))
    Observable('obs_CaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p0'))
    Observable('obs_CaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p0'))
    Observable('obs_CaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p0'))
    Observable('obs_CaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p0'))
    Observable('obs_CaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p0'))
    Observable('obs_CaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p0'))
    Observable('obs_CaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p0'))


    Observable('obs_pCaMKII_CaM1C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C', CaMKII_p='p1'))
    Observable('obs_pCaMKII_CaM2C', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C', CaMKII_p='p1'))
    Observable('obs_pCaMKII_CaM1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1N', CaMKII_p='p1'))
    Observable('obs_pCaMKII_CaM2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2N', CaMKII_p='p1'))
    Observable('obs_pCaMKII_CaM1C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C1N', CaMKII_p='p1'))
    Observable('obs_pCaMKII_CaM2C1N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM2C1N', CaMKII_p='p1'))
    Observable('obs_pCaMKII_CaM1C2N', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM1C2N', CaMKII_p='p1'))
    Observable('obs_pCaMKII_CaM4', CaMKII(CaMKII_b1=None, CaMKII_s='CaMKII_CaM4', CaMKII_p='p1'))

    Observable('obs_CaMKII_APO', CaMKII(CaMKII_b1=None, CaMKII_s='APO', CaMKII_p='p0'))
    Observable('obs_Ca', Ca())

    return model
