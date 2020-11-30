
''' UTILITY FUNCTIONS '''

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np
import pylab as pl


def single_exp_fit (curve, dt, max=None, min=None):

    ''' given:
              - a numpy array of values representing a spike curve,
              - the time resolution dt of the curve

        return the time constant of the exponential function that describe its decay phase, starting from the maximum value
    '''
    if max == None:
        # Start the fitting from the last occurrence of the maximum value in the passed array
        max_i = len(curve)-np.argmax(np.flip(curve))-1
        max = curve[max_i]

    else :
        max_i = max
        max = curve[max_i]

    if min == None:
        min_i = np.argmin(curve[max_i:])

    else:
        min_i = min

    decay_phase = curve[max_i: min_i]

    npoints = len(curve)
    # t = pl.linspace(0, dt*(npoints-max_i-(npoints-min_i)), num=npoints-max_i-(npoints-min_i))
    t = pl.linspace(0,len(decay_phase)*dt,num=len(decay_phase))


    def func(x, l):
        y = max * np.exp(-x/l)
        return y

    popt, pcov = curve_fit(func, t, decay_phase)
    tao = popt[0]

    fitted_curve = np.zeros(max_i)
    fitted_curve[:] = np.nan
    for i in range(len(decay_phase)):
        t = i*dt
        # print(decay_phase[0]*np.exp(-t/tao))
        fitted_curve=np.append(fitted_curve,decay_phase[0]*np.exp(-t/tao))

    fitted_curve=np.append(fitted_curve,np.zeros(len(curve[min_i:])))
    fitted_curve[min_i:] = np.nan

    return tao, fitted_curve

def double_exp_fit (curve, dt):

    time = pl.linspace(0,len(curve)*dt,len(curve))

    def func(t,a_f,t_f,t_s):

        a_s = 1-a_f
        y = curve[0]*(a_f*np.exp(-t/t_f)+a_s*np.exp(-t/t_s))

        return y

    popt, pcov = curve_fit(func,time,curve,bounds=[0,[1,np.inf,np.inf]])

    if popt[1]<popt[2]:
        t_fast = popt[1]
        t_slow = popt[2]
        a_fast = popt[0]
        a_slow = 1-popt[0]
    else :
        t_fast = popt[2]
        t_slow = popt[1]
        a_fast = 1-popt[0]
        a_slow = popt[0]

    param = {
                'A_fast' : a_fast,
                'Tau_fast' : t_fast,
                'A_slow' : a_slow,
                'Tau_slow' : t_slow,
            }

    fitted_curve = curve[0]*(popt[0]*np.exp(-time/popt[1])+(1-popt[0])*np.exp(-time/popt[2]))

    return param,fitted_curve

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
