import os, sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

def lawsMagnification(mgnfctnFctr,steps):

# 1) Linear: for step number n
#           t(n) = t_0 + (t_f-t_0)*n/(N-1) for n = 0,1,...,N-1 .
    tLin = np.zeros(steps)
    for n in range(steps):
        tLin[n] = 1.+n*(mgnfctnFctr-1.)/(steps-1)
# 2) Parabolic: for step number n
#           t(n) = t_0 + (t_f-t_0)*n^2/(N-1)^2 for n = 0,1,...,N-1 .
    tPar= np.zeros(steps)
    for n in range(steps):
        tPar[n] = 1.+n**2*(mgnfctnFctr-1.)/(steps-1)**2
# 3) Smooth sign-function: for step number n
#           t(n) = .5*(t_0+t_f) + .5*(t_f-t_0)*tanh(x(n)), where
#           x(n) = (6*n-3*(N-1))/(N-1) for n=0,1,...,N-1 .
# In this approach x(0) = -3., x(N-1) = 3.; so, tanh(3.) = - tanh(-3.) = .9951
    tSSF= np.zeros(steps)
    for n in range(steps):
        x = (6.*n-3.*(steps-1))/(steps-1)
        tSSF[n] = .5*(1.+mgnfctnFctr)+.5*(mgnfctnFctr-1.)*np.tanh(x)
# Plotting all cases:
    step = range(steps)
    tMin = .975*min(tLin)
    tMax = 1.025*max(tLin)
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
    
    ax0 = plt.subplot(gs[0])
    plt.plot(step,tLin,'-x',color='r')
    x0Title = 'Linear magnification'
    ax0.set_title(x0Title,fontsize='16')
    ax0.set_xlim([-1,steps+1])
    ax0.set_ylim([tMin,tMax])
    ax0.set_xlabel('Step n')
    ax0.set_ylabel('t')
    ax0.grid(True)
    
    ax1 = plt.subplot(gs[1])
    plt.plot(step,tPar,'-x',color='r')
    x1Title = 'Parabolic magnification'
    ax1.set_title(x1Title,fontsize='16')
    ax1.set_xlim([-1,steps+1])
    ax1.set_ylim([tMin,tMax])
    ax1.set_xlabel('Step n')
    ax1.set_ylabel('t')
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[2])
    plt.plot(step,tSSF,'-x',color='r')
    x2Title = 'Smooth sign-function magnification'
    ax2.set_title(x2Title,fontsize='16')
    ax2.set_xlim([-1,steps+1])
    ax2.set_ylim([tMin,tMax])
    ax2.set_xlabel('Step n')
    ax2.set_ylabel('t')
    ax2.grid(True)

    fig.tight_layout()
    plt.show()
    return
  
steps = 21
factor = .85
lawsMagnification(factor,steps)



