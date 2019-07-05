import os, sys
import numpy as np
import inspect

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

def printAttributes(object,name,title):
#
# List of all attributes of 'object' for checking:
#
    attrList = inspect.getmembers(object)
    strTitle = "\nattrList ("+name+" = "+title+"):\n{}\n"
    print strTitle.format(attrList)

def plotParamLens(s_center,knll,cnll,title0,title1):
# Another way - use gridspec
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    
    ax0 = plt.subplot(gs[0])
    plt.plot(s_center,1.e6*knll,'-x',color='r')
    ax0.set_xlabel('s, m')
    ax0.set_ylabel('10^6 * knll, m')
    ax0.set_title(title0,color='m',fontsize=14)
    ax0.grid(True)

    ax1 = plt.subplot(gs[1])
    plt.plot(s_center,cnll,'-x',color='r')
    ax1.set_xlabel('s, m')
    ax1.set_ylabel('cnll, m^1/2')
    ax1.set_title(title1,color='m',fontsize=14)
    ax1.grid(True)
       
    fig.tight_layout()
    plt.show()
    return

class NonlinearInsertion(object):
  
    # Generation of the nonlinear lenses as set of segments of the nonlinear insertion
    #
    # Source: 
    #   1) Nonlinear Accelerator Lattices with One and Two Analytic Invariants.
    #      V. Danilov and S. Nagaitsev. Phys. Rev. ST Accel. Beams 13, 084002 (2010);
    #      https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.13.084002.  
    #   2) Complex Representation of Potentials and Fields for the Nonlinear 
    #      Magnetic Insert of the Integrable Optics Test Accelerator.
    #      Chad Mitchell. March 2017; https://esholarship.org/uc/item/7dt4t236.
    #
    # Input attributes:
    #   length:     the length of the nonlinear inserttion (float, m);
    #   phase:      the phase advance modulo 2pi through the nonlinear insertion;
    #   t:          the nonlinear strength parameter (float, dimensionless, defaults to 0.1);
    #   c:          the nonlinear aperture parameter (float, m^1/2, is defined by poles in the x-axis, defaults to 0.01);
    #   num_lens:   the number of lonlinear lenses as  an parts of the nonlinear insertion (int, defaults to 18).
    #   
    # Output attributes:
    #   s_vals (ndArray): coordinates of the center of each nonlinear lens (float ndArray, m);
    #   knll (ndArray):   "strength" of each nonlinear lens (float ndArray, m);
    #   cnll (ndArray):   aperture parameters for each nonlinear lens (float ndArray, m^1/2).
    #   
    def __init__(self, length, phase, t = 0.1, c = 0.01, num_lens = 18):
        print "Input data:\nlength = ",length,", phase = ",phase,", t = ",t,", c = ",c,", num_lens = ",num_lens
        self.length = length
        self.phase = phase
        self.t = t
        self._c = c
        self.num_lens = num_lens
    # Aperture parameter c must be positive:
    @property
    def c(self):
        return self._c
    @c.setter
    def c(self, cval):
        if cval < 0:
            raise ValueError("Aperture parameter c must be positive")     
        self._c = c
    def generate_lens(self):

    # Focal length f0 of the insertion (m):
        f0 = self.length/4.0*(1.0+1.0/np.tan(np.pi*self.phase)**2)
        print "f0 = ",f0
    # Coordinates s_vals of the center of each nonlinear lens (m):
        first_lens = .5*(self.length/self.num_lens)
        last_lens = self.length - first_lens
        s_vals = np.linspace(first_lens,last_lens,self.num_lens) 
        self.s_vals = s_vals
        
    # Set the beta-functions (m):
        beta_n = self.length*(1.-s_vals*(self.length-s_vals)/self.length/f0)/np.sqrt(1.0-(1.0-self.length/2.0/f0)**2)
#        self.betas = beta_n
        
        cnll = self.c*np.sqrt(beta_n)
        self.cnll = cnll

        knn = self.t*self.length/self.num_lens/beta_n**2
        knll = knn*cnll**2
        self.knll = knll

        return self
        
def validate_lens(self, beta_values):
#
# Method to valdate parameters of the nonlinear lens from method
# 'generate_lens', using approach from S. Romanov MADX-structure
# of the IOTA ring.
#
    return

# Verifying of the script:    
l0 = 1.8
mu0 = .3
cval = .01
tval = .4
num_lens = 18

#-------- Only for checking: ------------
# insertionNL  = NonlinearInsertion(l0, mu0, tval, cval, num_lens)
# To recognize attributes of 'insertionNL':
# printAttributes(insertionNL,'insertionNL','NonlinearInsertion(l0, mu0, cval, tval)')
# dataInsertion = insertionNL.generate_lens()
# To recognize attributes of '':
# printAttributes(dataInsertion,'dataInsertion','insertionNL.generate_lens()')
#-------End of checking -------------

dataInsertion = NonlinearInsertion(l0, mu0, tval, cval, num_lens).generate_lens()
coords_lens = dataInsertion.s_vals
knll_lens = dataInsertion.knll
cnll_lens = dataInsertion.cnll
print "Output data:"
print "coords_lens = ",coords_lens
print "knll_lens = ",knll_lens
print "cnll_lens = ",cnll_lens

title = "Nonlinear Insertion: L={:.1f} m, phase= {:.2f}, t={:.2f}, c={:.2f} m^1/2".format(l0,mu0,tval,cval)
print "title = ",title
plotParamLens(coords_lens,knll_lens,cnll_lens,title,title)

#
# Result shows, that input value 'tval' strength ('knll') of the set of lenses corresponds to the
# central lens(es) of nonlinear insertion!
#




#-------- Only for checking of LaTex in Python: ------------
from IPython.display import display, Math
title1 = r'Nonlinear Insertion: L={}'.format(l0)+' m, \mu_0={}'.format(mu0)+r' t={}'.format(tval)
title1 += r', '+r'c={}'.format(cval)+r'\sqrt{m}'
print "title1 = ",Math(title1)
display(Math(title1))
            

