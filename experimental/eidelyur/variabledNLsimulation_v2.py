# 
# This script develops the script 'variabledNLsimulation_v1.py' (Yury Eidelman)
#
#    Started at June 28, 2019
#
# The three laws to change the strengths 't' of all nonlinear lens are implemented.
# From initial value t_i to final value t_f during N stepsthese laws are follows.
# 1) Linear: for step number n
#           t(n) = t_0 + (t_f-t_0)*n/(N-1) for n = 0,1,...,N-1 .
# 2) Parabolic: for step number n
#           t(n) = t_0 + (t_f-t_0)*n^2/(N-1)^2 for n = 0,1,...,N-1 .
# 3) Smooth sign-function: for step number n
#           t(n) = .5*(t_0+t_f) + .5*(t_f-t_0)*tanh(x(n)), where
#           x(n) = (6*n-3*(N-1))/(N-1) for n=0,1,...,N-1 .
# In this approach x(0) = -3., x(N-1) = 3.; so, tanh(3.) = - tanh(-3.) = .9951
#

import synergia
import os, sys
import inspect
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

import rssynergia 
from rssynergia.base_diagnostics import lfplot
from rssynergia.base_diagnostics import plotbeam
from rssynergia.base_diagnostics import pltbunch

def plotcoordDistr(bunchParticles):
#    
# Plot X-X', Y-Y', and X-Y distributions for 'bunchParticles'
#
# bunchParticles is a 'bunch' object;
# particles is 2D array: (numberOfParticles,(x,x',y,y',s,dp(?),ID);
#
    numbPartcls = bunchParticles.shape[0]
    particles = bunchParticles.real
    newCoordinates = np.zeros((6,numbPartcls))
    for k in range(numbPartcls):
        for j in range(6):
            newCoordinates[j,k] = 1.e3*particles[k,j]       # Units: mm and mrad 
    xmax = 1.15*np.max(abs(newCoordinates[0,:]))
    xpmax = 1.15*np.max(abs(newCoordinates[1,:]))
    ymax = 1.15*np.max(abs(newCoordinates[2,:]))
    ypmax = 1.15*np.max(abs(newCoordinates[3,:]))
    meanX = np.mean(newCoordinates[0,:])
    meanPX = np.mean(newCoordinates[1,:])
    stdX = np.std(newCoordinates[0,:])
    stdPX = np.std(newCoordinates[1,:])
    meanY = np.mean(newCoordinates[2,:])
    meanPY = np.mean(newCoordinates[3,:])
    stdY = np.std(newCoordinates[2,:])
    stdPY = np.std(newCoordinates[3,:])

# Another way - use gridspec
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
    
    ax0 = plt.subplot(gs[0])
    plt.plot(newCoordinates[0,:],newCoordinates[2,:],'.',color='k')
    x0Title = "X,mm: <> = {:.3f} +- {:.3f}\nY,mm: <> = {:.3f} +- {:.3f}".format(meanX,stdX,meanY,stdY)
    ax0.set_title(x0Title,color='m',fontsize='16')
    ax0.set_xlim([-xmax,xmax])
    ax0.set_ylim([-ymax,ymax])
    ax0.set_xlabel('X, mm',color='m',fontsize='14')
    ax0.set_ylabel('Y, mm',color='m',fontsize='14')
    ax0.grid(True)
    
    ax1 = plt.subplot(gs[1])
    plt.plot(newCoordinates[0,:],newCoordinates[1,:],'.',color='b')
    x1Title = "X,mm: <> = {:.3f} +- {:.3f}\nX\',mrad: <> = {:.3f} +- {:.3f}".format(meanX,stdX,meanPX,stdPX)
    ax1.set_title(x1Title,color='m',fontsize='16')
    ax1.set_xlim([-xmax,xmax])
    ax1.set_ylim([-xpmax,xpmax])
    ax1.set_xlabel('X, mm',color='m',fontsize='14')
    ax1.set_ylabel('X\', mrad',color='m',fontsize='14')
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[2])
    plt.plot(newCoordinates[2,:],newCoordinates[3,:],'.',color='r')
    x2Title = "Y,mm: <> = {:.3f} +- {:.3f}\nY\',mrad: <> = {:.3f} +- {:.3f}".format(meanY,stdY,meanPY,stdPY)
    ax2.set_title(x2Title,color='m',fontsize='16')
    ax2.set_xlim([-ymax,ymax])
    ax2.set_ylim([-ypmax,ypmax])
    ax2.set_xlabel('Y, mm',color='m',fontsize='14')
    ax2.set_ylabel('Y\', mrad',color='m',fontsize='14')
    ax2.grid(True)
    
#    fig.canvas.set_window_title('Synergia Phase Space Distribution')
    fig.tight_layout()
    plt.show()
    return

def plotTracks(tracksCoords,numberTracks):
#    
# Plot'numberTracks' tracks from 'tracksCoords'
#
# tracksCoords is 3D array: (totalTurns,particles,(x,y))
#
#    print "numberTracks = ",numberTracks
    trackColor = ['r','b','k','m','g']
    numbPoints = tracksCoords.shape[0]
#    print "numbPoints = ",numbPoints
    xmax = 1.15*np.max(np.max(abs(tracksCoords[:,0:numberTracks,0])))
    ymax = 1.15*np.max(np.max(abs(tracksCoords[:,0:numberTracks,1])))

    turn = np.arange(0,numbPoints)
# Another way - use gridspec
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    
    ax0 = plt.subplot(gs[0])
    for prtcl in range(numberTracks):
        plt.plot(turn,tracksCoords[0:numbPoints,prtcl,0],'.-',color=trackColor[prtcl])
#    x0Title = "X,mm: <> = {:.3f} +- {:.3f}\nY,mm: <> = {:.3f} +- {:.3f}".format(meanX,stdX,meanY,stdY)
#    ax0.set_title(x0Title,color='m',fontsize='16')
    ax0.set_ylim([-xmax,xmax])
    ax0.set_xlabel('Turn',color='m',fontsize='14')
    ax0.set_ylabel('X, mm',color='m',fontsize='14')
    ax0.grid(True)

    ax1 = plt.subplot(gs[1])
    for prtcl in range(numberTracks):
        plt.plot(turn,tracksCoords[0:numbPoints,prtcl,1],'.-',color=trackColor[prtcl])
#    x0Title = "X,mm: <> = {:.3f} +- {:.3f}\nY,mm: <> = {:.3f} +- {:.3f}".format(meanX,stdX,meanY,stdY)
#    ax0.set_title(x0Title,color='m',fontsize='16')
    ax1.set_ylim([-ymax,ymax])
    ax1.set_xlabel('Turn',color='m',fontsize='14')
    ax1.set_ylabel('Y, mm',color='m',fontsize='14')
    ax1.grid(True)
       
#    fig.canvas.set_window_title('Synergia Phase Space Distribution')
    fig.tight_layout()
    plt.show()
    return

def plotParamLens(s_center,knll,cnll,title0,title1):
#
# Plot distribution of the strength 'knll' of the nonlinear lens inside
# nonlinear insertion:
#
    knll_plot = np.zeros(len(knll))
    for n in range(len(knll)):
        knll_plot[n]=1.e6*knll[n]
# Another way - use gridspec
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    
    ax0 = plt.subplot(gs[0])
    plt.plot(s_center,knll_plot,'-x',color='r')
    ax0.set_xlabel('s, m',color='m',fontsize=14)
    ax0.set_ylabel('10^6 * knll, m',color='m',fontsize=14)
    ax0.set_title(title0,color='m',fontsize=16)
    ax0.grid(True)

    ax1 = plt.subplot(gs[1])
    plt.plot(s_center,cnll,'-x',color='r')
    ax1.set_xlabel('s, m',color='m',fontsize=14)
    ax1.set_ylabel('cnll, m^1/2',color='m',fontsize=14)
    ax1.set_title(title1,color='m',fontsize=16)
    ax1.grid(True)
       
    fig.tight_layout()
    plt.show()
    return

def printAttributes(object,name,title):
#
# List of all attributes of 'object' for checking:
#
    attrList = inspect.getmembers(object)
    strTitle = "\nattrList ("+name+" = "+title+"):\n{}\n"
    print strTitle.format(attrList)

def tracksCoords(bunchParticles):
#
# Preparation of the track coordinates:
#
# 'bunchParticle' is a 'bunch' object;
# 'particles' is 2D array: (numberParrticles,(x,x',y,y',s,dE,ID));
#
    numbPartcls = bunchParticles.shape[0]
    particles = bunchParticles.real
    trackCoordinates = np.zeros((numbPartcls,2))
    for prtcl in range(numbPartcls):
        trackCoordinates[prtcl,0] = 1.e3*particles[prtcl,0]       # x, mm
        trackCoordinates[prtcl,1] = 1.e3*particles[prtcl,2]       # y, mm
#        if prtcl < 3:
#            print "Particle {}: x = {} mm, y = {} mm". \
#            format(prtcl,trackCoordinates[prtcl,0],trackCoordinates[prtcl,1])
    return trackCoordinates 
    
class NonlinearInsertion(object):
#  
# Generation of the nonlinear lenses as set of segments of the nonlinear insertion
#
# Source: 
#   1) Nonlinear Accelerator Lattices with One and Two Analytic Invariants.
#      V. Danilov and S. Nagaitsev. Phys. Rev. ST Accel. Beams 13, 084002 (2010);
#      https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.13.084002.  
#   2) Complex Representation of Potentials and Fields for the Nonlinear 
#      Magnetic Insert of the Integrable Optics Test Accelerator.
#      Chad Mitchell. March 2017; https://esholarship.org/uc/item/7dt4t236.
#   3) Madx CERN User Guide. Chapter 10.10 - Nonlinear Lens with Elliptical Potential.
#      http://mad.web.cern.ch/mad/
#
# Input args:
#   length:     the length of the nonlinear insertion (float, m);
#   phase:      the phase advance modulo 2pi through the nonlinear insertion;
#   t:          the strength parameter for center of the insertion (float, dimensionless, 
#               defaults to 0.1);
#   c:          the aperture parameter for center of the insertion 
#               (float, m^1/2, is defined by poles in the x-axis, defaults to 0.01);
#   num_lens:   the number of lonlinear lenses as an segments of the insertion (int, defaults to 20).   

#
# Output attributes are the same as input one.
#
    def __init__(self, length, phase, t = 0.1, c = 0.01, num_lens = 20):
        self.length = length
        self.phase = phase
        self.t = t
        self._c = c
        self.num_lens = num_lens
#        print "Input data for NonlinearInsertion:\nlength = ",self.length,", phase = ",self.phase, \
#              ", t = ",self.t,", c = ",self.c,", num_lens = ",self.num_lens
# Aperture parameter c must be positive:
    @property
    def c(self):
        return self._c
    @c.setter
    def c(self, cval):
        if cval < 0:
            raise ValueError("Aperture parameter c must be positive")     
        self._c = c
#
# Output attributes of 'generate_lens' method:
#
#   same as output of 'NonlinearInsertion'class and as well:
#   s_vals (ndArray): coordinates of the center of each nonlinear lens (float ndArray, m);
#   knll (ndArray):   "strength" of each nonlinear lens (float ndArray, m);
#   cnll (ndArray):   aperture parameters for each nonlinear lens (float ndArray, m^1/2).
#   
    def generate_lens(self,flag):
        indxShift = self.num_lens-2*((self.num_lens+1)/2)+1
# Focal length f0 of the insertion (m):
        f0 = self.length/4.0*(1.0+1.0/np.tan(np.pi*self.phase)**2)
#        print "f0 = ",f0
# Coordinates s_vals of the center of each nonlinear lens (m):
        first_lens = .5*(self.length/self.num_lens)
        last_lens = self.length - first_lens
        s_vals = np.linspace(first_lens,last_lens,self.num_lens) 
        self.s_vals = s_vals
        
# Set the structural beta-function of the nonlinear magnet (m):
        beta_n = self.length*(1.-s_vals*(self.length-s_vals)/self.length/f0)/ \
                 np.sqrt(1.0-(1.0-self.length/2.0/f0)**2)
#        self.betas = beta_n
        
        cnll = self.c*np.sqrt(beta_n)

        knn = self.t*self.length/self.num_lens/beta_n**2
        knll = knn*cnll**2
# Sequence of lenses start from the minimal value of knll (flag = 1):
        self.cnll = cnll
        self.knll = knll
# Sequence of lenses start from the maximal value of knll (flag = 2):
        if flag == 2:
            cnll_help = []
            knll_help = []
            indxMax = 0
            for n in range(self.num_lens-1):
                if knll[n] < knll[n+1]:
                    indxMax = n+1
                else:
                    break
            for n in range (self.num_lens):
                if n <= indxMax:
                    cnll_help.append(float(cnll[indxMax-n]))
                    knll_help.append(float(knll[indxMax-n]))
                else:
                    cnll_help.append(float(cnll[n-indxMax-indxShift]))
                    knll_help.append(float(knll[n-indxMax-indxShift]))
            self.cnll = cnll_help
            self.knll = knll_help
        return self
                
# Pickle helper is not necessary but is retained for this example
#
class Pickle_helper:
    __getstate_manages_dict__ = 1
    def __init__(self, *args):
        self.args = args
    def __getinitargs__(self):
        return self.args
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__ = state
        
# Definition of class to ramp nonlinear lens

class Ramp_actions(synergia.simulation.Propagate_actions, Pickle_helper):
#
# Args of 'Ramp_actions' method are: 
# 'type'             - type of magnification (1 - relative, 2 - absolute),
# 'stepNumber'       - current step of magnification,
# 'strengthLens'     - set of strengthes 't' of central lens of the nonlinear insertion for all steps of
#                      magnification (relative magnification) or set of strengthes 't' of all lenses for 
#                      current step (absolute magnification),
# 'updateOutputFlag' - flag to output the strength of one of nonlinear lens after it's magnification 
#                      for current step,
# controlName        - name of lens with maximal strength to use in output for checking of process 
#                     of magnification.    
#

#
# The arguments to __init__ are what the Ramp_actions instance is initialized with:
    def __init__(self, type,stepNumber,strengthLens,outputFlag,controlName):
        selfObject = synergia.simulation.Propagate_actions.__init__(self)
# To recognize attributes of 'selfObject':
#        printAttributes(selfObject,'selfObject','synergia.simulation.Propagate_actions.__init__(self)')

# Pickling the arguments to the initializer allows the module to resume
# after checkpointing. They should be in the same order as the arguments to __init__.
        Pickle_helper.__init__(self, type,stepNumber,strengthLens,outputFlag,controlName)
        self.type = type
        self.stepNumber = stepNumber
        self.strengthLens = strengthLens        
        self.outputFlag = outputFlag
        self.controlName = controlName
    
    def turn_end_action(self, stepper, bunch, turn_num):
#---------------------------
# For checking:
#        testObject = stepper.get_lattice_simulator().get_lattice()
# To recognize attributes of 'testObject':
#        printAttributes(testObject,'testObject','stepper.get_lattice_simulator().get_lattice()')
#        print "testName = '{}'".format(testObject.get_name())
#---------------------------

# Relative magnification:
        if self.type == 1:
            if self.stepNumber == 0:
                self.multiplier = self.strengthLens[0]
                print "Initialization lattice (relative magnification): Step ",self.stepNumber, \
                      ", multiplier = ",self.multiplier
            else:
                self.multiplier = self.strengthLens[self.stepNumber]/self.strengthLens[self.stepNumber-1]
# Output title for checking of variables update:   
                print "Modified lattice (relative magnification): Step ",self.stepNumber, \
                      ", multiplier = ",self.multiplier
            for element in stepper.get_lattice_simulator().get_lattice().get_elements():
# To recognize attributes of 'element':
#                printAttributes(element,'element', \
#                                    'stepper.get_lattice_simulator().get_lattice().get_elements()')
                if element.get_type() == "nllens":
                    old_knll = element.get_double_attribute("knll")
                    new_knll = self.multiplier*old_knll
                    element.set_double_attribute("knll", new_knll)
# Output for checking of variables update checking nonlinear lens 'n.11' only:  
                    if ((self.outputFlag == 1) and (element.get_name() == self.controlName)):
                        print element.get_name(),":  knll=",old_knll," --> ",new_knll
# Absolute magnification:
        if self.type == 2:
# Output title for checking of variables update:   
            print "Modified lattice (absolute magnification): Step ",self.stepNumber
            crrntLens = 0
            for element in stepper.get_lattice_simulator().get_lattice().get_elements():
# To recognize attributes of 'element':
#                printAttributes(element,'element', \
#                                    'stepper.get_lattice_simulator().get_lattice().get_elements()')
                if element.get_type() == "nllens":
                    old_knll = element.get_double_attribute("knll")
                    new_knll = self.strengthLens[crrntLens]
                    element.set_double_attribute("knll", new_knll)
                    crrntLens += 1
# Output for checking of variables update checking nonlinear lens 'n.11' only:  
                    if ((self.outputFlag == 1) and (element.get_name() == self.controlName)):
                        print element.get_name(),":  knll=",old_knll," --> ",new_knll
        stepper.get_lattice_simulator().update()

def t_on_knll_function(l0,mu0,cval,lensNumb):
#
# "Reverse" dependence dimensionless strength 'tval' of nonlinear central lens on 
# parameter 'knll' of this lens
#
    nPoints = 50
    knll = np.zeros(nPoints)
    t = np.zeros(nPoints)
    knll_logMin = math.log10(1.e-7)
    knll_logMax = math.log10(1.e-4)

# Focal length f0 of the insertion (m):
    f0 = l0/4.0*(1.0+1.0/np.tan(np.pi*mu0)**2)
#    print "f0 = ",f0," m"

# Coordinate of the centers of the nonlinear lenses (m):
    first_lens_center = .5*(l0/lensNumb)
    last_lens_center = l0 - first_lens_center
    s_vals = np.linspace(first_lens_center,last_lens_center,lensNumb) 
#    print "s_val =",s_vals        
# Coordinate of the center of the nonlinear lens in the middle of nonlinear inserion (m):
    s_center = s_vals[(num_lens+1)/2]
# Structural beta-function in the nonlinear magnet (m):
    beta_center = l0*(1.-s_center*(l0-s_center)/l0/f0)/np.sqrt(1.0-(1.0-l0/2.0/f0)**2)
    cnll_center = cval*np.sqrt(beta_center)
#    print "s_center = ",s_center," m, beta_center = ",beta_center," m, cnll_center = ",cnll_center," m"

    for n in range(nPoints):
        knll_log10 = knll_logMin + n*(knll_logMax - knll_logMin)/nPoints
        knll[n] = math.pow(10.,knll_log10)
        t[n] = knll[n]*beta_center**2/(l0/lensNumb*cnll_center**2)
    
    fig_10 = plt.figure(figsize=(15,5))
    gs_10 = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    
    ax_10 = plt.subplot(gs_10[0])
#    plt.semilogx(knll,t,'-x',color='r')
    plt.loglog(knll,t,'-x',color='r')
    ax_10.set_xlabel('knnl, m',color='m',fontsize=14)
    ax_10.set_ylabel('Srength Parameter of the central lens, t',color='m',fontsize=14)
# Attempt to change number of grid lines:
#    start, end = ax_10.get_xlim()
#    ax_10.xaxis.set_ticks(np.arange(start, end, (end-start)/30))

    title_t = "Nonlinear Insertion ({} lenses): L={:.2f} m, phase= {:.2f}, c={:.2f} m^1/2". \
               format(lensNumb,l0, mu0, cval)
    ax_10.set_title(title_t,color='m',fontsize=14)
    ax_10.grid(True)

    fig_10.tight_layout()
    plt.show()
    return

def lawsMagnification(t_i,t_f,steps):

# For relative magnification: t_i = 1., t_f = magnification: 
#
# Three laws of magnification are in use
#
# 1) Linear: for step number n
#       t(n) = t_i + (t_f-t_i)*n/(N-1) for n = 0,1,...,N-1 .
    tLin = np.zeros(steps)
    for n in range(steps):
        tLin[n] = t_i+n*(t_f-t_i)/(steps-1)
# 2) Parabolic: for step number n
#            t(n) = t_i + (t_f-t_i)*n^2/(N-1)^2 for n = 0,1,...,N-1 .
    tPar= np.zeros(steps)
    for n in range(steps):
        tPar[n] = t_i+n**2*(t_f-t_i)/(steps-1)**2
# 3) Smooth sign-function: for step number n
#           t(n) = .5*(t_f+t_i) + .5*(t_f-t_i)*tanh(x(n)), where
#           x(n) = (6*n-3*(N-1))/(N-1) for n=0,1,...,N-1 .
# In this approach x(0) = -3., x(N-1) = 3.; so, tanh(3.) = - tanh(-3.) = .9951
    tSSF= np.zeros(steps)
    for n in range(steps):
        x = (6.*n-3.*(steps-1))/(steps-1)
        tSSF[n] = .5*(t_f+t_i)+.5*(t_f-t_i)*np.tanh(x)
# Plotting all cases:
    step = range(steps)
    tMin = .975*min(tLin)
    tMax = 1.025*max(tLin)
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
    
    ax0 = plt.subplot(gs[0])
    plt.plot(step,tLin,'-x',color='r')
    x0Title = 'Linear Magnification'
    ax0.set_title(x0Title,color='m',fontsize='16')
    ax0.set_xlim([-1,steps+1])
    ax0.set_ylim([tMin,tMax])
    ax0.set_xlabel('Step n',color='m',fontsize='14')
    ax0.set_ylabel('t',color='m',fontsize='14')
    ax0.grid(True)
    
    ax1 = plt.subplot(gs[1])
    plt.plot(step,tPar,'-x',color='r')
    x1Title = 'Parabolic Magnification'
    ax1.set_title(x1Title,color='m',fontsize='16')
    ax1.set_xlim([-1,steps+1])
    ax1.set_ylim([tMin,tMax])
    ax1.set_xlabel('Step n',color='m',fontsize='14')
    ax1.set_ylabel('t',color='m',fontsize='14')
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[2])
    plt.plot(step,tSSF,'-x',color='r')
    x2Title = 'Smooth Sign-function Magnification'
    ax2.set_title(x2Title,color='m',fontsize='16')
    ax2.set_xlim([-1,steps+1])
    ax2.set_ylim([tMin,tMax])
    ax2.set_xlabel('Step n',color='m',fontsize='14')
    ax2.set_ylabel('t',color='m',fontsize='14')
    ax2.grid(True)

    fig.tight_layout()
    plt.show()

    selection = int(raw_input("\nYour selection of the law magnification \
    \n(1 - linear, 2 - parabolic, 3 - smooth sign-function; -1 - exit): "))
    return selection
        
#   
# Main method 'simulation'
#
def simulation():
#
# Main predefined parameters of the nonlinear insertion:
    insrtn_l0 = 1.8      # total length, m
    insrtn_mu0 = .3      # phase, rad (/2pi)
    insrtn_c = .01       # aperture factor,  m^(1/2)
    num_lens = 20        # number of lens inside insertion
#
# Interactive input of the parameters for simulation:
#
    particlesInBunch = int(raw_input('\nTotal number of particles (= -1 to interrupt simulation):')) 
    if particlesInBunch == -1:
        return

    totalTurns = int(raw_input('\nTotal number if turns (= -1 to interrupt simulation):')) 
    if totalTurns == -1:
        return

    updateAfterTurns = int(raw_input( \
    '\nPeriodicity (in turns) to update the parameters and distribution plots \n(nonlinear structure; = -1 to interrupt simulation)'))
    if updateAfterTurns == -1:
        return
    stepsInMgnfctn = int(totalTurns/updateAfterTurns)+0
    print "steps for magnification: ",stepsInMgnfctn

    updateOutputFlag = int(raw_input('\nupdateOutputFlag (0 - no, 1 - yes, -1 - to interrupt simulation):'))
    if updateOutputFlag == -1:
        return

    magnificationType = int(raw_input( \
                     '\nMagnification type \n(1 - relative, 2 - absolute, 0 - to interrupt simulation):'))
    if magnificationType == 0:
        return
    else:
        if magnificationType == 1:
            mgnfctnFctr = float(raw_input( \
    "\nFactor of relative magnification (RM) of the strength 't' of all (!) nonlinear lenses \n (RM = t_f/t_i; -1. - to interrupt simulation):"))
            if mgnfctnFctr == -1.:
                return
            else: 
                t_i = 1.
                t_f = mgnfctnFctr
        else:
            print "\nInformation for help (20 nonlinear lenses inside of the insertion): \n"
            t_on_knll_function(insrtn_l0,insrtn_mu0,insrtn_c,20)
            t_i = float(raw_input( \
    "\nInitial value 't_i' of the strength of the central (!) nonlinear lens \n (-1.- to interrupt simulation):"))
            if t_i == -1.:
                return
            t_f = float(raw_input( \
    "\nFinal value 't_f' of the strength of nonlinear lens \n (-1.- to interrupt simulation):"))
            if t_f == -1.:
                return
    print ""
    law = lawsMagnification(t_i,t_f,stepsInMgnfctn)
    print 'Your selection of law magnification: ', law
    if law == -1:
        return

# Input data for simulation:
    print "\n################################################################\n###"
    print "###            Parameters for simulation:\n###"
    print "###     Particles in the bunch = ",particlesInBunch
    print "###     Total number of turns = ",totalTurns
    print "###     Periodicity (in turns) to update the parameters = ",updateAfterTurns
    print "###     magnificationType = ",magnificationType
    if magnificationType == 1:
        print "###     Factor of relative magnification (RM) = ",mgnfctnFctr
    if magnificationType == 2:
        print "###     For absolute magnification (AM) initial value t_i = ",t_i
        print "###     For absolute magnification (AM) final value t_f = ",t_f
    laws = ['linear', 'parabolic', 'smooth sign-function']
    print "###     Law of magnification: ",laws[law-1]
    print "###     Steps in magnification: ",stepsInMgnfctn
    print "###\n###        Predefined parameters for nonlinear insertion:\n###"
    print "###     Length = ",insrtn_l0," m"
    print "###     Phase = ",insrtn_mu0," rad (/2pi)"
    print "###     Aperture factor = ",insrtn_c," m^(1/2)"
    print "###     Number of lens inside insertion = ",num_lens
    print "###\n################################################################"
#
# For relative type of maginfication (magnificationType = 1):
#
    if magnificationType == 1:
#
# t_i = 1. and t_f is total factor of magnification.
# So, 1D-array 'strengthLens[0:stepsInMgnfctn]' describes current value of the 
# strength (knll) of lens for current step n; Then 1D-array 'magnifications[0:stepsInMgnfctn]'
# describe magnification factor to pass from old_knll_value = knll[n-1] to 
# new_knll_value = knll[n] on step n:
#    new_knll_value = magnifications[n]*old_knll_value .
# Factor 'magnifications' is the same for all lens of nonlinear insertion!
#
        strengthLens = np.zeros(stepsInMgnfctn)
        magnifications = np.zeros(stepsInMgnfctn)
        totalMgnfcn = 1.
#
# For absolute magnification  (magnificationType = 2):
#
    if magnificationType == 2:
#
# parameters t_i and t_f characterize only central lens of nonlinear insertion. 
# So, the strength of 't' for all rest lenses must be recalculate in corresponding  
# distribution of beta-function inside the insertion by using method 'generate_lens'. 
# So, 1D-array 'strengthLens[0:stepsInMgnfctn]' describes value of the strength 
# of central lens only for current step n. 
# 
        strengthLens = np.zeros(stepsInMgnfctn)
    for n in range(stepsInMgnfctn):
        if law == 1:
# 1) Linear: for step number n
#           t(n) = t_i + (t_f-t_i)*n/(N-1) for n = 0,1,...,N-1 .
            strengthLens[n] = t_i+n*(t_f-t_i)/(stepsInMgnfctn-1)
        elif law == 2:
# 2) Parabolic: for step number n
#           t(n) = t_i + (t_f-t_i)*n^2/(N-1)^2 for n = 0,1,...,N-1 .
            strengthLens[n] = t_i+n**2*(t_f-t_i)/(stepsInMgnfctn-1)**2
        elif law == 3:
# 3) Smooth sign-function: for step number n
#           t(n) = .5*(t_i+t_f) + .5*(t_f-t_i)*tanh(x(n)), where
#           x(n) = (6*n-3*(N-1))/(N-1) for n=0,1,...,N-1 .
# In this approach x(0) = -3., x(N-1) = 3.; so, tanh(3.) = - tanh(-3.) = .9951
            x = (6.*n-3.*(stepsInMgnfctn-1))/(stepsInMgnfctn-1)
            strengthLens[n] = .5*(t_i+t_f)+.5*(t_f-t_i)*np.tanh(x)
        if magnificationType == 1:
            if n == 0:
                print "\nRelative magnification:"
                magnifications[n] = strengthLens[n]
            else:
                magnifications[n] = strengthLens[n]/strengthLens[n-1]
            print "    magnifications[{}] = {}".format(n,magnifications[n])
            totalMgnfcn *= magnifications[n]
            if n == stepsInMgnfctn-1:
                print "Total relative magnification (RM) will be = ",totalMgnfcn
        if magnificationType == 2:
            if n == 0:
                print \
            "\nStrengths 't' and corresponding values 'knll' of cenrtal lens for absolute magnification:"
# Calculate value 'knll', which correspond to current value of strngth 't' = strengthLens[n]:
            f0Crrnt = insrtn_l0/4.0*(1.0+1.0/np.tan(np.pi*insrtn_mu0)**2)
            first_lens_center = .5*(insrtn_l0/num_lens)
            last_lens_center = insrtn_l0 - first_lens_center
# Coordinates of the center of the nonlinear lenses in the nonlinear inserion (m):
            s_vals = np.linspace(first_lens_center,last_lens_center,num_lens) 
#            print "s_val =",s_vals        
# Coordinate of the center of the nonlinear lens in the middle of nonlinear inserion (m):
            s_center = s_vals[(num_lens+1)/2]
# Structural beta-function of the nonlinear magnet (m):
            beta_center = insrtn_l0*(1.-s_center*(insrtn_l0-s_center)/insrtn_l0/f0Crrnt)/ \
                          np.sqrt(1.0-(1.0-insrtn_l0/2.0/f0Crrnt)**2)
            cnll_center = insrtn_c*np.sqrt(beta_center)
#            print "s_center = ",s_center," m, beta_center = ",beta_center, \
#                  " m, cnll_center = ",cnll_center," m"
            knll_center = insrtn_l0/num_lens*strengthLens[n]*(cnll_center/beta_center)**2
            print "   t[{}]] = {} ==> knll = {} m".format(n,strengthLens[n],knll_center)
#
# Simulated lattice:
#
    fileIOTA = ".../ioptics/ioptics/lattices/Iota8-2/lattice_1IO_nll_center.madx"
    print "\nIOTA Nonlinear lattice: {} \n".format(fileIOTA)
    lattice = synergia.lattice.MadX_reader().get_lattice("iota", \
    "../ioptics/ioptics/lattices/Iota8-2/lattice_1IO_nll_center.madx")
# To recognize attributes of 'lattice':
#    printAttributes(lattice,'lattice','synergia.lattice.MadX_reader().get_lattice')


#    fileIOTA = ".../ioptics/ioptics/lattices/Iota8-4/lattice_8-4_1IO_nll_forTest.madx"
#    print "\nIOTA Nonlinear lattice: {} \n".format(fileIOTA)
#    lattice = synergia.lattice.MadX_reader().get_lattice("iota", \
#    "../ioptics/ioptics/lattices/Iota8-4/lattice_8-4_1IO_nll_forTest.madx")

# For checking only:
#    k = 0
#    for elem in lattice.get_elements():
#        if k == 0:
#            printAttributes(elem,'elem','lattice.get_elements')
#       k += 1
#        if elem.get_type() == 'nllens':
#            elem.set_string_attribute("extractor_type", "chef_propagate")
#        else:
#            elem.set_string_attribute("extractor_type", "chef_map")
#        print "elem ({}): name = {}, type = {}, stringAttrbt ={}". \
#              format(k,elem.get_name(),elem.get_type(),elem.get_string_attribute("extractor_type"))
    
    knllLenses = []
    nameLenses = []
    placeLenses = []
    numberLenses = 0
    for element in lattice.get_elements():
        if element.get_type() == 'nllens':
            knllLenses.append(float(element.get_double_attribute("knll")))
            nameLenses.append(element.get_name())
            placeLenses.append(int(numberLenses))
            numberLenses += 1 
    num_lens = numberLenses        # number of lens inside insertion
#    print "placeLenses: ",placeLenses
#    print "nameLenses: ",nameLenses        
#    print "knllLenses: ",knllLenses
#    print "Number of lenses: ",numberLenses
# Name of lens with maximal strength to use in output for checking of process of magnification    
    controlName = nameLenses[np.argmax(knllLenses)]
#    print "controlName: ",controlName

    startSequenceLenses = 1              # First lens has minimal knll
    if knllLenses[0] > knllLenses[1]:
        startSequenceLenses = 2          # First lens has maximal knll
#    print "startSequenceLenses = ",startSequenceLenses    


# Original version:
#     lattice_simulator = synergia.simulation.Lattice_simulator(lattice, 2)
# Bunch:
#     bunch = synergia.optics.generate_matched_bunch_transverse(lattice_simulator, 1e-6, \
#                                                          1e-6, 1e-3, 1e-4, 1e9, 10000, seed=1234)

# YuE version:
    stepperCrrnt = synergia.simulation.Independent_stepper_elements(lattice,2,3)
    lattice_simulator_Crrnt = stepperCrrnt.get_lattice_simulator()
# Bunch:
    bunch_origin = synergia.optics.generate_matched_bunch_transverse( \
                   lattice_simulator_Crrnt, 1e-6, 1e-6, 1e-3, 1e-4, 1e9, particlesInBunch, seed=1234)
# For checking:
# To recognize attributes of 'bunch_origin':
#     printAttributes(bunch_origin,'bunch_origin','synergia.optics.generate_matched_bunch_transverse')
#     particlesTmp = bunch_origin.get_local_particles()
# To recognize attributes of 'particlesTmp':
#     printAttributes(particlesTmp,'particlesTmp','bunch_origin.get_local_particles')
# 'particlesCrrnt' is 2D array: (numberoFParticle,(x,x',y,y',s,dE,ID));
#     particlesCrrnt = particlesTmp.real
#     print "                 particlesCrrnt:"
#     for prtcl in range(5):
#         print "x (m) for particle {}: {}".format(prtcl,particlesCrrnt[prtcl,0])
#         print "y (m) for particle {}: {}".format(prtcl,particlesCrrnt[prtcl,2])
#         print "s (m) for particle {}: {}".format(prtcl,particlesCrrnt[prtcl,4])
# End of checking


#-------------------------------------------------
# For checking only:
#
# 1) Attributes:
#     printAttributes(bunch,'bunch','synergia.optics.generate_matched_bunch_transverse')
# 2) Distributions X-Y, X-X', Y-Y' using method 'pltbunch.plot_bunch':
    loclTitle = "\nThese distributions were constructed using "
    loclTitle += "'synergia.optics.generated_matched_bunch_transverse' method:\n"
    print loclTitle
    pltbunch.plot_bunch(bunch_origin)     
# 3) Distributions X-Y, X-X', Y-Y' using method 'plotcoordDistr':
    bunchParticles = bunch_origin.get_local_particles()
# To recognize attributes of 'bunchParticles':
#     printAttributes(bunchParticles,'bunchParticles', 'bunch.get_local_particles()')
    plotcoordDistr(bunchParticles)
#--------------------------------------------------

# Steppers (YuE: both case 'splitoperator' and 'independent' work properly!):
#     stepper = 'splitoperator'
    stepper = 'independent'
    if stepper == "splitoperator":
# Use the Split operator stepper with a dummy collective operator (with evenly-spaced steps)
        no_op = synergia.simulation.Dummy_collective_operator("stub")
        stepper = synergia.simulation.Split_operator_stepper(
                            lattice_simulator_Crrnt, no_op, 4)
    elif stepper == "independent":
# Use the Independent particle stepper (by element)
        stepper = synergia.simulation.Independent_stepper_elements(
                            lattice_simulator_Crrnt, 4)
    else:
        sys.stderr.write("fodo.py: stepper must be either 'independent' or 'splitoperator'\n")
        exit(1)

# Bunch simulator:
    bunch_simulator = synergia.simulation.Bunch_simulator(bunch_origin)


# This diagnostics does not use!
# Diagnostics:
#    diagnostic_flag = 'None'
#    for part in range(0, 0):
#        bunch_simulator.add_per_step(synergia.bunch.Diagnostics_track("step_track_%02d.h5" % part,
#                                                                   part))
#    if diagnostic_flag == 'step_full2':
#        bunch_simulator.add_per_step(synergia.bunch.Diagnostics_full2("step_full2.h5"))
#    if diagnostic_flag == 'step_particles':
#        bunch_simulator.add_per_step(synergia.bunch.Diagnostics_particles("step_particles.h5"))
#    for part in range(0, 0):
#        bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_track("turn_track_%02d.h5" % part,
#                                                                   part))
#    if diagnostic_flag == 'turn_full2':
#    bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_full2("turn_full2.h5"))
#    if diagnostic_flag == 'turn_particles':
#        bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_particles("turn_particles.h5"))
    

#---------------------------
# Propagate
#---------------------------
# Ramp action is instantiated and passed to the propagator instance during the propagate method

    print "\n-------------------\n"
    print "           Nonlinear parameters will be CHANGED after each {} turns".format(updateAfterTurns)
    print "\n-------------------\n"

# Кe-setting the original 'bunch_origin' object, because it was changed (for some unknown reason) 
# while pulling a 'bunch' object through a fixed number of turns in a linear structure
    bunch_origin = synergia.optics.generate_matched_bunch_transverse(lattice_simulator_Crrnt, 1e-6, \
                                                          1e-6, 1e-3, 1e-4, 1e9, particlesInBunch, seed=1234)
# For checking (to verify that particles from "old" and "new" 'bunch_origin' objects are the same):
#     particlesOrg4 = bunch_origin.get_local_particles()
# To recognize attributes of 'particlesOrg2':
#     printAttributes(particlesOrg4,'particlesOrg4','bunch_origin.get_local_particles')
# End of checking (result: particles in both "old" and "new" objects are the same!)

    bunch = bunch_origin
# For checking:
#     particlesTmp2 = bunch.get_local_particles()
# To recognize attributes of 'particlesTmp2':
#     printAttributes(particlesTmp2,'particlesTmp2','bunch.get_local_particles')
#     particlesCrrnt2 = particlesTmp2.real
#     print "                 particlesCrrnt (again for nonlinear):"
#     for prtcl in range(5):
#         print "x (m) for particle {}: {}".format(prtcl,particlesCrrnt2[prtcl,0])
#         print "y (m) for particle {}: {}".format(prtcl,particlesCrrnt2[prtcl,2])
# End of checking

    bunch_simulator = synergia.simulation.Bunch_simulator(bunch)

    propagator = synergia.simulation.Propagator(stepper)
#     propagator.set_checkpoint_period(0)
#     propagator.set_checkpoint_with_xml(True)

# tracksNonLinear is 3D array: (totalTurns,bunchParticles,(x,y)) 
    tracksNonLinear = np.zeros((totalTurns,particlesInBunch,2))

    nUpdate = 1
    stepOfMgnfcn = 1
    totalTimeCPU = 0.
    for turnCrrnt in range(totalTurns):
        timeStart = os.times()
#        
# Without of initialization:
#        propagatorCrrnt = propagator.propagate(bunch_simulator, 1, 1, 0)
# To recognize attributes of 'propagatorCrrnt':
#        printAttributes(propagatorCrrnt,'propagatorCrrnt', \
#                    'propagator.propagate(bunch_simulator, 1, 1, 0)')
        if turnCrrnt == 0:
#------------------
# Initialization of the lattice before first turn:
#
            if magnificationType == 1:
                ramp_actions = Ramp_actions(magnificationType,0,strengthLens, \
                                            updateOutputFlag,controlName)   
            if magnificationType == 2:
                dataInsertion = \
                NonlinearInsertion(insrtn_l0, insrtn_mu0, strengthLens[stepOfMgnfcn], \
                                   insrtn_c, num_lens).generate_lens(startSequenceLenses)
                knll_lens = dataInsertion.knll
                ramp_actions = Ramp_actions(magnificationType,0,knll_lens, \
                                            updateOutputFlag,controlName)   
            propagatorCrrnt = propagator.propagate(bunch_simulator, ramp_actions, 1, 1, 0)
#
# End of initialization of the lattice before first turn
#------------------

# bunchParticles is 2D array: (numberParrticles,(x,x',y,y',s,dE,ID))
        bunchParticles = bunch.get_local_particles()
# coordsTracks is 2D array: (bunchParticles,(x,y)) 
        coordsTracks = tracksCoords(bunchParticles)
        numbPartcls = bunchParticles.shape[0]
        for prtcl in range(numbPartcls):
            for k in range(2):
                tracksNonLinear[turnCrrnt,prtcl,k] = coordsTracks[prtcl,k]
#            if prtcl < 3:
#                print "tracksNonLinear (turn {}) for particle {}: x = {} mm, y = {} mm". \
#                format(turnCrrnt,prtcl,tracksNonLinear[turnCrrnt,prtcl,0], \
#                       tracksNonLinear[turnCrrnt,prtcl,1])
        turnNumber = turnCrrnt+1
        timeEnd = os.times()
        timeOfTurn = float(timeEnd[0] - timeStart[0])              # CPU time in seconds
        totalTimeCPU += timeOfTurn
        print ('Turn %3d is completed (CPU time = %6.3f seconds)' % (turnNumber, timeOfTurn))
        if turnCrrnt == totalTurns-1:
            break
        sys.stdout.flush()
        if nUpdate == updateAfterTurns:
            timeStart = os.times()
            print "\n"
            plotcoordDistr(bunchParticles)
#== #
#== # Possibility for future to redefine parameters "in-fly" of simulation:
#== #
#==             if updateInsideSmlnFlag == 1:
#==                 print "Old multiplyier for knl = {}".format(knlMultiplier)
#== # Multiplier 'knlMultiplier' is the same for all nonlinear lenses:   
#==                 knlMultiplier = float(raw_input('\nNew multiplyier for knl:'))
#==                 print "Old multiplyier for cnll = {}".format(cnllMultiplier)
#== # IF NEEDED: multiplier 'cnllMultiplier' is the same for all nonlinear lenses:   
#==                 cnllMultiplier = float(raw_input('\nNew multiplyier for cnll:'))

            if magnificationType == 1:
#
# Relative magnification - for current step 'stepOfMgnfcn' > 1 multipliers for all lenses are the same 
# and equal to ratio strengthLens[stepOfMgnfcn]/strengthLens[stepOfMgnfcn-1] (except the first step): 
#
                if stepOfMgnfcn == 0:
                    knlMultiplier = strengthLens[stepOfMgnfcn] 
                else:
                    knlMultiplier = strengthLens[stepOfMgnfcn]/strengthLens[stepOfMgnfcn-1]
#                print "Step for relative magnification ",stepOfMgnfcn,": knlMultiplier = ",knlMultiplier
#
# REMINDER regarding of 'Ramp_actions' class!
#
# Args are: 
# magnificationType - type of magnification (1 - relative, 2 - absolute),
# stepOfMgnfcn      - current step of magnification,
# strengthLens      - set of strengthes 't' of central lens of the nonlinear insertion for all steps of
#                     magnification (relative magnification) or set of strengthes 't' of all lenses for 
#                     current step (absolute magnification),
# updateOutputFlag  - flag to output the strength of one of nonlinear lens after it's magnification 
#                     for current step,
# controlName       - name of lens with maximal strength to use in output for checking of process of
#                     magnification.    
#
                ramp_actions = Ramp_actions(magnificationType,stepOfMgnfcn,strengthLens, \
                                            updateOutputFlag,controlName)   
            if magnificationType == 2:
#
# Absolute magnification - for current step stepOfMgnfcn the strength 't' for central nonlinear lens 
# equals strengthLens[stepOfMgnfcn] 
#

#
# REMINDER regarding of 'NonlinearInsertion' class!
#
# Input args:
#   length:     the length of the nonlinear insertion (float, m);
#   phase:      the phase advance modulo 2pi through the nonlinear insertion;
#   t:          the strength parameter for center of the insertion (float, dimensionless, defaults to 0.1);
#   c:          the aperture parameter for center of the insertion 
#               (float, m^1/2, is defined by poles in the x-axis, defaults to 0.01);
#   num_lens:   the number of nonlinear lenses as an segments of the insertion (int, defaults to 20).
#
# Output attributes are the same as input one.
#
 
#
# REMINDER regarding of 'generate_lens' method!
#
# Input arg:
# startSequenceLenses - flag of the distribution 'knll' parameter of the lenses 
# (1 - nonlinear insertion in *.madx description of the IOTA ring started from lens with minimal strength,
#  2 - nonlinear insertion in *.madx description of the IOTA ring started from lens with maximal strength).
#
# Output attributes:
#
#   same as output of 'NonlinearInsertion' class and as well:
#   s_vals (ndArray) - coordinates of the center of each nonlinear lens (float ndArray, m);
#   knll (ndArray)   -   "strength" of each nonlinear lens (float ndArray, m);
#   cnll (ndArray)   -   aperture parameters for each nonlinear lens (float ndArray, m^1/2).
#   
                dataInsertion = \
                NonlinearInsertion(insrtn_l0, insrtn_mu0, strengthLens[stepOfMgnfcn], insrtn_c, num_lens). \
                generate_lens(startSequenceLenses)
                coords_lens = dataInsertion.s_vals
                knll_lens = dataInsertion.knll
                cnll_lens = dataInsertion.cnll
#                if stepOfMgnfcn > 0:
#                    print "Step for absolute magnification ",stepOfMgnfcn, \
#                          ": for central lens current 't' = ",strengthLens[stepOfMgnfcn]

#                print "coords_lens = ",coords_lens
#                print "knll_lens = ",knll_lens
#                print "cnll_lens = ",cnll_lens

# title_k for knll-plot, title_c - for cnll-plot:
                title_k = "Nonlinear Insertion: L={:.1f}m, phase= {:.2f}, t={:.4f}, c={:.2f}m^1/2". \
                format(insrtn_l0, insrtn_mu0, strengthLens[stepOfMgnfcn], insrtn_c)
#                print "title_k = ",title_k
                title_c = title_k
#                print "title_c = ",title_c
                plotParamLens(coords_lens,knll_lens,cnll_lens,title_k,title_c)
#                print "Step ",stepOfMgnfcn,": knll = ",knll_lens
                ramp_actions = Ramp_actions(magnificationType,stepOfMgnfcn,knll_lens, \
                                            updateOutputFlag,controlName)   
            stepOfMgnfcn += 1

            nUpdate = 0
            print "\n              After {} turns:\n".format(turnNumber)
            propagatorCrrnt = propagator.propagate(bunch_simulator, ramp_actions, 1, 1, 0)
            timeEnd = os.times()
            timeUpdateAndPlot = float(timeEnd[0] - timeStart[0])              # CPU time in seconds
            totalTimeCPU += timeUpdateAndPlot
            print ('\nUpdate and plotting are completed (CPU time = %6.3f seconds)\n' % timeUpdateAndPlot)
        nUpdate += 1
#     for prtcl in range(5):
#         print "x (mm) for particle {}: {}".format(prtcl,tracksNonLinear[:,prtcl,0])
#         print "y (mm) for particle {}: {}".format(prtcl,tracksNonLinear[:,prtcl,1])
    print "\n\n                        Final results: \n\n"
    plotcoordDistr(bunchParticles)
    plotTracks(tracksNonLinear,5)
    print ('\nFor %5d turns CPU time = %6.3f seconds\n' % (totalTurns, totalTimeCPU))
    return
#
# End of main method 'simulation'
#

#========================================================

fileIOTA = ".../ioptics/ioptics/lattices/Iota8-2/lattice_1IO_nll_center.madx"
# fileIOTA = ".../ioptics/ioptics/lattices/Iota8-4/lattice_8-4_1IO_nll_forTest.madx"
print "\nIOTA Nonlinear lattice: {} \n".format(fileIOTA)
lattice = synergia.lattice.MadX_reader().get_lattice("iota", \
          "../ioptics/ioptics/lattices/Iota8-2/lattice_1IO_nll_center.madx")

# --------- Games -----------------------------
# indices = np.argsort(knllLenses) 
# print "indices = ",indices 
# for n in range(nLenses+1):
#     print n,") name after sorting is ",nameLenses[indices[n]]
# for n in range(nLenses+1):
#     print n,") knll after sorting is ",knllLenses[indices[n]]
# for n in range(nLenses+1):
#     print n,") place after sorting is ",placeLenses[indices[n]]
# ----------- End of games --------------------

stepperCrrnt = synergia.simulation.Independent_stepper_elements(lattice,2,3)
lattice_simulator_Crrnt = stepperCrrnt.get_lattice_simulator()
# To recognize attributes of 'bunchParticles':
# printAttributes(lattice_simulator_Crrnt,'lattice_simulator_Crrnt','stepperCrrnt.get_lattice_simulator()')
# slicesHelp = lattice_simulator_Crrnt.get_slices()
# To recognize attributes of 'slicesHelp':
# printAttributes(slicesHelp,'slicesHelp','lattice_simulator_Crrnt.get_slices()')

# Bunch:
bunch_origin = synergia.optics.generate_matched_bunch_transverse(lattice_simulator_Crrnt, 1e-6, \
                                                          1e-6, 1e-3, 1e-4, 1e9, 1000, seed=1234)
#
# To compare two methods for drawing of the particles distributions:
#
loclTitle = "\nThese distributions were constructed using \
'synergia.optics.generated_matched_bunch_transverse' method"
loclTitle += "\nand plotted using two methods - 'pltbunch.plot_bunch' from the code synergia" 
loclTitle += "\nand 'plotcoordDistr' from this script (to verify method 'plotcoordDistr'):"
print loclTitle
pltbunch.plot_bunch(bunch_origin)     
# Distributions X-Y, X-X', Y-Y' using method 'plotcoordDistr':
bunchParticles = bunch_origin.get_local_particles()
# To recognize attributes of 'bunchParticles':
#     printAttributes(bunchParticles,'bunchParticles', 'bunch.get_local_particles()')
plotcoordDistr(bunchParticles)

selection = 'loop'
while selection == 'loop':
    simulation() 
    selection = raw_input("\nTo continue the simulation ('yes' or 'no'):")
    print'Your selection is ',selection
    if selection == 'yes':
        selection = 'loop'
#    if selection == 'no':
#        exit(0)

