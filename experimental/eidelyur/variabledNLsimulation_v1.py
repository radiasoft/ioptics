# 
# This script uses parts of scripts 'turn_aend_action_lattice_edit.py' (Chris) 
# and 'variabledNLsimulation_v0.py' (Yury)
#
#    Started at June 13, 2019
#
import synergia
import os, sys
import inspect
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
# particles is 2D array: (numberOfParticles,(x,x',y,y',s(?),dp(?),ID(?)));
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
    ax0.set_title(x0Title,fontsize='16')
    ax0.set_xlim([-xmax,xmax])
    ax0.set_ylim([-ymax,ymax])
    ax0.set_xlabel('X, mm')
    ax0.set_ylabel('Y, mm')
    ax0.grid(True)
    
    ax1 = plt.subplot(gs[1])
    plt.plot(newCoordinates[0,:],newCoordinates[1,:],'.',color='b')
    x1Title = "X,mm: <> = {:.3f} +- {:.3f}\nX\',mrad: <> = {:.3f} +- {:.3f}".format(meanX,stdX,meanPX,stdPX)
    ax1.set_title(x1Title,fontsize='16')
    ax1.set_xlim([-xmax,xmax])
    ax1.set_ylim([-xpmax,xpmax])
    ax1.set_xlabel('X, mm')
    ax1.set_ylabel('X\', mrad')
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[2])
    plt.plot(newCoordinates[2,:],newCoordinates[3,:],'.',color='r')
    x2Title = "Y,mm: <> = {:.3f} +- {:.3f}\nY\',mrad: <> = {:.3f} +- {:.3f}".format(meanY,stdY,meanPY,stdPY)
    ax2.set_title(x2Title,fontsize='16')
    ax2.set_xlim([-ymax,ymax])
    ax2.set_ylim([-ypmax,ypmax])
    ax2.set_xlabel('Y, mm')
    ax2.set_ylabel('Y\', mrad')
    ax2.grid(True)
    
#    fig.canvas.set_window_title('Synergia Phase Space Distribution')
    fig.tight_layout()
    plt.show()
    return

def plotTracks(tracksCoords,numberTracks):
#    
# Plot'numberTracks' tracks from 'tracksCoords'
#
# tracksCoords is 3D array: (totalTurns,particles,(x,y,s))
#
    print "numberTracks = ",numberTracks
    trackColor = ['r','b','k','m','g']
    numbPoints = tracksCoords.shape[0]
#    print "numbPoints = ",numbPoints
    xmax = 1.15*np.max(np.max(abs(tracksCoords[:,0:numberTracks,0])))
    ymax = 1.15*np.max(np.max(abs(tracksCoords[:,0:numberTracks,1])))
    smax = 1.15*np.max(np.max(tracksCoords[:,0:numberTracks,2]))

    turn = np.arange(0,numbPoints)
# Another way - use gridspec
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
    
    ax0 = plt.subplot(gs[0])
    for prtcl in range(numberTracks):
        plt.plot(turn,tracksCoords[0:numbPoints,prtcl,0],'.-',color=trackColor[prtcl])
#    x0Title = "X,mm: <> = {:.3f} +- {:.3f}\nY,mm: <> = {:.3f} +- {:.3f}".format(meanX,stdX,meanY,stdY)
#    ax0.set_title(x0Title,fontsize='16')
    ax0.set_ylim([-xmax,xmax])
    ax0.set_xlabel('Turn')
    ax0.set_ylabel('X, mm')
    ax0.grid(True)

    ax1 = plt.subplot(gs[1])
    for prtcl in range(numberTracks):
        plt.plot(turn,tracksCoords[0:numbPoints,prtcl,1],'.-',color=trackColor[prtcl])
#    x0Title = "X,mm: <> = {:.3f} +- {:.3f}\nY,mm: <> = {:.3f} +- {:.3f}".format(meanX,stdX,meanY,stdY)
#    ax0.set_title(x0Title,fontsize='16')
    ax1.set_ylim([-ymax,ymax])
    ax1.set_xlabel('Turn')
    ax1.set_ylabel('Y, mm')
    ax1.grid(True)
       
    ax2 = plt.subplot(gs[2])
    for prtcl in range(numberTracks):
        plt.plot(turn,tracksCoords[0:numbPoints,prtcl,2],'.-',color=trackColor[prtcl])
##    x2Title = "Y,mm: <> = {:.3f} +- {:.3f}\nY\',mrad: <> = {:.3f} +- {:.3f}".format(meanY,stdY,meanPY,stdPY)
##    ax2.set_title(x2Title,fontsize='16')
    ax2.set_ylim([-smax,smax])
    ax2.set_xlabel('Turn')
    ax2.set_ylabel('s, m')
    ax2.grid(True)
   
    fig.canvas.set_window_title('Synergia Phase Space Distribution')
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
    trackCoordinates = np.zeros((numbPartcls,3))
    for prtcl in range(numbPartcls):
        trackCoordinates[prtcl,0] = 1.e3*particles[prtcl,0]       # x, mm
        trackCoordinates[prtcl,1] = 1.e3*particles[prtcl,2]       # y, mm
        trackCoordinates[prtcl,2] = particles[prtcl,4]            # s, m
        if prtcl < 3:
            print "Particle {}: x = {} mm, y = {} mm, s = {} m". \
            format(prtcl,trackCoordinates[prtcl,0],trackCoordinates[prtcl,1],trackCoordinates[prtcl,2])
    return trackCoordinates 
    
# Pickle helper is not necessary but is retained for this example
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
        
# Derived class to ramp quadrupoles

# Will be passed into propagator and called at each appropriate 'interval' (step, turn, action, etc...)

class Ramp_actions(synergia.simulation.Propagate_actions, Pickle_helper):
# The arguments to __init__ are what the Ramp_actions instance is initialized with
    def __init__(self, multiplier1,multiplier2, outputFlag):
        selfObject = synergia.simulation.Propagate_actions.__init__(self)
# To recognize attributes of 'selfObject':
#        printAttributes(selfObject,'selfObject','synergia.simulation.Propagate_actions.__init__(self)')

# Pickling the arguments to the initializer allows the module to resume
# after checkpointing. They should be in the same order as the arguments to __init__.
        Pickle_helper.__init__(self, multiplier1, multiplier2, outputFlag)
        self.multiplier1 = multiplier1
        self.multiplier2 = multiplier2
        self.outputFlag = outputFlag
    def turn_end_action(self, stepper, bunch, turn_num):
#---------------------------
# For checking:
#        testObject = stepper.get_lattice_simulator().get_lattice()
# To recognize attributes of 'testObject':
#        printAttributes(testObject,'testObject','stepper.get_lattice_simulator().get_lattice()')
#        print "testName = '{}'".format(testObject.get_name())
#---------------------------
# Output title for checking of variables update:   
        if self.outputFlag == 1:
            print "Modifying lattice:"
        for element in stepper.get_lattice_simulator().get_lattice().get_elements():
            if element.get_type() == "nllens":
                old_knll = element.get_double_attribute("knll")
                element.set_double_attribute("knll", self.multiplier1*old_knll)
                old_cnll = element.get_double_attribute("cnll")
                element.set_double_attribute("cnll", self.multiplier2*old_cnll)
# Output for hecking of variables update:  
                if ((self.outputFlag == 1) and (element.get_name() == "n.11")):
                    print element.get_name(),":  knll=",old_knll,"-->", \
                       self.multiplier1*old_knll, ";  cnll=",old_cnll,"-->",self.multiplier2*old_cnll
        stepper.get_lattice_simulator().update()
    
# Routine Simulation Setup

# Lattice:


fileIOTA = ".../ioptics/ioptics/lattices/Iota8-2/lattice_1IO_nll_center.madx"
print "\nIOTA Nonlinear lattice: {} \n".format(fileIOTA)
lattice = synergia.lattice.MadX_reader().get_lattice("iota", \
"../ioptics/ioptics/lattices/Iota8-2/lattice_1IO_nll_center.madx")

'''

fileIOTA = ".../ioptics/ioptics/lattices/Iota8-4/lattice_8-4_1IO_nll_forTest.madx"
print "\nIOTA Nonlinear lattice: {} \n".format(fileIOTA)
lattice = synergia.lattice.MadX_reader().get_lattice("iota", \
"../ioptics/ioptics/lattices/Iota8-4/lattice_8-4_1IO_nll_forTest.madx")
'''

#----------------------------------
# To recognize attributes of 'lattice':
# printAttributes(lattice,'lattice','synergia.lattice.MadX_reader().get_lattice')

# madxAttributes = synergia.lattice.MadX_reader()
# To recognize attributes of 'doubleAttributes':
# printAttributes(madxAttributes,'madxAttributes','synergia.lattice.MadX_reader()')

# doubleVariable = madxAttributes.get_double_variable()
# To recognize attributes of 'doubleVariable':
# printAttributes(doubleVariable,'doubleVariable','madxAttributes.get_double_variablte()')
# print "doubleVariable = ", doubleVariable
#---------------------------------
'''
# For checking only:
k = 0
for elem in lattice.get_elements():
    if k == 0:
        printAttributes(elem,'elem','lattice.get_elements')
    k += 1
    if elem.get_type() == 'nllens':
        elem.set_string_attribute("extractor_type", "chef_propagate")
    else:
        elem.set_string_attribute("extractor_type", "chef_map")
    print "elem ({}): name = {}, type = {}, stringAttrbt ={}". \
          format(k,elem.get_name(),elem.get_type(),elem.get_string_attribute("extractor_type"))
'''    

# Original version:
# attice_simulator = synergia.simulation.Lattice_simulator(lattice, 2)
# Bunch:
# bunch = synergia.optics.generate_matched_bunch_transverse(lattice_simulator, 1e-6, \
#                                                          1e-6, 1e-3, 1e-4, 1e9, 10000, seed=1234)

# YuE version:
stepperCrrnt = synergia.simulation.Independent_stepper_elements(lattice,2,3)
lattice_simulator_Crrnt = stepperCrrnt.get_lattice_simulator()
# Bunch:
bunchInParticles = 100
bunch_origin = synergia.optics.generate_matched_bunch_transverse(lattice_simulator_Crrnt, 1e-6, \
                                                          1e-6, 1e-3, 1e-4, 1e9, bunchInParticles, seed=1234)
# For checking:
# To recognize attributes of 'bunch_origin':
# printAttributes(bunch_origin,'bunch_origin','synergia.optics.generate_matched_bunch_transverse')
particlesTmp = bunch_origin.get_local_particles()
# To recognize attributes of 'particlesTmp':
# printAttributes(particlesTmp,'particlesTmp','bunch_origin.get_local_particles')
# 'particlesCrrnt' is 2D array: (numberoFParticle,(x,x',y,y',s,dE,ID));
particlesCrrnt = particlesTmp.real
print "                 particlesCrrnt:"
for prtcl in range(5):
    print "x (m) for particle {}: {}".format(prtcl,particlesCrrnt[prtcl,0])
    print "y (m) for particle {}: {}".format(prtcl,particlesCrrnt[prtcl,2])
    print "s (m) for particle {}: {}".format(prtcl,particlesCrrnt[prtcl,4])
# End of checking


#-------------------------------------------------
# For checking only:
#
# 1) Attributes:
# printAttributes(bunch,'bunch','synergia.optics.generate_matched_bunch_transverse')
# 2) Distributions X-Y, X-X', Y-Y' using method 'pltbunch.plot_bunch':
loclTitle = "\nThese distributions were constructed using \
'synergia.optics.generated_matched_bunch_transverse' method:\n"
print loclTitle
pltbunch.plot_bunch(bunch_origin)     
# 3) Distributions X-Y, X-X', Y-Y' using method 'plotcoordDistr':
bunchParticles = bunch_origin.get_local_particles()
# To recognize attributes of 'bunchParticles':
# printAttributes(bunchParticles,'bunchParticles', 'bunch.get_local_particles()')
plotcoordDistr(bunchParticles)
#--------------------------------------------------


# Steppers (YuE: both case 'splitoperator' and 'independent' work properly!):
# stepper = 'splitoperator'
stepper = 'independent'
if stepper == "splitoperator":
    # Use the Split operator stepper with a dummy collective operator
    # (with evenly-spaced steps)
    no_op = synergia.simulation.Dummy_collective_operator("stub")
    stepper = synergia.simulation.Split_operator_stepper(
                            lattice_simulator_Crrnt, no_op, 4)
elif stepper == "independent":
    # Use the Independent particle stepper (by element)
    stepper = synergia.simulation.Independent_stepper_elements(
                            lattice_simulator_Crrnt, 4)
else:
    sys.stderr.write("fodo.py: stepper must be either 'independent' or 'splitoperator'\n")
    sys.exit(1)

# Bunch simulator:
bunch_simulator = synergia.simulation.Bunch_simulator(bunch_origin)

'''
# Diagnostics:
diagnostic_flag = 'None'
for part in range(0, 0):
    bunch_simulator.add_per_step(synergia.bunch.Diagnostics_track("step_track_%02d.h5" % part,
                                                                   part))
if diagnostic_flag == 'step_full2':
    bunch_simulator.add_per_step(synergia.bunch.Diagnostics_full2("step_full2.h5"))
if diagnostic_flag == 'step_particles':
    bunch_simulator.add_per_step(synergia.bunch.Diagnostics_particles("step_particles.h5"))
for part in range(0, 0):
    bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_track("turn_track_%02d.h5" % part,
                                                                   part))
if diagnostic_flag == 'turn_full2':
    bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_full2("turn_full2.h5"))
if diagnostic_flag == 'turn_particles':
    bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_particles("turn_particles.h5"))
'''    

totalTurns = 20

#---------------------------
# Propagate
#---------------------------
# Ramp action is instantiated and passed to the propagator instance during the propagate method

        
plotAfterEachTurn = 10

print "\n-------------------\n"
print "           Nonlinear parameters are not changed"
print "\n-------------------\n"

bunch = bunch_origin
# For checking:
particlesTmp1 = bunch.get_local_particles()
particlesCrrnt1 = particlesTmp1.real
print "                 particlesCrrnt (again for linear):"
for prtcl in range(5):
    print "x (m) for particle {}: {}".format(prtcl,particlesCrrnt1[prtcl,0])
    print "y (m) for particle {}: {}".format(prtcl,particlesCrrnt1[prtcl,2])
    print "s (m) for particle {}: {}".format(prtcl,particlesCrrnt1[prtcl,4])
# End of checking

bunch_simulator = synergia.simulation.Bunch_simulator(bunch)

propagator = synergia.simulation.Propagator(stepper)
# propagator.set_checkpoint_period(0)
# propagator.set_checkpoint_with_xml(True)


# tracksLinear is 3D array: (totalTurns,bunchParticles,(x,y,s)) 
tracksLinear = np.zeros((totalTurns,bunchInParticles,3))

nUpdate = 0
totalTimeCPU = 0.
for turnCrrnt in range(totalTurns):
    timeStart = os.times()
    propagatorCrrnt = propagator.propagate(bunch_simulator, 1, 1, 0)
# To recognize attributes of 'propagatorCrrnt':
#    printAttributes(propagatorCrrnt,'propagatorCrrnt', \
#                    'propagator.propagate(bunch_simulator, ramp_actions, 1, 1, 0)')
# bunchParticles is 2D array: (numberParrticles,(x,x',y,y',s,dE,ID))
    bunchParticles = bunch.get_local_particles()
# coordsTracks is 2D array: (bunchParticles,(x,y,s)) 
    coordsTracks = tracksCoords(bunchParticles)
    numbPartcls = bunchParticles.shape[0]
    for prtcl in range(numbPartcls):
        for k in range(3):
            tracksLinear[turnCrrnt,prtcl,k] = coordsTracks[prtcl,k]
        if prtcl < 3:
            print "tracksLinear (turn {}) for particle {}: x = {} mm, y = {} mm, s = {} m". \
            format(turnCrrnt,prtcl,tracksLinear[turnCrrnt,prtcl,0], \
                   tracksLinear[turnCrrnt,prtcl,1],tracksLinear[turnCrrnt,prtcl,2])
    turnNumber = turnCrrnt+1
    timeEnd = os.times()
    timeOfTurn = float(timeEnd[0] - timeStart[0])              # CPU time in seconds
    totalTimeCPU += timeOfTurn
    print ('Turn %3d is completed (CPU time = %6.3f seconds)' % (turnNumber, timeOfTurn))
    sys.stdout.flush()
    nUpdate += 1
    if nUpdate == plotAfterEachTurn:
        nUpdate = 0
        print "\n              After {} turns:\n".format(turnNumber)
        timeStart = os.times()
        plotcoordDistr(bunchParticles)
        timeEnd = os.times()
        timeOfPlot = float(timeEnd[0] - timeStart[0])              # CPU time in seconds
        totalTimeCPU += timeOfPlot
        print ('\nPlotting is completed (CPU time = %6.3f seconds)\n' % timeOfPlot)
for prtcl in range(5):
    print "x (mm) for particle {}: {}".format(prtcl,tracksLinear[:,prtcl,0])
    print "y (mm) for particle {}: {}".format(prtcl,tracksLinear[:,prtcl,1])
    print "s (m) for particle {}: {}".format(prtcl,tracksLinear[:,prtcl,2])
plotTracks(tracksLinear,5)
print ('\nFor %5d turns CPU time = %6.3f seconds\n' % (totalTurns, totalTimeCPU))
        
updateAfterEachTurn = 10

print "\n-------------------\n"
print "           Nonlinear parameters will be CHANGED with values = 5% after each {} turns". \
format(updateAfterEachTurn)
print "\n-------------------\n"

bunch = bunch_origin
# For checking:
particlesTmp2 = bunch.get_local_particles()
particlesCrrnt2 = particlesTmp2.real
print "                 particlesCrrnt (again for nonlinear):"
for prtcl in range(5):
    print "x (m) for particle {}: {}".format(prtcl,particlesCrrnt2[prtcl,0])
    print "y (m) for particle {}: {}".format(prtcl,particlesCrrnt2[prtcl,2])
    print "s (m) for particle {}: {}".format(prtcl,particlesCrrnt2[prtcl,4])
# End of checking

bunch_simulator = synergia.simulation.Bunch_simulator(bunch)

propagator = synergia.simulation.Propagator(stepper)
# propagator.set_checkpoint_period(0)
# propagator.set_checkpoint_with_xml(True)

# tracksNonLinear is 3D array: (totalTurns,bunchParticles,(x,y,s)) 
tracksNonLinear = np.zeros((totalTurns,bunchInParticles,3))

nUpdate = 0
totalTimeCPU = 0.
for turnCrrnt in range(totalTurns):
    timeStart = os.times()
    propagatorCrrnt = propagator.propagate(bunch_simulator, 1, 1, 0)
# To recognize attributes of 'propagatorCrrnt':
#    printAttributes(propagatorCrrnt,'propagatorCrrnt', \
#                    'propagator.propagate(bunch_simulator, ramp_actions, 1, 1, 0)')
# bunchParticles is 2D array: (numberParrticles,(x,x',y,y',s,dE,ID))
    bunchParticles = bunch.get_local_particles()
# coordsTracks is 2D array: (bunchParticles,(x,y,s)) 
    coordsTracks = tracksCoords(bunchParticles)
    numbPartcls = bunchParticles.shape[0]
    for prtcl in range(numbPartcls):
        for k in range(3):
            tracksNonLinear[turnCrrnt,prtcl,k] = coordsTracks[prtcl,k]
        if prtcl < 3:
            print "tracksNonLinear (turn {}) for particle {}: x = {} mm, y = {} mm, s = {} m". \
            format(turnCrrnt,prtcl,tracksNonLinear[turnCrrnt,prtcl,0], \
                   tracksNonLinear[turnCrrnt,prtcl,1],tracksNonLinear[turnCrrnt,prtcl,2])
    turnNumber = turnCrrnt+1
    timeEnd = os.times()
    timeOfTurn = float(timeEnd[0] - timeStart[0])              # CPU time in seconds
    totalTimeCPU += timeOfTurn
    print ('Turn %3d is completed (CPU time = %6.3f seconds)' % (turnNumber, timeOfTurn))
    sys.stdout.flush()
    nUpdate += 1
    if nUpdate == updateAfterEachTurn:
        nUpdate = 0
        print "\n              After {} turns:\n".format(turnNumber)
        timeStart = os.times()
        plotcoordDistr(bunchParticles)
        ramp_actions = Ramp_actions(1.05,.95,1)   
        propagatorCrrnt = propagator.propagate(bunch_simulator, ramp_actions, 1, 1, 0)
        timeEnd = os.times()
        timeUpdateAndPlot = float(timeEnd[0] - timeStart[0])              # CPU time in seconds
        totalTimeCPU += timeUpdateAndPlot
        print ('\nUpdate and plotting are completed (CPU time = %6.3f seconds)\n' % timeUpdateAndPlot)
for prtcl in range(5):
    print "x (mm) for particle {}: {}".format(prtcl,tracksNonLinear[:,prtcl,0])
    print "y (mm) for particle {}: {}".format(prtcl,tracksNonLinear[:,prtcl,1])
    print "s (m) for particle {}: {}".format(prtcl,tracksNonLinear[:,prtcl,2])
plotTracks(tracksNonLinear,5)
print ('\nFor %5d turns CPU time = %6.3f seconds\n' % (totalTurns, totalTimeCPU))