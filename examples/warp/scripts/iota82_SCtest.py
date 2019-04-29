from warp import * 
from warp.data_dumping.openpmd_diag.particle_diag import ParticleDiagnostic
from iota82_1IO_dQ_1 import generateIOTA
from iota82_1IO_dQ_1 import Dedge
import TransverseDiagnostic as tdiag

import matplotlib.pyplot as pyplt 
import numpy as np
import os

##################
### Initialize ###
##################

def setup():
    pass


diagDir = 'diags/xySlice/hdf5'

def cleanupPrevious(outputDirectory = diagDir):
    if os.path.exists(outputDirectory):
        files = os.listdir(outputDirectory)
        for file in files:
            if file.endswith('.h5'):
                os.remove(os.path.join(outputDirectory,file))

cleanupPrevious()

##########################################
### Create Beam and Set its Parameters ###
##########################################

top.lrelativ = True
top.relativity = 1

beam = Species(type=Proton, name='Proton') 

#Initial distribution determined from matched envelope in Synergia
top.emitx = 4.0*5.0e-6 
top.emity = 4.0*5.0e-6
beam.a0 = sqrt(top.emitx * 0.627584359882) 
beam.b0 = beam.a0 
beam.ap0 = -1 * top.emitx * -0.0113761749091 / beam.a0 
beam.bp0 = -1 * top.emity * 0.0104556853994  / beam.b0 

beam.ibeam = 4.25e-3 
beam.ekin = 2.5e6 
beam.vthz = 0 
beam.ibeam = beam.ibeam/(top.gammabar**2) 


derivqty() 

top.npmax = 40000

w3d.distrbtn = "K-V" 

w3d.cylinder = True  

#####################
### Setup Lattice ###
#####################

turnLength = 39.9682297148
steps = 4000.


top.zlatstrt = 0.   #  z of lattice start (added to element z's on generate).  
top.zlatperi = turnLength # Lattice periodicity


top.dt = turnLength / steps / beam.vbeam  

madtowarp(generateIOTA())


#############  THIN LENS DIPOLE EDGES ######################

temp = Dedge(h=1.428571428300,e1=0.000000,hgap=0.029,fint=0.500000)

stepsize = 39.9682297148 / steps 
resetlat()
drawlattice(height=0.7, narc=25); limits(square=1)

Rxx, Ryy = temp.tomatrix() #Get matrix vales for edge element

print "Rxx: %s Ryy: %s" % (Rxx[1,0],Ryy[1,0])

dipolocs = array(list(top.dipozs)+list(top.dipoze)) # apparently warp imports numpy
print dipolocs
ndipolocs = nint(dipolocs/stepsize)
print ndipolocs

def thin_lens(Rxx, Ryy):
    """ thin_lens(Rxx, Ryy): apply a thin lens kick with a strength invf = 1/f [1/m]
        where invf is the R21 element of Rxx and Ryy
    """
    top.pgroup.uxp += Rxx[1,0]*top.pgroup.xp*top.pgroup.uzp
    top.pgroup.uyp += Ryy[1,0]*top.pgroup.yp*top.pgroup.uzp

def thin_lens_lattice(Rxx=Rxx, Ryy=Ryy):
    """ thin_lens_lattice(Rxx=Rxx, Ryy=Ryy):
        Generate a lattice of thin lenses at locations given by ndipolocs
    """
    if (top.it%steps) in ndipolocs:
        print "DIPOLE EDGE KICK",top.it
        thin_lens(Rxx, Ryy)

#############################
### Simulation Parameters ###
#############################


top.prwall = pr1 = 0.14

#Set cells
w3d.nx = 256
w3d.ny = 256
w3d.nz = 1



#Set boundaries
w3d.xmmin = -0.14 
w3d.xmmax =  0.14 
w3d.ymmin = -0.14 
w3d.ymmax =  0.14 
w3d.zmmin = -2e-3
w3d.zmmax =  2e-3

top.pboundxy = 0          



top.ibpush   = 2

top.fstype = 1

###################################
### Envelop Solver and Plotting ###
###################################

env.zl = 0  #Start of lattice
env.zu = 39.9682297148 #End of lattice

package("env")
generate()
step()

# xenv = env.aenv
# yenv = env.benv
# zCoor = env.zenv


# envData = np.concatenate((np.transpose(np.array([env.zenv])),np.transpose(np.array([env.aenv])),np.transpose(np.array([env.benv]))),axis=1)

# np.savetxt('envelopeData.txt',envData, delimiter = ' ')

# pyplt.figure()
# pyplt.plot(zCoor, xenv)
# pyplt.plot(zCoor, yenv)
# pyplt.savefig('beamSigma_Drift.png')




############################
### Particle Diagnostics ###
############################

diagP0 = ParticleDiagnostic( period=1, top=top, w3d=w3d,
        species= { species.name : species for species in listofallspecies },
        comm_world=comm_world, lparallel_output=False, write_dir = diagDir[:-4] )

diagP = ParticleDiagnostic( period=4000, top=top, w3d=w3d,
        species= { species.name : species for species in listofallspecies },
        comm_world=comm_world, lparallel_output=False, write_dir = diagDir[:-4] )

installafterstep(diagP0.write)
installafterstep(diagP.write)

#################################
### Generate and Run PIC Code ###
#################################

package("wxy")
generate()
fieldsolve()

td1 = tdiag.TransverseDiagnostic()

installafterstep(thin_lens_lattice)
installafterstep(td1.record)

#Execute First Step

step(1)

uninstallafterstep(diagP0.write)

#Execute Remainder

step(steps - 1)

td1.dataWrite(1)

for i in range(49):
    step(steps - 1)
    td1.dataWrite(i + 2)


