import numpy as np

from maps.ring_map import ring_map
from maps.integrable_element import integrable_element

from elliptic_post_processing import ellipticPostProcessing as epp

from matplotlib import pyplot as plt


#
# Potential parameters
#

t = 0#0.4   # the physical strength should follow t/beta
c = 0.001 # the physical aperture should follow c*sqrt(beta)

#
# Particle initial conditions
#

x0 = 0.3*c*(1. - 2.*np.random.rand())
y0 = 0.3*c*(1. - 2.*np.random.rand())
px0 = 0.
py0 = 0.

#
# lattice parameters
#

nli_length_0 = 1.0 # meters
n_magnet_elems = 10
focal_length = .3 # meters

#
# simulation parameters
#

length_decrement = 1.2 # reduce the drift length by this factor every step
n_steps = 10
n_phases = 10

#
# Run the simulation
#

step = 0
phase_step = 0

nli_length = nli_length_0

k = 1./focal_length
ring_matrix = np.array([[1., 0., 0., 0.],
						[-k, 1., 0., 0.],
						[0., 0., 1., 0.],
						[0., 0., -k, 1.]])

x = np.array([x0])
y = np.array([y0])
px = np.array([px0])
py = np.array([py0])

def compute_beta(beta0, gamma0, alpha0, ds, n_steps):
    
    beta_array = []
    s_array = []
    
    sval = 0.5*ds
    # drift transport matrix
    m11 = 1.
    m12 = ds
    m12O2 = 0.5*m12
    m21 = 0.
    m22 = 1.
    
    betaNew = m11**2*beta0 - 2*m11*m12O2*alpha0 + m12O2**2*gamma0
    alphaNew = -m11*m21*beta0 + (m11*m22+m12O2*m21)*alpha0 - m12O2*m22*gamma0
    gammaNew = m21**2*beta0 - 2*m21*m22*alpha0 + m22**2*gamma0
    # start midway into the magnet
    
    print "num steps: {}".format(n_steps)
    print "step size ds: {}".format(ds)
    
    
    step = 0
    while step < n_steps:
        print "step {} has sval {}".format(step+1,sval)
        
        s_array.append(sval)
        beta_array.append(betaNew)

        betaBegin = betaNew
        alphaBegin = alphaNew
        gammaBegin = gammaNew

        betaNew = m11**2*betaBegin - 2*m11*m12*alphaBegin + m12**2*gammaBegin
        alphaNew = -m11*m21*betaBegin + (m11*m22+m12*m21)*alphaBegin - m12*m22*gammaBegin
        gammaNew = m21**2*betaBegin - 2*m21*m22*alphaBegin + m22**2*gammaBegin

        sval+= ds

        step += 1
    
    return np.array(s_array), np.array(beta_array)

delta_H = []
phase = []

ring_flip = ring_map(ring_matrix)

ds = nli_length / n_magnet_elems
# recompute beta for the new drift length
beta = np.sqrt(nli_length/k)*np.sqrt(1. - nli_length*k/4.)
hamiltonian = epp(t, c, beta)


print "x0: {}".format(x)
print "px0: {}".format(px)
print "y0: {}".format(y)
print "py0: {}".format(py)
H0 = hamiltonian.processparticles(x,px,y,py)
print "Initial Hamiltonian is: {}".format(H0)
ring_flip.integrate_element(x, px, y, py)
print "x1: {}".format(x)
print "px1: {}".format(px)
print "y1: {}".format(y)
print "py1: {}".format(py)
HR = hamiltonian.processparticles(x,px,y,py)
print "After ring flip: {}".format(HR)



H0 = hamiltonian.computeHamiltonian(x0/np.sqrt(beta), px0*np.sqrt(beta),
									y0/np.sqrt(beta), py0*np.sqrt(beta))

print "H0 is {}".format(H0)

s_array, beta_array = compute_beta(beta, 1./beta, 0., nli_length/n_magnet_elems, n_magnet_elems)
# Set the element strengths for the integrable element
ts_1 = t*c*c/ beta_array #t/beta_array
cs_1 = c * np.sqrt(beta_array)

# We are using symmetry here
ts_2 = np.flip(ts_1, 0)
cs_2 = np.flip(cs_1, 0)

t_array = t*c*c/beta_array
c_array = c*np.sqrt(beta_array)

first_integrable_element = \
	integrable_element(ts_1, cs_1, ds)

last_integrable_element = \
	integrable_element(ts_2, cs_2, ds)

first_integrable_element.integrate_element(x, px, y, py)
ring_flip.integrate_element(x, px, y, py)
last_integrable_element.integrate_element(x, px, y, py)

Hf = hamiltonian.computeHamiltonian(x[0]/np.sqrt(beta), px[0]*np.sqrt(beta),
									y[0]/np.sqrt(beta), py[0]*np.sqrt(beta))

print "Hfinal is {}".format(Hf)

#print phase
#print beta_array
#print delta_H

#nli_length_0 = 1.0

#beta0 = np.sqrt(nli_length_0/k)*np.sqrt(1. - nli_length_0*k/4.)
#s_array_0, beta_array0 = compute_beta(beta0, 1./beta0, 0., nli_length_0/n_magnet_elems, n_magnet_elems)

#print beta_array0.shape
#s_array = np.linspace(0,nli_length_0,len(beta_array0))
#print s_array.shape

plotting = True

if plotting:
    fig = plt.figure()
    ax = fig.gca()
    #ax.plot(phase, delta_H, c='r')
    ax.plot(s_array, beta_array, c='b')
    ax.plot(np.flip(-1.*s_array,0),np.flip(beta_array,0),c='r')
    ax.set_xlabel('s')
    ax.set_ylabel(r'$\beta$')
    #ax.set_xlabel('phase  through NLI [Rad]')
    #ax.set_ylabel(r'$\delta$H')
    ax.set_title('Updated integrator')
    plt.show()
    #fig.savefig('updated_integrator.png',bbox_inches='tight')
