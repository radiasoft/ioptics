import numpy as np


def normalized_coord(bunch,betax,alphax,betay,alphay):

    newBunch = np.copy(bunch)

    newBunch[:,0] = bunch[:, 0] / np.sqrt(betax)
    newBunch[:,1] = (betax * bunch[:,1] + alphax * bunch[:,0]) / np.sqrt(betax)
    newBunch[:,2] = bunch[:, 2] / np.sqrt(betay)
    newBunch[:,3] = (betay * bunch[:,3] + alphay * bunch[:,2]) / np.sqrt(betay)

    return newBunch


def elliptic_coord(normalizedBunch,beta,t,c):
    eCoord = np.copy(normalizedBunch)

    eCoord[:,0] = eCoord[:,0]*1.0/c
    eCoord[:,2] = eCoord[:,2]*1.0/c

    u = 0.5*(np.sqrt((eCoord[:,0] + 1.)**2 + eCoord[:,2]**2) + np.sqrt((eCoord[:,0] -1.)**2 + eCoord[:,2]**2))
    v = 0.5*(np.sqrt((eCoord[:,0] + 1.)**2 + eCoord[:,2]**2) - np.sqrt((eCoord[:,0] -1.)**2 + eCoord[:,2]**2))

    return [u,v]


def calc_bunch_hamiltonian(normalizedBunch):

    x = normalizedBunch[:,0]
    y = normalizedBunch[:,2]
    px = normalizedBunch[:,1]
    py = normalizedBunch[:,3]

    H = 0.5 * (px**2 + py**2) + 0.5*(x**2 + y**2)

    return H


def elliptic_hamiltonian(u,v,beta,t,c):

    f2u = u * np.sqrt(u**2 -1.) * np.arccosh(u)
    g2v = v * np.sqrt(1. - v**2) * (-np.pi/2 + np.arccos(v))

    elliptic = (f2u + g2v) / (u**2 - v**2)
    kfac = -1.*t*c*c

    return kfac*elliptic


def calc_second_invariant(normalizedBunch,u,v,beta,t,c):

    t = -1.0*t
    c = 1.0*c

    x = normalizedBunch[:,0]
    px = normalizedBunch[:,1]
    y = normalizedBunch[:,2]
    py = normalizedBunch[:,3]

    p_ang = (x*py - y*px)**2
    p_lin = (px*c)**2

    #harmonic part of potential
    f1u = c**2 * u**2 * (u**2 -1.)
    g1v = c**2 * v**2 * (1.-v**2)

    #elliptic part of potential
    f2u = -t * c**2 * u * np.sqrt(u**2-1.) * np.arccosh(u)
    g2v = -t * c**2 * v * np.sqrt(1.-v**2) * (0.5*np.pi - np.arccos(v))

    #combined - Adjusted this from Stephen's code
    fu = (0.5 * f1u - f2u)
    gv = (0.5 * g1v + g2v)

    invariant = (p_ang + p_lin) + 2.*(c**2) * (fu * v**2 + gv * u**2)/(u**2 - v**2)

    return invariant


def calc_bunch_Inv(bunch,beta,alpha,t,c):
    norm_coords = normalized_coord(bunch,beta,alpha,beta,alpha)

    u,v = elliptic_coord(norm_coords,beta,t,c)
    hArray = calc_bunch_hamiltonian(norm_coords) + elliptic_hamiltonian(u,v,beta,t,c)
    iArray = calc_second_invariant(norm_coords,u,v,beta,t,c)

    return hArray,iArray