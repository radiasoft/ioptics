
import numpy as np
import random
from scipy.optimize import newton


class EllipticWaterbag:
    
    def __init__(self, _t, _c, _beta, _betaPrime=0.):
        """ Generate a matched bunch for a fixed emittance
        Args:
        t (float) the elliptic potential strength
        c (float) the elliptic potential c
        beta (float) the beta function where the bunch is being matched
        betaPrime (float) the derivative of the beta function, defaults to zero
        """
        self.ellipticT = -1.*_t
        self.ellipticC = _c
        self.beta      = _beta
        self.betaPrime = _betaPrime

    def computeHamiltonian(self, xHat, pxHat, yHat, pyHat):
        """Compute the Hamiltonian (1st invariant) for the integrable elliptic potential"""

        quadratic = 0.5 * (pxHat**2 + pyHat**2) #+ 0.5 * (xHat**2 + yHat**2)

        elliptic = 0.
        kfac = 1.
        if self.ellipticT != 0.:
            xN = xHat / self.ellipticC
            yN = yHat / self.ellipticC

            # Elliptic coordinates
            u = (np.sqrt((xN + 1.)**2 + yN**2) +
                 np.sqrt((xN - 1.)**2 + yN**2))/2.
            v = (np.sqrt((xN + 1.)**2 + yN**2) -
                 np.sqrt((xN - 1.)**2 + yN**2))/2.

            f2u = u * np.sqrt(u**2 - 1.) * np.arccosh(u)
            g2v = v * np.sqrt(1. - v**2) * (-np.pi/2 + np.arccos(v))

            kfac = self.ellipticT * self.ellipticC**2
            elliptic = (f2u + g2v) / (u**2 - v**2)

        hamiltonian = quadratic + self.computePotential(xHat, yHat)
        return hamiltonian
        
    def computePotential(self, xHat, yHat):
        quadratic = 0.5 * (xHat**2 + yHat**2)

        elliptic = 0.
        kfac = 1.
        if self.ellipticT != 0.:
            xN = xHat / self.ellipticC
            yN = yHat / self.ellipticC

            # Elliptic coordinates
            u = ( np.sqrt((xN + 1.)**2 + yN**2) +
                  np.sqrt((xN - 1.)**2 + yN**2) )/2.
            v = ( np.sqrt((xN + 1.)**2 + yN**2) -
                  np.sqrt((xN - 1.)**2 + yN**2) )/2.

            f2u = u * np.sqrt(u**2 - 1.) * np.arccosh(u)
            g2v = v * np.sqrt(1. - v**2) * (-np.pi/2 + np.arccos(v))

            kfac = self.ellipticT * self.ellipticC**2
            elliptic = (f2u + g2v) / (u**2 - v**2)

        potential = quadratic + kfac * elliptic
        return potential
        
    def whatsLeft(self, yHat):
        return self.emittance - self.computePotential(0, yHat)
    
    def generate_waterbag(self, emittance, nParticles):
        """ Generate a matched bunch with single emittance and number of particles
        Args:
        emittance (float) the value of fixed H
        nParticles(int)   the number of particles for the bunch
        
        Returns:
        bunch (list)  a list of numpy arrays of 4D phase space, (x, px, y, py)
        """
        
        # Generate some bounds on the transverse size to reduce waste in generating the bunch
        

        
        # Generate particles by creating trials and finding particles with potential less than emittance, then assign the rest to momentum
        ptclsMade = 0
        phaseSpaceList = []
        while ptclsMade < nParticles:
            newH = emittance*random.random()
            y0 = np.sqrt(newH)
            self.emittance = newH
            yMax = newton(self.whatsLeft, y0) 
            xMax = self.ellipticC
            trialValue = 1e10
            while trialValue >= newH:
                xTrial = 2.*(0.5 - random.random())*xMax
                yTrial = 2.*(0.5 - random.random())*yMax
                trialValue = self.computePotential(xTrial, yTrial)
            initialValue = trialValue
            if initialValue < newH:
                pMag = np.sqrt(2*(newH - initialValue))
                pDir = 2*np.pi* random.random()
                pxHat = pMag * np.cos(pDir)
                pyHat = pMag * np.sin(pDir)
                xReal = xTrial * np.sqrt(self.beta)
                yReal = yTrial * np.sqrt(self.beta)
                pxReal = (pxHat + 0.5*self.betaPrime*xTrial)/np.sqrt(self.beta)
                pyReal = (pyHat + 0.5*self.betaPrime*yTrial)/np.sqrt(self.beta)
                ptclCoords = np.array([xReal, pxReal, yReal, pyReal])
                phaseSpaceList.append(ptclCoords)
                ptclsMade += 1
            else:
                print "Value out of bounds"
        
        return phaseSpaceList


    def generate_waterbag_shell(self, eps0, epsf, nParticles):
        """
        Generates a waterbag that uniformly fills from the 4D phase space from eps0 to epsf.
        Args:
        eps0 (float) Lower bound for H
        epsf (float) Upper bound for H
        nParticles(int)   the number of particles for the bunch

        Returns:
        bunch (list)  a list of numpy arrays of 4D phase space, (x, px, y, py)
        """

        # Generate some bounds on the transverse size to reduce waste in generating the bunch

        # Generate particles by creating trials and finding particles with potential less than emittance,
        # then assign the rest to momentum
        ptclsMade = 0
        phaseSpaceList = []
        while ptclsMade < nParticles:
            newH = (epsf - eps0) * random.random() + eps0
            y0 = np.sqrt(newH)
            self.emittance = newH
            yMax = newton(self.whatsLeft, y0)
            xMax = self.ellipticC
            trialValue = 1e10
            while trialValue >= newH:
                xTrial = 2. * (0.5 - random.random()) * xMax
                yTrial = 2. * (0.5 - random.random()) * yMax
                trialValue = self.computePotential(xTrial, yTrial)
            initialValue = trialValue
            if initialValue < newH:
                pMag = np.sqrt(2 * (newH - initialValue))
                pDir = 2 * np.pi * random.random()
                pxHat = pMag * np.cos(pDir)
                pyHat = pMag * np.sin(pDir)
                xReal = xTrial * np.sqrt(self.beta)
                yReal = yTrial * np.sqrt(self.beta)
                pxReal = (pxHat + 0.5 * self.betaPrime * xTrial) / np.sqrt(self.beta)
                pyReal = (pyHat + 0.5 * self.betaPrime * yTrial) / np.sqrt(self.beta)
                ptclCoords = np.array([xReal, pxReal, yReal, pyReal])
                phaseSpaceList.append(ptclCoords)
                ptclsMade += 1
            else:
                print "Value out of bounds"

        return phaseSpaceList


if __name__ == '__main__':

    myBunchGenerator = EllipticWaterbag(0.4, 0.01, 0.637494274, 0.0)

    list0 = np.array(myBunchGenerator.generate_waterbag(20.00e-6, 100000))

    print list0[99985,3],str(list0[99985,3])

    bunchFile = open('myNewWbBunch.txt', 'w')
    for idx in range(0,100000):
        ptclString = str(list0[idx,0])+' '+str(list0[idx,1])+' '+str(list0[idx,2])+' '\
                     +str(list0[idx,3])+' '+str(0.000000)+' '+str(0.000000)+'\n'
        bunchFile.write(ptclString)





