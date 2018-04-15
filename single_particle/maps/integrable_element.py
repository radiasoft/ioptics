import numpy as np

class integrable_element:

    def __init__(self, t_array, c_array, ds):
        """
        Class for representing a single integrable element.

        :t_array: -- values for the elliptic strength t
        :c_array: -- values for the elliptic width c in the
                     units of the particle coordinates
        :ds: -- The step size associated with each t_array
                and c_array value
        """

        self.n_steps = np.shape(t_array)[0]
        self.t_array = t_array
        self.c_array = c_array
        self.ds = ds

        self.integrable_kick = integrable_kick()

    def integrate_element(self, x, px, y, py):
        """
        Integrate a coordinate or set of coordinates
        across the specified element
        """
        step = 0
        while step < self.n_steps:

            self.integrable_kick.single_step(self.t_array[step],
                                             self.c_array[step],
                                             x, px, y, py, self.ds)

            step += 1


class integrable_kick:

    def __init__(self):
        """
        Class for taking a single step across the integrable element. Class uses
        a 4th order explicit symplectic integrator to isolate ds step size from
        physical dynamics.
        """
        # Time step parameters
        a = 2.**(1./3.)
        self.z0 = -a/(2.-a)
        self.z1 = 1./(2.-a)


    def single_step(self, t_nle, c_nle, x, px, y, py, ds):
        """
        Use a fourth order in step size integrator to isolate dynamics
        from the step size
        """

        #self.__2nd_order(t_nle, c_nle, x, px, y, py, self.z1*ds)
        #self.__2nd_order(t_nle, c_nle, x, px, y, py, self.z0*ds)
        #self.__2nd_order(t_nle, c_nle, x, px, y, py, self.z1*ds)
        
        self.__2nd_order(t_nle, c_nle, x, px, y, py, ds)

    def __2nd_order(self, t_nle, c_nle, x, px, y, py, ds):
        """
        A single second-order-in-ds step
        """

        self.__ioptics_drift(x, px, y, py, 0.5*ds)
        self.__renormalized_kick(t_nle, c_nle, x, px, y, py, ds)
        self.__ioptics_drift(x, px, y, py, 0.5*ds)

        #self.__ioptics_drift(x, px, y, py, 0.5*ds)
        #self.__ioptics_kick(t_nle, c_nle, x, px, y, py, ds)
        #self.__ioptics_drift(x, px, y, py, 0.5*ds)


    def __ioptics_drift(self, x, px, y, py, ds):
        """drift"""

        x += px * ds
        y += py * ds


    def __ioptics_kick(self, t_nle, c_nle, x, px, y, py, ds):
        """kick"""

        zed = x / c_nle + 1.j*y
        sqrt = np.sqrt(c_nle - zed*zed)
        arcsin = -c_nle * np.log(c_nle*zed + sqrt)
        Force = zed/(sqrt*sqrt) + arcsin/(sqrt*sqrt*sqrt)

        delta_px = (t_nle/c_nle)*np.real(Force)*ds
        delta_py = -(t_nle/c_nle)*np.imag(Force)*ds

        px += delta_px
        py += delta_py

    def __renormalized_kick(self, t_nle, c_nle, x, px, y, py, ds):
        """kick which converts coordinates to dimensionless quantities before computing"""

        #let's assume we are providing unnormalized coordinates (units m)
        #then we simply need to divide by c_nle = c*sqrt(beta) to get dimensionless units
        
        #compute complex z
        zed = (x + 1.j*y)/c_nle
        #compute sqrt
        sqrt = np.sqrt( 1 - zed*zed)
        #compute complex arcsin
        arcsin = -1.j * np.log(1.j*zed + sqrt) #make positive i inside log
        
        #dF is z/(1-z^2) + arcsin(z)/(1-z^2)^(3/2)
        Force = zed/(sqrt*sqrt) + arcsin/(sqrt*sqrt*sqrt)
        
        #dPx is real component, dPy is negative imaginary component
        #remember here that in dimensionless variables, kick strengtgh knll = tc^2/beta, and cnll = c*sqrt(beta)
        #so, knll/cnll = tc/beta^3/2 = dimensionless
        delta_px = (-1.*t_nle/c_nle)*np.real(Force)*ds
        delta_py = (t_nle/c_nle)*np.imag(Force)*ds
        
        
        px += delta_px
        py += delta_py
