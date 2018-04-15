from integrable_element import integrable_element as ie
import numpy as np

class ring_map:

    def __init__(self, matrix):

        self.matrix_map = matrix


    def integrate_element(self, x, px, y, py):

        self.__apply_matrix(x, px, y, py)


    def __apply_matrix(self, x, px, y, py):

        zeds = np.zeros((4, np.shape(x)[0]))

        zeds[0,:] = x[:]
        zeds[1,:] = px[:]
        zeds[2,:] = y[:]
        zeds[3,:] = py[:]

        zeds = np.einsum('ij,jk->ik', self.matrix_map, zeds)

        x[:] = zeds[0,:]
        px[:] = zeds[1,:]
        y[:] = zeds[2,:]
        py[:] = zeds[3,:]
