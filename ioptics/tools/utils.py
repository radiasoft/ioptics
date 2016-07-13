import h5py as h5
import numpy as np
import os
import re
from elliptic import calc_bunch_Inv
from operator import itemgetter


def get_bunch(inputfile):
    """
    Load 6D particle data from Synergia particle output file.
    :param inputfile:
    :return particles - [particle,dimension]:
    """
    f = h5.File(inputfile,'r')

    charge = f['charge'][()]
    mass = f['mass'][()]
    pz = f['pz'][()]
    rep = f['rep'][()]
    sn = f['s_n'][()]
    tlen = f['tlen'][()]

    particles = np.array(f['particles'])
    npart = particles.shape[0]

    return particles

def sorted_turn_list(directory):
    """
    Return a list of sorted bunch files in a directory
    :param directory - Directory location with Synergia bunch files.:
    :return sortedList - List of files in directory sorted by turn number based on their name.:
    """
    fileList = []
    sortedList = []
    for fl in os.listdir(directory):
        if os.path.splitext(fl)[1] == '.h5' and fl.find('particles') > -1:
            filename = os.path.basename(fl)
            basename = os.path.splitext(filename)[0]
            fileList.append((int(re.findall(r'\d+', basename)[0]),filename))

    fileList = sorted(fileList, key=itemgetter(0))
    for tupl in fileList:
        sortedList.append(tupl[1])
    return sortedList


def lostlist(directory,npart):
    """
    Find particle data file from final turn in 'directory' and return a list of missing particles based on
    particle ID.
    :param directory -  Directory containing particle data files:
    :param npart -  Initial particle number for the simulation:
    :return lostlist - List of IDs for lost particles:
    """
    lostlist = []
    lastfile = sorted_turn_list(directory)[-1]
    print "Last turn in directory:%s" % lastfile
    bunchIn = get_bunch(directory + '/' + lastfile)
    print "Shape of last bunch array: (%s,%s)" % (bunchIn.shape[0], bunchIn.shape[1])
    for i in range(npart):
        if not np.any(bunchIn[:, 6] == i):
            lostlist.append(i)

    return lostlist

def get_invariants(directory,npart,beta,alpha,t,c):
    """
    Calculate the invariants for particles in an elliptic potential for a series of particle files. Excludes
    lost particles from calculation.
    :param directory -  Directory with particle files:
    :param npart -  Largest number of particles in a file:
    :param beta - Design beta value:
    :param alpha -  Design alpha value:
    :param t -  Nonlinear magnet strength:
    :param c -  Nonlinear magnet c parameter:
    :return:
    """
    Harray = []
    Iarray = []
    lostParts = lostlist(directory, npart)

    filelist = sorted_turn_list(directory)

    for bunchFile in filelist:
        if bunchFile.endswith('.h5') and bunchFile.find('particles') != -1:
            bunchIn = get_bunch(directory + '/' + bunchFile)


            for lost in lostParts:
                rowlost = np.where(bunchIn[:, 6] == lost)[0]
                try:
                    rowval = rowlost[0]
                except IndexError:
                    rowval = None
                if rowval != None:
                    bunchIn = np.delete(bunchIn, rowval, 0)

                rowval = None

            sBunch = bunchIn[np.argsort(bunchIn[:, 6])]
            Hval, Ival = calc_bunch_Inv(sBunch, beta, alpha, t, c)
            Harray.append(Hval)
            Iarray.append(Ival)

    return np.transpose(np.array(Harray)), np.transpose(np.array(Iarray))


def calculate_invariants(array,beta,alpha,t,c):
    """
    Calculates invariants from array already loaded to memory of form [turn,particle,dimension]
    :param array -  Array of particle coordinates from which to calculate invariants:
    :param beta - Design beta value:
    :param alpha -  Design alpha value:
    :param t - Nonlinear magnet strength:
    :param c -  Nonline magnet c parameter:
    :return:
    """
    Harray = []
    Iarray = []
    for turn in range(array.shape[0]):
        Hval,Ival = calc_bunch_Inv(array[turn,:,:],beta,alpha,t,c)
        Harray.append(Hval)
        Iarray.append(Ival)

    return np.transpose(np.array(Harray)),np.transpose(np.array(Iarray))


def get_all_turns(directory,npart,mod=1):
    """
    Pull particle data from multiple files and return a 3D array with all 6D particle data over multiple turns.
    Excludes lost particles from return.
    :param directory -  directory with particle data:
    :param npart - maximum number of particles in a file:
    :param mod - Optional, Set interval of files to grab (mod=1 is all, mod=4 would be every 4th):
    :return 3D array with structure [turnNumber,particle,dimension]:
    """
    turn = []

    lostParts = lostlist(directory, npart)
    filelist = sorted_turn_list(directory)

    for i, bunchFile in enumerate(filelist):
        if i % mod == 0:
            bunchIn = get_bunch(directory + '/' + bunchFile)


            for lost in lostParts:
                rowlost = np.where(bunchIn[:, 6] == lost)[0]
                try:
                    rowval = rowlost[0]
                except IndexError:
                    rowval = None
                if rowval != None:
                    bunchIn = np.delete(bunchIn, rowval, 0)

                rowval = None

            sBunch = bunchIn[np.argsort(bunchIn[:, 6])]
            turn.append(sBunch)
            #print bunchFile
    return np.array(turn)

def get_all_lost(directory,npart):

    turn = []
    lostParts = []
    lost_list = utils.lostlist(directory, npart)
    filelist = utils.sorted_turn_list(directory)

    for i, bunchFile in enumerate(filelist):
        if i % 4 == 0:
            bunchIn = get_bunch(directory + '/' + bunchFile)


            for lost in lost_list:
                rowlost = np.where(bunchIn[:, 6] == lost)[0]
                try:
                    rowval = rowlost[0]
                except IndexError:
                    rowval = None
                if rowval != None:
                    lostParts.append(bunchIn[rowval,:])

                #rowval = None

            sBunch = lostParts[np.argsort(lostParts[:, 6])]
            turn.append(sBunch)
            #print bunchFile
    return np.array(turn)