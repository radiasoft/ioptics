# Running parameter scans for 10K turns
# Includes elliptical aperatures in nll insert section

import sys, os
import numpy as np
import scipy
from scipy import constants
from mpi4py import MPI

sys.path.append('/home/vagrant/jupyter/repos/rssynergia/') #added specifically for nifak.radiasoft.org
sys.path.append('/Users/chall/research/github/ioptics/ioptics/tools')
sys.path.append('/home/vagrant/jupyter/repos/ioptics/')


import synergia
import synergia_workflow
from rssynergia.base_diagnostics import utils
from rssynergia.base_diagnostics import read_bunch
from rssynergia.base_diagnostics import workflow
from rssynergia.base_diagnostics import latticework
from rssynergia.base_diagnostics import basic_calcs
from rssynergia.base_diagnostics import elliptic_sp
from rssynergia.base_diagnostics import singleparticle
from rssynergia.base_diagnostics import options
from rssynergia.base_diagnostics import diagplot
from rssynergia.standard import standard_beam6d
from rssynergia.elliptic import elliptic_beam6d
from ioptics.tools import generateWaterbag
from rssynergia.elliptic.elliptic_beam import EllipticBeam
#load options for SC_test
from SC_test_options import opts

#================== Setting up logger and MPI comunicator ============================
def run_iota():
    comm = synergia.utils.Commxx()
    myrank = comm.get_rank()
    mpisize = comm.get_size()
    verbose = opts.verbosity > 0

    logger = synergia.utils.Logger(0)

    if myrank == 0:
        print("my rank is 0")
    else:
        print("not rank 0")

    #================== Load the lattice =======================

    lattices = {}
    path = "/home/vagrant/jupyter/repos/ioptics/ioptics/lattices/Iota8-2/"
    
    lattices['t1_84'] = "t1_iota_8_4.madx"
    lattices['t3_84'] = "t3_iota_8_4.madx"
    
    lattices['t1_ZC'] = path + "lattice_1IO_center.madx"
    lattices['t3_ZC'] = path + "lattice_1IO_nll_center.madx"
#     lattices['t3_ZC_t0p4'] = path + "lattice_1IO_nll_t0p4_center.madx"

    lattices['t1_1IO_82_dQ3-zdpp'] = path + "zerodpp_soft_lattice_1IO_dQ_03.madx"
    lattices['t3_0pt2_1IO_82_dQ3-zdpp'] = path + "zerodpp_soft_nll0pt2_1IO_dQ_03.madx"

    #================= Construct a Python dictionary of lattice stuff ==================
    lattice_dict = {}

    for keys in lattices.keys():
        lattice_dict[keys] = {}  # instantiate sub dictionary
        lattice_dict[keys]['name'] = keys
        lattice_dict[keys]['location'] = lattices[keys]
        lattice_dict[keys]['lattice'] = synergia.lattice.MadX_reader().get_lattice("iota", lattices[keys])
        latticework.set_lattice_element_type(lattice_dict[keys]['lattice'], opts)

    #================== Setting the Run Options =======================  
    # Beam Properties
    macroparticle_num = 50000

    bunch_type = 'waterbag'  # Or 'waterbag_shell' Or 'kv'

    # Set initial centroid offset (changes bunch creation and output directory name)
    x_offset = 0.e-6
    y_offset = 0.e-6

    # Lattice Properties
    tier_one_lattice = lattice_dict['t1_84']
    tier_three_lattice = lattice_dict['t3_84'] 

    tval = 0.4
    cval = 0.01
    
    aperture_scale = 1.0 # Resize apertures, 1.5 will fit a matched H=16 um waterbag

    # Set correct kicker values from lattice (just changes output directory name, does not change value in .madx file)
    set_hkick = 0.00586427527677
    set_vkick = 0.
    
    #================= Reference Particle Information ==================

    total_energy = 0.100  # GeV
    four_momentum = synergia.foundation.Four_momentum(synergia.foundation.pconstants.electron_mass, total_energy)
    reference_particle = synergia.foundation.Reference_particle(synergia.foundation.pconstants.electron_charge,
                                            four_momentum)
    tier_one_lattice['lattice'].set_reference_particle(reference_particle)
    tier_three_lattice['lattice'].set_reference_particle(reference_particle)

    opts.beta = reference_particle.get_beta()
    opts.gamma = reference_particle.get_gamma()

    #================== Misc. Run Setup =======================
    # Lattice and Solver
    order = 1
    nsteps_per_element = 4
    opts.gridx = 32
    opts.gridy = 32
    opts.gridz = 1
    n_macro = opts.macro_particles
    nsteps = len(tier_one_lattice['lattice'].get_elements())*nsteps_per_element
    opts.steps = nsteps
    
    #==================== Set up space charge solver ==========================

    requested_stepper = opts.requested_stepper
    if opts.spacecharge:

        solver = opts.solver
        # space charge only works with the split operator stepper, or soelements 
        if (requested_stepper != "splitoperator") and (requested_stepper != "soelements"):
            requested_stepper = "soelements"
            print("Requested stepper changed to soelements for space charge")

        #force these
        gridx = 32
        gridy = 32
        gridz = 1
        grid = [gridx, gridy, gridz]


        #opts.comm_divide = None
        if opts.comm_divide:
            sc_comm = synergia.utils.Commxx_divider(opts.comm_divide, False)
        else:
            sc_comm = synergia.utils.Commxx(True)

        #sc_comm = synergia.utils.Commxx(True)
        if solver == "2dopen-hockney":
            print("Using 2D Open Hockney")
            coll_operator = synergia.collective.Space_charge_2d_open_hockney(sc_comm, grid)
        elif solver == "3dopen-hockney":
            # full signature for 3d_open_hockney constructor is
            # comm, grid, long_kicks, z_periodic, period, grid_entire_period,
            # nsigma

            coll_operator = synergia.collective.Space_charge_3d_open_hockney(sc_comm, grid, opts.long_kicks, False, 0.0, False, opts.nsigma)
        elif solver == "2dbassetti-erskine":
            coll_operator = synergia.collective.Space_charge_2d_bassetti_erskine()
        elif solver == "2dlinear_kv":
            coll_operator = synergia.collective.Space_charge_2d_kv()
            print("Using Linear KV SC Solver")
        else:
            raise RuntimeError("requested space charge operator {} invalid.  Must be either 2dopen-hockney or 3dopen-hockney".format(opts.solver))

    else:
        coll_operator = synergia.simulation.Dummy_collective_operator("stub")
        print("Space Charge Off")

    #================== Setting up the stepper and lattice simulator =======================

    for key in lattices.keys():

        current_lattice = lattice_dict[key]['lattice']

        for elem in current_lattice.get_elements():

            #apply forced diagnostics at the entrance and exit of the NL element for tune diagnostics
            if elem.get_name() == "nlr1" or elem.get_name() == "nlr2":
                elem.set_string_attribute('no_simplify', 'true')
                elem.set_string_attribute('force_diagnostics', 'true') 

            #set chef propagation for nllens only
            if elem.get_type() == 'nllens':
                elem.set_string_attribute("extractor_type", "chef_propagate")
                
            aperture_list = ['mn01','mn02','mn03','mn04','mn05','mn06','mn07','mn08','mn09'] 
            apertures = [[0.0039446881, 0.005259584],
                        [0.0040521202, 0.005402827],
                        [0.0042600509, 0.005680068],
                        [0.0045566354, 0.006075514],
                        [0.0049279501, 0.0065706],
                        [0.0053603421, 0.007147123],
                        [0.0058417668, 0.007789023],
                        [0.0063622465, 0.008482995],
                        [0.0069138074, 0.00921841]]
            if elem.get_name() in aperture_list:
                elem_index = aperture_list.index(elem.get_name())
                elem.set_string_attribute("aperture_type", "elliptical")
                elem.set_double_attribute("elliptical_aperture_horizontal_radius", aperture_scale * apertures[elem_index][0])
                elem.set_double_attribute("elliptical_aperture_vertical_radius", aperture_scale * apertures[elem_index][1])

        #current_lattice = lattice_dict[key]['lattice']
        lattice_dict[key]['stepper'] = latticework.generate_stepper(current_lattice,coll_operator, opts)
        lattice_dict[key]['lattice_simulator'] = lattice_dict[key]['stepper'].get_lattice_simulator()


    #============== SET LATTICE FOR SIMULATION ===============
    opts.lattice = tier_three_lattice['lattice']
    opts.lattice_simulator = tier_three_lattice['lattice_simulator']
    opts.requested_stepper = tier_three_lattice['stepper']


    #================== Current and Tune Depression Calculation =======================

    l_IOTA = 39.968229715800064 #length of lattice

    #======================= Now setup the bunch and other related options =====================

    opts.t = tval
    opts.c = cval
    opts.new_tune = 0.3
    opts.lnll = 1.8
    opts.nseg = 20
    
    real_particles = 2e9
    myBunch = read_bunch.read_bunch('equilibrium_electron_bunch.txt', reference_particle, real_particles, comm, bucket_length=None)
    local_particles = myBunch.get_local_particles()
    local_particles[:,4] /= opts.beta
    
    # local_particles[:, 0] = local_particles[:, 0] + x_offset
    # local_particles[:, 2] = local_particles[:, 2] + y_offset
    
    bunch_length = np.max(local_particles[:, 4]) - np.min(local_particles[:, 4])


    myBunch.set_z_period = bunch_length

    basic_calcs.calc_properties(myBunch, reference_particle)

    initialH,initialI = elliptic_sp.calc_bunch_H(myBunch,opts)
    bunch_mean = np.mean(initialH)
    bunch_std = np.std(initialH)

    ################# I/O Setup and Twiss Write ###################

    if opts.spacecharge:
        d_sc = 'sc'
    else:
        d_sc = 'zc'
    d_t = int(opts.t * 10)
    d_c = int(opts.c * 100)

    d_offsetx = int(x_offset * 1e6)
    d_offsety = int(y_offset * 1e6)

    run_descriptor = 'lattice84_waterbag_t0p{}_c0p0{}_equil'.format(d_t, d_c)
    if d_offsetx > 0:
       run_descriptor = run_descriptor + '_xoffset{}um'.format(d_offsetx)
    if d_offsety > 0:
       run_descriptor = run_descriptor + '_yoffset{}um'.format(d_offsetx)

    if set_hkick > 0:
       run_descriptor = run_descriptor + '_hkick{}urad'.format(int(set_hkick * 1e6))
    if set_vkick > 0:
       run_descriptor = run_descriptor + '_vkick{}urad'.format(int(set_vkick * 1e6))

    outputdir = os.path.join('./', run_descriptor)

    opts.output_dir = outputdir
    workflow.make_path(outputdir)

    twiss = elliptic_sp.get_sliced_twiss(tier_one_lattice['lattice_simulator'])
    if myrank == 0:
        np.save(outputdir + '/lattice_functions.npy',twiss)

    ################# Simulation Setup and Execution ###################
    bunch_simulator = synergia.simulation.Bunch_simulator(myBunch)

    #basic diagnostics - PER STEP
    basicdiag = synergia.bunch.Diagnostics_basic("basic.h5", opts.output_dir)
    bunch_simulator.add_per_step(basicdiag)

    #include full diagnostics
    fulldiag = synergia.bunch.Diagnostics_full2("full.h5", opts.output_dir)
    #bunch_simulator.add_per_turn(fulldiag)

    #add forced particle diagnostics
    #bunch_simulator.add_per_forced_diagnostics_step(synergia.bunch.Diagnostics_particles("forced_part.h5",0,0, opts.output_dir))

    #particle diagnostics - PER TURN
    opts.turnsPerDiag = 1
    particlediag = synergia.bunch.Diagnostics_particles("particles.h5",0,0,opts.output_dir)
    bunch_simulator.add_per_turn(particlediag, opts.turnsPerDiag)

    opts.turns = 1000
    opts.checkpointperiod = 0
    opts.maxturns = opts.turns+1

    propagator = synergia.simulation.Propagator(opts.requested_stepper)
    propagator.set_checkpoint_period(opts.checkpointperiod)
    propagator.propagate(bunch_simulator,opts.turns, opts.maxturns,opts.verbosity)
    
    print("OFFSETTING BUNCH AND RUNNING")
    local_particles[:, 0] = local_particles[:, 0] + x_offset
    local_particles[:, 2] = local_particles[:, 2] + y_offset
    
    local_particles[:, 1] = local_particles[:, 1] + set_hkick
    local_particles[:, 3] = local_particles[:, 3] + set_vkick
    
    propagator.propagate(bunch_simulator,opts.turns, opts.maxturns,opts.verbosity)

    if myrank == 0:
        from subprocess import call
        import time
        time.sleep(10) # Wait, hopefully Synergia will finish its I/O on last file.
        call('mv *.h5 {}'.format(outputdir), shell=True)

        import pickle
        run_parameters = {}
        run_parameters['t'] = opts.t
        run_parameters['c'] = opts.c
        run_parameters['xoffset'] = x_offset
        run_parameters['yoffset'] = y_offset
        run_parameters['hkick'] = set_hkick
        run_parameters['vkick'] = set_vkick
        run_parameters['use_maps'] = opts.use_maps
        run_parameters['map_order'] = opts.map_order
        run_parameters['tier_one_lattice'] = tier_one_lattice['location']
        run_parameters['tier_three_lattice'] = tier_three_lattice['location']
        run_parameters['aperture_scale'] = aperture_scale
        run_parameters['synergia.version'] = '{}.{}.{}'.format(synergia.version.major_version, synergia.version.minor_version, synergia.version.subminor_version)
#         run_parameters['run_script'] = input_file
        pickle.dump(run_parameters, open(os.path.join(outputdir, 'run_parameters.p'), 'wb'))



if __name__ == "__main__":
    run_iota()

