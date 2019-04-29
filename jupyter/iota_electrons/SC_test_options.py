#!/usr/bin/env python

import synergia_workflow

opts = synergia_workflow.Options("iota")


opts.add("map_order", 1, "Map order", int)
#default output directory
opts.add("output_dir","iota66_1IO_nl","Directory for output files", str)
#opts.add("steps", steps, "Number of steps per turn", int)
opts.add("steps_per_element",5,"Number of steps per element", int)


opts.add("verbosity", 1, "Verbosity of propagation", int)
opts.add("turns", 1000, "Number of turns", int)
opts.add("maxturns", 2000, "Maximum number of turns to run before checkpointing and quitting", int)
opts.add("checkpointperiod", 3000, "Number of turns to run between checkpoints", int)

opts.add("radius", 0.5, "aperture radius [m]", float)
opts.add("emit",9.74e-6, "H0 value corresponding to real sigma horizontal emittance of 0.3 mm-mrad", float)
opts.add("stdz", 0.05, "sigma read z [m]", float) #5 cm bunch length for IOTA
opts.add("dpop", 0.0, "Delta-p/p spread", float)

opts.add("macro_particles", 10 * 10240, "Number of macro particles", int)
opts.add("real_particles", 1.0e11, "Number of real particles", float)
opts.add("seed", 349250524, "Pseudorandom number generator seed", int)

opts.add("bunch_file","myBunch.txt","txt file for bunch particles", str)

#----------Space Charge Stuff---------------------
opts.add("gridx", 32, "grid points in x for solver", int)
opts.add("gridy", 32, "grid points in y for solver", int)
opts.add("gridz", 1, "grid points in z for solver", int)
opts.add("spacecharge", False, "whether space charge is on", bool)
opts.add("solver", "2dopen-hockney", "solver to use, '2dopen-hockney','3dopen-hockney', '2dbassetti-erskine', '2dlinear_kv'", str)

#options for controlling chef propagation vs. chef mapping!
opts.add("use_maps", "none", "use maps for propagation either all, none, onlyrf, nonrf")    #none means chef propagate
opts.add("requested_stepper", "splitoperator", "Simulation stepper, either 'independent','elements','splitoperator','soelements'", str)

#----------MPI STUFF---------------------
opts.add("comm_divide", 18, "size of communicator")
opts.add("concurrent_io", 8, "number of concurrent io threads for checkpointing", int)
