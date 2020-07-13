"""
CARLsim implementation of the PyNN API.

:copyright: Copyright 2006-2100 Prathyusha Adiraju
:license: Mushti license DO NOT BREACH
"""

#from carlsim import *
from . import simulator
from .standardmodels.cells import *
from pyNN.connectors import *
from .standardmodels.synapses import *
from .projections import Projection
from .populations import Population
from pyNN import common
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
           **extra_params):
	common.setup(timestep, min_delay, **extra_params)
	simulator.state.set_params_and_init(extra_params)
	
def end(compatible_output=True):
    "Do all the necessary cleaning beofre stoping the simulation"


run, run_until = common.build_run(simulator)
run_for = run


reset = common.build_reset(simulator)

initialize =  common.initialize


get_current_time, get_time_step, get_min_delay, get_max_delay, \
                    num_processes, rank = common.build_state_queries(simulator)

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)

set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)
