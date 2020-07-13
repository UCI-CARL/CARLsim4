"""
Definition of default parameters (and hence, standard parameter names) for
standard current source models.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN.standardmodels import StandardCurrentSource
from pyNN.parameters import Sequence
from .. import simulator
from ..carlsim import *

class DCSource(StandardCurrentSource):
    """Source producing a single pulse of current of constant amplitude.

    Arguments:
        `start`:
            onset time of pulse in ms
        `stop`:
            end of pulse in ms
        `amplitude`:
            pulse amplitude in nA
    """
    def __init__(self, amplitude):
        self.amplitude = float(amplitude)

    def inject_into(self, group):
        simulator.state.network.setExternalCurrent(group, self.amplitude)
