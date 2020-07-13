#!/usr/bin/env python

"""
setup.py file 
"""

from distutils.core import setup, Extension


carlsim_module = Extension('_carlsim',
                           sources=['carlsim_wrap.cxx', '../../carlsim/interface/src/carlsim.cpp',
                           '../../carlsim/interface/src/callback_core.cpp', 
                           '../../carlsim/interface/src/linear_algebra.cpp', 
                           '../../carlsim/interface/src/poisson_rate.cpp', 
                           '../../carlsim/interface/src/user_errors.cpp',
                           '../../carlsim/kernel/src/print_snn_info.cpp',
                           '../../carlsim/kernel/src/snn_cpu_module.cpp',
                           '../../carlsim/kernel/src/snn_manager.cpp',
                           '../../carlsim/kernel/src/spike_buffer.cpp',
                           '../../carlsim/monitor/connection_monitor.cpp',
                           '../../carlsim/monitor/connection_monitor_core.cpp',
                           '../../carlsim/monitor/group_monitor.cpp',
                           '../../carlsim/monitor/group_monitor_core.cpp',
                           '../../carlsim/monitor/spike_monitor.cpp',
                           '../../carlsim/monitor/spike_monitor_core.cpp',
                           '../../carlsim/monitor/neuron_monitor.cpp',
                           '../../carlsim/monitor/neuron_monitor_core.cpp',
                           '../../tools/spike_generators/spikegen_from_vector.cpp',
                           '../../tools/visual_stimulus/visual_stimulus.cpp'
                           ]
                           )


setup (name = 'carlsim',
       version = '0.2',
       description = """CARLsim simulator as Python library""",
       ext_modules = [carlsim_module],
       py_modules = ["carlsim"],
       )
