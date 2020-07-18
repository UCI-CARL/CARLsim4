#!/usr/bin/env python

"""
setup.py file 
"""

from distutils.core import setup, Extension
import os
import sysconfig
from Cython.Distutils import build_ext

class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)


def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-ffast-math", "-w", "-c"]


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
                           ], extra_compile_args=extra_compile_args, language='c++'
                           )


setup (name = 'carlsim',
       version = '0.2',
       description = """CARLsim simulator as Python library""",
       ext_modules = [carlsim_module],
       cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
       py_modules = ["carlsim"],
)


