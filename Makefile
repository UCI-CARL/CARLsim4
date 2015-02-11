# Main Makefile for compiling, testing, and installing CARLsim
# these variables collect the information from the other modules

carlsim_major_num := 3
carlsim_minor_num := 0
carlsim_build_num := 0

default_targets :=
common_sources :=
common_objs :=
output_files :=
libraries :=
objects :=

# carlsim components
kernel_dir     = carlsim/kernel
interface_dir  = carlsim/interface
conn_mon_dir   = carlsim/connection_monitor
spike_mon_dir  = carlsim/spike_monitor
group_mon_dir  = carlsim/group_monitor
server_dir     = carlsim/server
test_dir       = carlsim/test

# carlsim tools
tools_dir            = tools
tools_spikegen_dir   = $(tools_dir)/spike_generators
tools_visualstim_dir = $(tools_dir)/visual_stimulus
tools_swt_dir        = $(tools_dir)/simple_weight_tuner

# CARLsim flags specific to the CARLsim installation
CARLSIM_FLAGS += -I$(kernel_dir)/include -I$(interface_dir)/include \
				 -I$(tools_spikegen_dir) -I$(tools_visualstim_dir) \
				 -I$(spike_mon_dir) -I$(conn_mon_dir) -I$(group_mon_dir) -I$(tools_swt_dir)

# CAUTION: order of .mk includes matters!!!
include user.mk
include carlsim/carlsim.mk
include carlsim/libcarlsim.mk

# *.dat and results files are generated during carlsim_tests execution
output_files += *.dot *.log tmp* *.status *.dat results carlsim/*.a

# this blank 'default' is required
default:

.PHONY: default clean distclean
default: $(default_targets)

clean:
	$(RM) $(objects)

distclean:
	$(RM) $(objects) $(libraries) $(output_files) doc/html

devtest:
	@echo $(objects) $(libraries) $(output_files) doc/html

# Print a help message
help:
	@ echo
	@ echo 'CARLsim Makefile options:'
	@ echo
	@ echo "make            Compiles the CARLsim code using the default compiler"
	@ echo "make all          (Same thing)"
	@ echo "make install    Installs CARLsim library (may require root privileges)"
	@ echo "make clean      Cleans out all object files"
	@ echo "make distclean  Cleans out all objects files and output files"
	@ echo "make help       Brings up this message!"
