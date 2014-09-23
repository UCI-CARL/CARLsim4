# Main Makefile for compiling, testing, and installing CARLsim
# these variables collect the information from the other modules
# TODO: decide whether or not to allow uses to make examples etc.
# TODO: name the library the correct number and update the wiki
# with the approach for the library name.
# TODO: Update User Guide on wiki with correct layout.

default_targets :=
common_sources :=
common_objs :=
output_files :=
libraries :=
objects :=

# carlsim components
kernel_dir     = carlsim/kernel
interface_dir  = carlsim/interface
spike_mon_dir  = carlsim/spike_monitor
spike_gen_dir  = tools/carlsim_addons/spike_generators
server_dir     = carlsim/server
test_dir       = carlsim/test

# carlsim tools
input_stim_dir = tools/carlsim_addons/input_stimulus

# CARLsim flags specific to the CARLsim installation
CARLSIM_FLAGS += -I$(kernel_dir)/include -I$(interface_dir)/include -I$(spike_mon_dir)

# CAUTION: order of .mk includes matters!!!
include user.mk
include carlsim/carlsim.mk
include carlsim/test/gtest.mk
include carlsim/test/carlsim_tests.mk

# *.dat and results files are generated during carlsim_tests execution
output_files += *.dot *.log tmp* *.status *.dat results

# this blank 'default' is required
default:

.PHONY: default clean distclean
default: $(default_targets)

examples: $(carlsim_programs)

pti-examples: $(pti_programs)

# TODO: need to create a results directory in the main directory and delete later with
# every carlsim_tests run. distclean/clean should take care of it.
# TODO: create Makefile for devs that want to build from src
clean:
	$(RM) $(objects)
# TODO: see what distclean should really do by convention
distclean:
	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(libraries) $(output_files)

devtest:
	@echo $(CARLSIM_SRC_DIR) $(carlsim_tests_objs)

# TODO: rewrite help instructions to be similar to the ECJ approach; formatted nicer.
help:
	@echo -e '\n'Type \'make\' or \'make all\' to make CARLsim and CARLsim \
	examples.'\n'
	@echo -e Type \'make pti\' to make the pti library, install it, \
	and make the pti examples.'\n'
	@echo -e Type \'make uninstall\' to uninstall the pti library.'\n'
	@echo -e To compile a specific example, type \'make \<example folder \
	name\>\'.'\n'
	@echo -e Note: simpleEA, tuneFiringRates, and SORFs examples \
	require CARLsim PTI installation.'\n'
