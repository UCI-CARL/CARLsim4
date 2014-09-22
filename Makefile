# Main Makefile for compiling, testing, and installing CARLsim
# TODO: make issue for automatically setting the correct CUDA
# environment variables.
# these variables collect the information from the other modules
default_targets :=
common_sources :=
common_objs :=
output_files :=
libraries :=
objects :=

# carlsim components
#kernel_dir     = carlsim/kernel
#interface_dir  = carlsim/interface
#spike_mon_dir  = carlsim/spike_monitor
#spike_gen_dir  = tools/carlsim_addons/spike_generators
# TODO: comment the rest of these out to make sure we don't need them.
#server_dir     = carlsim/server
#test_dir       = carlsim/test
# carlsim tools
#input_stim_dir = tools/carlsim_addons/input_stimulus
# additional directories
# TODO: double check that I don't need these any more. I shouldn't. Each one
# will have it's own Makefile
ex_dir         = examples
proj_dir       = projects

# location of .h files
vpath %.h $(EO_INSTALL_DIR)/src $(kernel_dir)/include \
$(ex_dir)/common $(interface_dir)/include $(spike_mon_dir) \
$(spike_gen_dir) $(test_dir)

# location of .cpp files
vpath %.cpp $(kernel_dir)/src $(interface_dir)/src $(test_dir) \
$(spike_info_dir) $(spike_gen_dir) $(ex_dir)/common/
# location of .cu files
vpath %.cu $(kernel_dir)/src $(spike_gen_dir) $(test_dir)
# location of .h files
# TODO: remove EO_INSTALL_DIR part when I'm done
# TODO: add ECJ stuff here if I need to
# TODO: remove $interface_dir)/include when you do the same patsubst
# as spike_monitor
# TODO: make the help in this makefile similar to the ECJ help. It is
# formatted nicer.
# TODO: Maybe put init.mk into common.mk or something like that?
# TODO: Move spike_generator to carlsim folder and fix the makefile

# core makefile includes
include user.mk
# TODO: maybe this should be moved to just carlsim folder and called carlsim.mk and then
# we could have explicit kernel things in the carlsim.mk makefile. I think this is
# what I will do.
include carlsim/carlsim.mk
include carlsim/test/gtest.mk
include carlsim/test/carlsim_tests.mk

# *.dat and results files are generated during carlsim_tests execution
output_files += *.dot *.log tmp* *.status *.dat results

# TODO: fix these eventually
# include all directories in examples
# TODO: remove this
#example_includes := $(addsuffix /examples.mk, $(wildcard examples/*))
#include $(example_includes)
# include all directories in projects
#project_includes := $(addsuffix /projects.mk, $(wildcard $(proj_dir)/*))
#include $(project_includes)

# this blank 'default' is required
default:

.PHONY: default clean distclean
default: $(default_targets)

examples: $(carlsim_programs)

pti-examples: $(pti_programs)

# TODO: eventually need to remove this
#clean:
#	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(output_files) $(GTEST_LIB_DIR)
# TODO: include all the correct stuff in this one too
# TODO: need to create a results directory in the main directory and delete later with
# every carlsim_tests run. distclean/clean should take care of it.
clean:
	$(RM) $(objects)
# TODO: see what distclean should really do by convention
distclean:
	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(libraries) $(output_files)

devtest:
	@echo $(CARLSIM_SRC_DIR) $(carlsim_tests_objs)

# TODO: rewrite help instructions
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
