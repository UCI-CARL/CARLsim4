# Main Makefile for compiling, testing, and installing CARLsim
# these variables collect the information from the other modules
# TODO: decide whether or not to allow uses to make examples etc.
# TODO: Update User Guide on wiki with correct layout.
# TODO: add instructions about running examples and projects and that they go in same dir.results by default
# TODO: make issue saying to make motion energy a library and remove mtrand.h include dependency.
# TODO: make issue telling Ting to fix the visual studio stuff.
# TODO: make issue about creating tutorials out of examples and linking with doxygen.
# TODO: create Makefile for devs that want to build from src
# TODO: add these to make our documentation and then add an issue
#@ echo "make docs     Builds the class documentation, found in docs/classsdocs"
#@ echo "make doc        (Same thing)"

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

clean:
	$(RM) $(objects)

distclean:
	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(libraries) $(output_files) doc/html

devtest:
	@echo $(CARLSIM_SRC_DIR) $(carlsim_tests_objs)

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
