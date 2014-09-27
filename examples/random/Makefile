# Makefile for building example program from the CARLsim source or library

# NOTE: if you are compiling your code in a directory different from
# examples/<example_name> or projects/<project_name> then you need to either
# move the configured user.mk file to this directory or set the path to
# where CARLsim can find the user.mk.
USER_MK_PATH = ../../
include $(USER_MK_PATH)user.mk

example := random
output := *.dot *.dat *.log *.csv

# You should not need to edit the file beyond this point
# ------------------------------------------------------

# carlsim components
	kernel_dir     = $(CARLSIM_LIB_INSTALL_DIR)/kernel
	interface_dir  = $(CARLSIM_LIB_INSTALL_DIR)/interface
	spike_mon_dir  = $(CARLSIM_LIB_INSTALL_DIR)/spike_monitor
	spike_gen_dir  = $(CARLSIM_LIB_INSTALL_DIR)/spike_generators
# TODO: add stuff so that we will install the correct headers for the server,
# and input_stimulus
# we are compiling from lib
	CARLSIM_FLAGS += -I$(CARLSIM_LIB_INSTALL_DIR)/include/kernel \
									 -I$(CARLSIM_LIB_INSTALL_DIR)/include/interface \
									 -I$(CARLSIM_LIB_INSTALL_DIR)/include/spike_monitor
	CARLSIM_LIBS  += $(CARLSIM_LIB_INSTALL_DIR)/lib/libCARLsim.a

local_src  := main_$(example).cpp
local_prog := $(example)
output_files += $(local_prog)

.PHONY: clean distclean
# compile from CARLsim lib
$(local_prog): $(local_src)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) \
	$(CARLSIM_LIBS) $< -o $@

clean:
	$(RM) $(output_files)

distclean:
	$(RM) $(output_files) *.log *.dat results/*.dat results/*.csv

devtest:
	@echo $(output_files)
