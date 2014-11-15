# Makefile for building example program from the CARLsim library

# NOTE: if you are compiling your code in a directory different from
# examples/<example_name> or projects/<project_name> then you need to either
# move the configured user.mk file to this directory or set the path to
# where CARLsim can find the user.mk.
USER_MK_PATH = ../../
include $(USER_MK_PATH)user.mk

example := spikeInfo
output := *.dot *.dat *.log *.csv results/*

# You should not need to edit the file beyond this point
# ------------------------------------------------------

# carlsim components
kernel_dir     = $(CARLSIM_LIB_DIR)/kernel
interface_dir  = $(CARLSIM_LIB_DIR)/interface
spike_mon_dir  = $(CARLSIM_LIB_DIR)/spike_monitor
spike_gen_dir  = $(CARLSIM_LIB_DIR)/spike_generators
# we are compiling from lib
CARLSIM_FLAGS += -I$(CARLSIM_LIB_DIR)/include/kernel \
								 -I$(CARLSIM_LIB_DIR)/include/interface \
								 -I$(CARLSIM_LIB_DIR)/include/spike_monitor
CARLSIM_LIBS  += -L$(CARLSIM_LIB_DIR)/lib -lCARLsim

local_src  := main_$(example).cpp
local_prog := $(example)

.PHONY: clean distclean
# compile from CARLsim lib
$(local_prog): $(local_src)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) \
	$(CARLSIM_LIBS) $< -o $@

clean:
	$(RM) $(local_prog)

distclean:
	$(RM) $(local_prog) *.log *.dat

