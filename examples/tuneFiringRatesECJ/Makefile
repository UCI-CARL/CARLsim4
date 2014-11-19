# Makefile for building example program from the CARLsim library

# NOTE: if you are compiling your code in a directory different from
# examples/<example_name> or projects/<project_name> then you need to either
# move the configured user.mk file to this directory or set the path to
# where CARLsim can find the user.mk.
USER_MK_PATH = ../../
include $(USER_MK_PATH)user.mk

example := tuneFiringRatesECJ

# name of ECJ parameter file.
ecj_param_file := $(example)Experiment.params
output := *.dot *.dat *.log *.stat results/*

# -----------------------------------------------------------------------------
# You should not need to edit the file beyond this point
# -----------------------------------------------------------------------------

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

# carlsim ecj components
ECJ_PTI_FLAGS += -I$(ECJ_PTI_DIR)/include
ECJ_PTI_LIBS  += $(ECJ_PTI_DIR)/lib/libCARLsimPTI.a

local_src  := main_$(example).cpp
local_prog := $(example)
carlsim_prog := carlsim_$(example)

.PHONY: clean distclean
# create executable bash script for user to run
$(local_prog): $(local_src) $(carlsim_prog)
	@echo "#!/bin/bash" > $(local_prog)
	@echo "java -cp \"$(ECJ_DIR)/jar/ecj.22.jar:$(ECJ_PTI_DIR)/lib/CARLsim-ECJ.jar\" ecjapp.CARLsimEC ./$(ecj_param_file)" >> $(local_prog)
	@chmod u+x $(local_prog)

# compile from CARLsim lib
$(carlsim_prog): $(local_src)
	$(NVCC) $(CARLSIM_INCLUDES) $(ECJ_PTI_FLAGS) $(CARLSIM_FLAGS) \
	$(CARLSIM_LFLAGS) $(CARLSIM_LIBS) $(ECJ_PTI_LIBS) $< -o $@

clean:
	$(RM) $(local_prog) $(carlsim_prog)

distclean:
	$(RM) $(local_prog) $(carlsim_prog) $(output)
