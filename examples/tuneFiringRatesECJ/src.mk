# module include file for building example/project from source
# mainly used by CARLsim devs
example := tuneFiringRatesECJ

# name of ECJ parameter file.
ecj_param_file := $(example)Experiment.params
output := *.dot *.dat *.log *.stat results/*

# this must be included first because CARLSIM_SRC_DIR is set here
# path of CARLsim root src folder (for compiling from source)
CARLSIM_SRC_DIR ?= $(HOME)/CARLsim
# set this to wherever user.mk is located
USER_MK_PATH    ?= $(CARLSIM_SRC_DIR)

# -----------------------------------------------------------------------------
# You should not need to edit the file beyond this point
# -----------------------------------------------------------------------------

include $(USER_MK_PATH)/user.mk

# carlsim components
kernel_dir     = $(CARLSIM_SRC_DIR)/carlsim/kernel
interface_dir  = $(CARLSIM_SRC_DIR)/carlsim/interface
spike_mon_dir  = $(CARLSIM_SRC_DIR)/carlsim/spike_monitor
spike_gen_dir  = $(CARLSIM_SRC_DIR)/tools/spike_generators
server_dir     = $(CARLSIM_SRC_DIR)/carlsim/server
# carlsim tools
input_stim_dir = $(CARLSIM_SRC_DIR)/tools/input_stimulus

# we are compiling from lib
CARLSIM_FLAGS += -I$(kernel_dir)/include -I$(interface_dir)/include \
								 -I$(spike_mon_dir) -I$(spike_gen_dir) -I$(server_dir) \
								 -I$(input_stim_dir)

# carlsim ecj components
ECJ_PTI_FLAGS += -I$(ECJ_PTI_DIR)/include
ECJ_PTI_LIBS  += $(ECJ_PTI_DIR)/lib/libCARLsimPTI.a

# local info (vars can be overwritten)
local_src := main_$(example).cpp
local_prog := $(example)
carlsim_prog := carlsim_$(example)

.PHONY: clean distclean
# create executable bash script for user to run
$(local_prog): $(local_src) $(carlsim_prog)
	@echo "#!/bin/bash" > $(local_prog)
	@echo "java -cp \"$(ECJ_DIR)/ecj.jar:$(ECJ_PTI_DIR)/lib/CARLsim-ECJ.jar\" ecjapp.CARLsimEC ./$(ecj_param_file)" >> $(local_prog)
	@chmod u+x $(local_prog)
# this must come after CARLSIM_FLAGS have been set
include $(CARLSIM_SRC_DIR)/carlsim/carlsim.mk

# compile from CARLsim src
$(carlsim_prog): $(local_src) $(carlsim_inc) $(carlsim_sources) $(carlsim_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(ECJ_PTI_FLAGS) $(CARLSIM_FLAGS) \
		$(CARLSIM_LFLAGS) $(CARLSIM_LIBS) $(ECJ_PTI_LIBS) $(carlsim_objs) $< -o $@

clean:
	$(RM) $(objects) $(local_prog) $(carlsim_prog)

distclean:
	$(RM) $(objects) $(local_prog) $(carlsim_prog) $(output)

