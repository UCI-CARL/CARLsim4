# module include file for building example/project from source
# mainly used by CARLsim devs
example := v1MTLIP
output := *.dot *.txt *.log *.csv results/*

# this must be included first because CARLSIM_SRC_DIR is set here
# path of CARLsim root src folder (for compiling from source)
CARLSIM_SRC_DIR ?= $(HOME)/Project/CARLsim
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

# local info (vars can be overwritten)
local_src := main_$(example).cpp
local_prog := $(example)
local_objs:= v1ColorME.2.1.o

# info passed up to Makefile
objects += $(local_objs)

.PHONY: default clean distclean
default: $(local_prog)
# this must come after CARLSIM_FLAGS have been set
include $(CARLSIM_SRC_DIR)/carlsim/carlsim.mk

$(local_prog): $(local_src) $(carlsim_inc) $(carlsim_sources) $(carlsim_objs) \
	$(local_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) \
	$(CARLSIM_LIBS) $(local_objs) $(carlsim_objs) $< -o $@

%.o: %.cu
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

clean:
	$(RM) $(objects) $(local_prog)

distclean:
	$(RM) $(objects) $(local_prog) $(output)
