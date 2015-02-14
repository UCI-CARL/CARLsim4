# Makefile for building Tutorial code from the CARLsim library

# NOTE: need to tell CARLsim where to find the user.mk
USER_MK_PATH = ../../../../
include $(USER_MK_PATH)user.mk

# list all targets in this directory
local_targets := basic_concepts
output := *.dot *.dat *.log *.csv results/*

# we are compiling from carlsim lib
CARLSIM_FLAGS += -I$(CARLSIM_LIB_DIR)/include/kernel \
				 -I$(CARLSIM_LIB_DIR)/include/interface \
				 -I$(CARLSIM_LIB_DIR)/include/spike_monitor \
				 -I$(CARLSIM_LIB_DIR)/include/group_monitor \
				 -I$(CARLSIM_LIB_DIR)/include/connection_monitor \
				 -I$(CARLSIM_LIB_DIR)/include/spike_generators \
				 -I$(CARLSIM_LIB_DIR)/include/visual_stimulus \
				 -I$(CARLSIM_LIB_DIR)/include/simple_weight_tuner
CARLSIM_LIBS  += -L$(CARLSIM_LIB_DIR)/lib -lCARLsim

.PHONY: all clean distclean
all: $(local_targets)
# compile from CARLsim lib

# list all rules, one per target
basic_concepts: main_basic_concepts.cpp
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) $< -o $@ $(CARLSIM_LIBS)

clean:
	$(RM) $(local_targets)

distclean:
	$(RM) $(local_targets) $(output)
