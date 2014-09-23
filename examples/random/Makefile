# Makefile for building example program from the CARLsim source or library
example := random
output := *.dot *.dat *.log



# set absolute path to configured user.mk file or just copy it to this
# directory and leave USER_MK_PATH blank.
USER_MK_PATH = /home/kris/Project/CARLsim/
include $(USER_MK_PATH)user.mk

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
									 -I$(CARLSIM_LIB_INSTALL_DIR)/include/spike_monitor \
									 $(CARLSIM_LIB_INSTALL_DIR)/lib/libCARLsim.a


# TODO: make it so that this doesn't have to be included in the directory.
# It should be included relative to the library install path or something.
# It could also be rolled into user.mk.
# TODO: this should be commented out, we don't want to have to use this
# include $(CARLSIM_SRC_DIR)/carlsim/carlsim.mk
# local info (vars can be overwritten)
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

# TODO: distclean
distclean:
	$(RM) $(output_files) *.log *.dat results/*.dat

devtest:
	@echo $(output_files)
