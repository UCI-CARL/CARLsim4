# Makefile to build CARLsim-PTI component
# This Makefile will build our CARLSIM-PTI binary from CARLsim source.
# It calls the Examples/src.mk file to compile the CARLSIM-ECJ PTI examples.

# set this to wherever user.mk is located
CARLSIM_LIB_DIR_INSTALL ?= /home/kris/Project/CARLsim
USER_MK_PATH ?= $(CARLSIM_LIB_DIR_INSTALL)
include $(USER_MK_PATH)/user.mk

all_targets :=
pti_programs :=
pti_deps :=
pti_objs :=
output_files :=
objects :=
izk_build_files :=
pti_dir = pti

# location of .cpp files
vpath %.cpp $(pti_dir)
# location of .h files
vpath %.h $(pti_dir)

# carlsim components
	kernel_dir     = $(CARLSIM_LIB_DIR_INSTALL)/kernel
	interface_dir  = $(CARLSIM_LIB_DIR_INSTALL)/interface
	spike_mon_dir  = $(CARLSIM_LIB_DIR_INSTALL)/spike_monitor
	spike_gen_dir  = $(CARLSIM_LIB_DIR_INSTALL)/spike_generators
# we are compiling from lib
	CARLSIM_FLAGS += -I$(CARLSIM_LIB_DIR_INSTALL)/include/kernel \
									 -I$(CARLSIM_LIB_DIR_INSTALL)/include/interface \
									 -I$(CARLSIM_LIB_DIR_INSTALL)/include/spike_monitor
	CARLSIM_LIBS  += $(CARLSIM_LIB_DIR_INSTALL)/lib/libCARLsim.a

all:

include pti/pti.mk
include pti/libCARLsimPTI.mk
include Examples/examples.mk
include ../izk/izk.mk

all: $(all_targets)

clean:
	$(RM) $(objects) $(pti_programs) $(output_files) $(izk_build_files)

devTest:
	@echo $(all_targets)
