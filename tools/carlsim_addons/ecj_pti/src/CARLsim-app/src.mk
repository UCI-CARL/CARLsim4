# Makefile to build CARLsim-PTI component
# This Makefile will build our CARLSIM-PTI binary from CARLsim source.

# set this to wherever user.mk is located
CARLSIM_SRC_DIR ?= /home/kris/Project/CARLsim
USER_MK_PATH ?= $(CARLSIM_SRC_DIR)
include $(USER_MK_PATH)/user.mk

# -----------------------------------------------------------------------------
# You should not need to edit the file beyond this point
# -----------------------------------------------------------------------------

all_targets :=
pti_programs :=
pti_deps :=
pti_objs :=
output_files :=
objects :=
libraries :=
izk_build_files :=
pti_dir = pti

# location of .cpp files
vpath %.cpp $(pti_dir)
# location of .h files
vpath %.h $(pti_dir)

# carlsim components
	kernel_dir     = $(CARLSIM_SRC_DIR)/kernel
	interface_dir  = $(CARLSIM_SRC_DIR)/interface
	spike_mon_dir  = $(CARLSIM_SRC_DIR)/spike_monitor
	spike_gen_dir  = $(CARLSIM_SRC_DIR)/spike_generators
# we are compiling from lib
	CARLSIM_FLAGS += -I$(CARLSIM_SRC_DIR)/include/kernel \
									 -I$(CARLSIM_SRC_DIR)/include/interface \
									 -I$(CARLSIM_SRC_DIR)/include/spike_monitor
	CARLSIM_LIBS  += $(CARLSIM_SRC_DIR)/lib/libCARLsim.a


all:

include pti/pti.mk
include Examples/src.mk
include ../izk/izk.mk

all: $(all_targets)

clean:
	$(RM) $(objects) $(libraries) $(pti_programs) $(izk_build_files)

distclean:
	$(RM) $(objects) $(libraries) $(pti_programs) $(output_files) $(izk_build_files)

