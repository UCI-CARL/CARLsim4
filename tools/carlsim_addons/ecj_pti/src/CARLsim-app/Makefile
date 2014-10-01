# Makefile to build CARLsim-PTI component

# You must first have CARLsim installed as a library to install this.

# set this to wherever user.mk is located
USER_MK_PATH ?= /home/kris/Project/CARLsim
include $(USER_MK_PATH)/user.mk

all_targets :=
all_projects :=
carlsim_programs  := 
pti_programs :=
pti_deps :=
pti_objs :=
sources   :=
libraries := 
carlsim_deps :=
carlsim_sources :=
carlsim_objs := 
common_sources :=
common_objs :=
output_files :=
objects :=
izk_build_files :=
pti_dir = pti

# location of .cpp files
vpath %.cpp $(pti_dir)
# location of .h files
vpath %.h $(pti_dir)

# carlsim components
	kernel_dir     = $(CARLSIM_LIB_INSTALL_DIR)/kernel
	interface_dir  = $(CARLSIM_LIB_INSTALL_DIR)/interface
	spike_mon_dir  = $(CARLSIM_LIB_INSTALL_DIR)/spike_monitor
	spike_gen_dir  = $(CARLSIM_LIB_INSTALL_DIR)/spike_generators
# we are compiling from lib
	CARLSIM_FLAGS += -I$(CARLSIM_LIB_INSTALL_DIR)/include/kernel \
									 -I$(CARLSIM_LIB_INSTALL_DIR)/include/interface \
									 -I$(CARLSIM_LIB_INSTALL_DIR)/include/spike_monitor
	CARLSIM_LIBS  += $(CARLSIM_LIB_INSTALL_DIR)/lib/libCARLsim.a


all:

include pti/pti.mk
include Examples/examples.mk
include ../izk/izk.mk

all: $(all_targets)

clean:
	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(output_files) $(izk_build_files)

