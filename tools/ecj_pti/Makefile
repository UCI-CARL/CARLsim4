# Makefile to build CARLsim-ECJ PTI Examples from installed CARLsim Library
# (libCARLsim.a). This file includes $(ex_dir)/exampmles.mk which builds the
# PTI exmaple programs from libCARLsim.a.

# set this to wherever user.mk is located
USER_MK_PATH ?= ../../
include $(USER_MK_PATH)/user.mk

# -----------------------------------------------------------------------------
# You should not need to edit the file beyond this point
# -----------------------------------------------------------------------------

all_targets :=
install_targets :=
pti_programs :=
pti_deps :=
pti_objs :=
output_files :=
objects :=
libraries :=
izk_build_files :=
pti_dir := src/CARLsim-app/pti
ex_dir := src/CARLsim-app/Examples
iz_dir := src/izk

# location of .cpp files
vpath %.cpp $(pti_dir)
# location of .h files
vpath %.h $(pti_dir)

# carlsim components
kernel_dir     = $(CARLSIM_LIB_DIR)/kernel
interface_dir  = $(CARLSIM_LIB_DIR)/interface
conn_mon_dir   = $(CARLSIM_LIB_DIR)/connection_monitor
spike_mon_dir  = $(CARLSIM_LIB_DIR)/spike_monitor
spike_gen_dir  = $(CARLSIM_LIB_DIR)/spike_generators

# we are compiling from lib
CARLSIM_FLAGS += -I$(CARLSIM_LIB_DIR)/include/kernel \
								 -I$(CARLSIM_LIB_DIR)/include/interface \
								 -I$(CARLSIM_LIB_DIR)/include/spike_monitor \
								 -I$(CARLSIM_LIB_DIR)/include/connection_monitor

CARLSIM_LIBS  += -L$(CARLSIM_LIB_DIR)/lib -lCARLsim

all:

include ecj_pti.mk
include $(pti_dir)/pti.mk
include $(pti_dir)/libCARLsimPTI.mk
include $(ex_dir)/examples.mk
include $(iz_dir)/izk.mk

all: $(all_targets)

install: $(install_targets)

clean:
	$(RM) $(objects) $(libraries) $(pti_programs) $(izk_build_files)

distclean:
	$(RM) $(objects) $(libraries) $(pti_programs) $(output_files) $(izk_build_files)

