# Makefile that includes modules for organization
#-------------------------------------------------------------------------------
# Begin user modifiable section
#-------------------------------------------------------------------------------
# absolute path of evolving objects installation 
EO_INSTALL_DIR ?= /opt/eo

# desired installation absolute path of pti 
PTI_INSTALL_DIR ?= /opt/pti

# if optional env vars do not exist, assign default values
# $(OPT_LEVEL): set to 1, 2, or 3 if you want to use optimization.  Default: 0.
# $(DEBUG_INFO): set to 1 to include debug info, set to 0 to not include 
# debugging info.  Default: 0.
CARLSIM_CUDAVER ?= 3
CARLSIM_FASTMATH ?= 0
CARLSIM_CUOPTLEVEL ?= 0
CARLSIM_DEBUG ?= 0

inc_dir = ../../include
src_dir = ../../src
lib_dir = ../../libpti
ex_dir  = ../../examples
interface_dir = ../../interface
test_core_dir = core

#-------------------------------------------------------------------------------
# End user modifiable section
#-------------------------------------------------------------------------------

# these variables collect the information from the other modules
all_targets :=
carlsim_programs  := 
pti_programs :=
sources   :=
libraries := 
carlsim_deps :=
carlsim_sources :=
carlsim_objs := 
common_sources :=
common_objs :=
output_files :=
objects :=
test_objects :=


# location of .cpp files
vpath %.cpp $(EO_INSTALL_DIR)/src $(EO_INSTALL_DIR)/src/do \
	$(EO_INSTALL_DIR)/src/es $(EO_INSTALL_DIR)/src/utils $(lib_dir) \
	$(ex_dir)/common/ $(src_dir) $(interface_dir)/src
# location of .cu files
vpath %.cu $(src_dir)
# location of .h files
vpath %.h $(EO_INSTALL_DIR)/src $(inc_dir) $(src_dir) $(ex_dir)/common \
	$(interface_dir)/include


all:
# include rules for building samples in a separate include file
# core makefile includes
include ../makefile.mk
include ../libpti/libpti.mk
include ../src/carlsim.mk
include makefile.gtest.mk
include core/carlsim_tests.mk

# House-keeping build targets.
all : gtest sample1_unittest


clean :
	rm -rf $(tests) *.o $(GTEST_LIB_DIR) sample1_unittest