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

# whether to include flag for regression testing
CARLSIM_TEST ?= 0

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

inc_dir = include
carlsim_dir = carlsim
lib_dir = libpti
ex_dir  = examples
interface_dir = interface
test_dir = test
util_dir = util

# location of .cpp files
vpath %.cpp $(EO_INSTALL_DIR)/src $(EO_INSTALL_DIR)/src/do \
$(EO_INSTALL_DIR)/src/es $(EO_INSTALL_DIR)/src/utils $(lib_dir) \
$(ex_dir)/common/ $(carlsim_dir)/src $(interface_dir)/src $(test_dir)

# location of .cu files
vpath %.cu $(carlsim_dir)/src
# location of .h files
vpath %.h $(EO_INSTALL_DIR)/src $(inc_dir) $(carlsim_dir)/include \
	$(ex_dir)/common $(interface_dir)/include $(test_dir) \

# this blank 'all' is required
all:

# core makefile includes
include makefile.mk
include libpti/libpti.mk
include carlsim/carlsim.mk
include test/gtest.mk
include test/carlsim_tests.mk

# include all directories in examples
example_includes := $(addsuffix /examples.mk, $(wildcard examples/*))
include $(example_includes)

.PHONY: all libraries examples pti_examples clean distclean tests
all: $(all_targets)

tests: gtest carlsim_tests

libraries: $(libraries)

examples: $(carlsim_programs)

pti-examples: $(pti_programs)

clean:
	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(output_files) $(GTEST_LIB_DIR) \

distclean:
	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(libraries) $(output_files)

devtest:
	@echo $(all_targets)
