##----------------------------------------------------------------------------##
##
##   CARLsim4 Project Makefile
##   -------------------------
##
##   Authors:   Michael Beyeler <mbeyeler@uci.edu>
##              Kristofor Carlson <kdcarlso@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   03/31/2016
##
##----------------------------------------------------------------------------##


#------------------------------------------------------------------------------
# Include configuration file
#------------------------------------------------------------------------------

# NOTE: If your CARLsim4 installation does not reside in the default path, make
# sure the environment variable CARLSIM4_INSTALL_DIR is set.
ifneq ($(CARLSIM4_INSTALL_DIR),)
	CARLSIM4_INC_DIR  := $(CARLSIM4_INSTALL_DIR)/inc
else
	CARLSIM4_INC_DIR  := /usr/local/include/carlsim
endif

# include compile flags etc.
include $(CARLSIM4_INC_DIR)/configure.mk

# main target for test suite
proj_target := carlsim_test
main_src_file := gtest_custom_main.cpp

.PHONY: project $(proj_target) clean distclean help gtest
default: $(proj_target)


#------------------------------------------------------------------------------
# Build local variables
#------------------------------------------------------------------------------

# include the make recipe to build the google testing framework
include gtest.mk
gtest_deps := $(addprefix $(GTEST_LIB_DIR)/,libgtest.a libgtest_main.a libgtest_custom_main.a)

cpp_files  := $(wildcard *.cpp)
cpp_files  := $(filter-out $(main_src_file),$(cpp_files))
cu_files   := $(wildcard *.cu)
inc_files  := $(wildcard *.h)

# compile .cpp files to -cpp.o, and .cu files to -cu.o
obj_files  := $(patsubst %.cpp, %-cpp.o, $(cpp_files))
obj_files  += $(patsubst %.cu, %-cu.o, $(cu_files))

# handled by clean and distclean
clean_files := $(obj_files) $(proj_target)
distclean_files := $(clean_files) results/* *.dot *.dat *.csv *.log

GTEST_LD := -L$(GTEST_LIB_DIR) -lgtest_custom_main
GTEST_CPPFLAGS += -isystem $(GTEST_DIR)/include -DGTEST_LINKED_AS_SHARED_LIBRARY=1

#------------------------------------------------------------------------------
# Project targets and rules
#------------------------------------------------------------------------------

$(proj_target): $(main_src_file) $(inc_files) $(cpp_files) $(obj_files) $(gtest_deps)
	$(NVCC) $(CARLSIM4_INC) $(GTEST_CPPFLAGS) $(CARLSIM4_LD) $(GTEST_LD) $(obj_files) $< -o $@

%-cpp.o: %.cpp $(cpp_files)
	$(NVCC) -c $(CARLSIM4_INC) $(GTEST_CPPFLAGS) $< -o $@
	# $(CXX) -c $(CXXINCFL) $(CXXFL) $(GTEST_CPPFLAGS) $< -o $@

%-cu.o: %.cu $(cu_files)
	$(NVCC) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $(GTEST_CPPFLAGS) $< -o $@

# rule for our local custom gtest main
$(GTEST_LIB_DIR)/libgtest_custom_main.a: $(GTEST_LIB_DIR)/gtest-all.o \
	$(GTEST_LIB_DIR)/gtest_custom_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/gtest_custom_main.o: gtest_custom_main.cpp
	@mkdir -p $(GTEST_LIB_DIR)
	@$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
	gtest_custom_main.cpp -o $@

clean:
	$(RM) $(clean_files)

distclean:
	$(RM) $(distclean_files)

help:
	$(info CARLsim4 Makefile options:)
	$(info )
	$(info make               Compiles CARLsim4 in default mode (release))
	$(info make clean         Cleans out all object files)
	$(info make distclean     Cleans out all object and output files)
	$(info make help          Brings up this message)
