##----------------------------------------------------------------------------##
##
##   CARLsim4 Tests
##   --------------
##
##   Authors:   Michael Beyeler <mbeyeler@uci.edu>
##              Kristofor Carlson <kdcarlso@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   03/04/2017
##
##----------------------------------------------------------------------------##


#------------------------------------------------------------------------------
# CARLsim4 Test Files
#------------------------------------------------------------------------------

GTEST_DIR := external/googletest
GTEST_FLG := -I$(GTEST_DIR)/include -L$(GTEST_DIR)/build
GTEST_LIB := -lgtest
ifeq ($(CARLSIM4_NO_CUDA),1)
	GTEST_LIB += -pthread
endif

test_dir := carlsim/test
test_inc_files := $(wildcard $(test_dir)/*.h)
test_cpp_files := $(wildcard $(test_dir)/*.cpp)
test_target := $(test_dir)/carlsim_tests
output += $(test_target)


#------------------------------------------------------------------------------
# CARLsim4 Targets and Rules
#------------------------------------------------------------------------------

.PHONY: test test_no_cuda $(test_target)

test: $(test_target)

$(test_target): $(test_cpp_files) $(test_inc_files)
	$(NVCC) $(CARLSIM4_FLG) $(GTEST_FLG) $(GTEST_LD) $(test_cpp_files) -o $@ $(GTEST_LIB) $(CARLSIM4_LIB)


