# Makefile for building test suite from the CARLsim library
# point to the user.mk file
USER_MK_PATH = ../../
include $(USER_MK_PATH)user.mk

# list of all test cpp files, but without directory and file extension (stem only)
# e.g., file "test/coba.cpp" should appear here as "coba"
# the prefix (directory "test") and suffix (".cpp") will be appended afterwards
# test cases might be run in reverse order (depends on compiler)
carlsim_tests_cpps := stdp stp cuba coba interface spike_mon conn_mon group_mon \
	core connect carlsim_tests_common spike_gen spike_counter poiss_rate

local_src := $(addsuffix .cpp,$(carlsim_tests_cpps))
local_objs := $(addsuffix .o,$(carlsim_tests_cpps))
local_deps := $(local_src) carlsim_tests.h
local_prog := carlsim_tests

output := *.dot *.dat *.log *.csv results/* lib/*

# -----------------------------------------------------------------------------
# You should not need to edit the file beyond this point
# -----------------------------------------------------------------------------

# we are compiling from lib
CARLSIM_FLAGS += -I$(CARLSIM_LIB_DIR)/include/kernel \
				 -I$(CARLSIM_LIB_DIR)/include/interface \
				 -I$(CARLSIM_LIB_DIR)/include/spike_monitor \
				 -I$(CARLSIM_LIB_DIR)/include/group_monitor \
				 -I$(CARLSIM_LIB_DIR)/include/connection_monitor \
				 -I$(CARLSIM_LIB_DIR)/include/spike_generators \
				 -I$(CARLSIM_LIB_DIR)/include/visual_stimulus \
				 -I$(CARLSIM_LIB_DIR)/include/simple_weight_tuner
CARLSIM_LIBS  += -L$(CARLSIM_LIB_DIR)/lib -lCARLsim

.PHONY: default clean distclean
# this must come before including gtest.mk
default: $(local_prog)

# include the make recipe to build the google testing framework
include gtest.mk
gtest_deps := $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a \
	$(GTEST_LIB_DIR)/libgtest_custom_main.a

# compile from CARLsim lib
$(local_prog): $(local_src) $(local_objs) $(gtest_deps)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) $(local_objs) \
	$(GTEST_CPPFLAGS) $(CARLSIM_LIBS) -Llib -lgtest_custom_main $* -o $@

%.o: %.cpp
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(GTEST_CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(local_prog) $(local_objs)

distclean:
	$(RM) $(local_prog) $(local_objs) $(output)

# rule for our local custom gtest main
$(GTEST_LIB_DIR)/libgtest_custom_main.a: $(GTEST_LIB_DIR)/gtest-all.o \
	$(GTEST_LIB_DIR)/gtest_custom_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/gtest_custom_main.o: gtest_custom_main.cpp
	@mkdir -p $(GTEST_LIB_DIR)
	@$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
	gtest_custom_main.cpp -o $@
