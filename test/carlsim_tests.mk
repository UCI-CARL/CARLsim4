# module include file for CARLsim pti 

objects += $(test_dir)/carlsim_tests.o
output_files += $(test_dir)/carlsim_tests

CARLSIM_TEST_FLAGS := -I$(CURDIR)/$(test_dir) -D__REGRESSION_TESTING__

gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a \
	$(GTEST_LIB_DIR)/libgtest_custom_main.a

local_dir := $(test_dir)
local_deps := carlsim_tests.h coba.cpp core.cpp interface.cpp spikeCounter.cpp \
	stdp.cpp stp.cpp spike_info_tests.cpp
local_src := $(addprefix $(local_dir)/,$(local_deps))
local_objs := $(addprefix $(local_dir)/,coba.o core.o spikeCounter.o \
	stdp.o stp.o spike_info_tests.o)

carlsim_tests_objs := $(local_objs)
objects += $(carlsim_tests_objs)


.PHONY: carlsim_tests
carlsim_tests: $(test_dir)/carlsim_tests $(local_objs)

$(local_dir)/carlsim_tests: $(local_objs) $(gtest_deps) \
	$(carlsim_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(CARLSIM_TEST_FLAGS) $(spike_info_flags) \
	$(carlsim_objs) $(GTEST_CPPFLAGS) -L$(GTEST_LIB_DIR) -lgtest_custom_main \
	$(carlsim_tests_objs) -o $@

$(local_dir)/%.o: $(local_dir)/%.cpp $(local_deps)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(CARLSIM_TEST_FLAGS) $(spike_info_flags) \
	$(GTEST_CPPFLAGS) -L$(GTEST_LIB_DIR) -lgtest_custom_main \
	-c $< -o $@

# rule for our local custom gtest main
$(GTEST_LIB_DIR)/libgtest_custom_main.a: $(GTEST_LIB_DIR)/gtest-all.o \
	$(GTEST_LIB_DIR)/gtest_custom_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/gtest_custom_main.o: gtest_custom_main.cpp $(GTEST_SRCS_)
	mkdir -p $(GTEST_LIB_DIR)
	$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
	$(test_dir)/gtest_custom_main.cpp -o $@
