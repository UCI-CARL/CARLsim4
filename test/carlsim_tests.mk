# module include file for CARLsim pti 

gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a \
	$(GTEST_LIB_DIR)/libgtest_custom_main.a

# list of all test cpp files, but without directory and file extension
# e.g., file "test/coba.cpp" should appear here as "coba"
# the prefix (directory "test") and suffix (".cpp") will be appended afterwards
#carlsim_tests_cpps := coba core interface stdp stp spike_mon
carlsim_tests_cpps := stp

local_dir := $(test_dir)
local_deps := carlsim_tests.h $(addsuffix .cpp,$(carlsim_tests_cpps))
local_src := $(addprefix $(local_dir)/,$(local_deps))
local_objs := $(addsuffix .o,$(addprefix $(local_dir)/,$(carlsim_tests_cpps)))

carlsim_tests_objs := $(local_objs)
objects += $(carlsim_tests_objs)
output_files += $(test_dir)/carlsim_tests


.PHONY: carlsim_tests
carlsim_tests: $(test_dir)/carlsim_tests $(local_objs)

$(local_dir)/carlsim_tests: $(local_objs) $(gtest_deps) \
	$(carlsim_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(carlsim_objs) $(spike_monitor_flags) \
	$(GTEST_CPPFLAGS) -L$(GTEST_LIB_DIR) -lgtest_custom_main \
	$(carlsim_tests_objs) -o $@

$(local_dir)/%.o: $(local_dir)/%.cpp $(local_deps)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(spike_monitor_flags) \
	$(GTEST_CPPFLAGS) -c $< -o $@

# rule for our local custom gtest main
$(GTEST_LIB_DIR)/libgtest_custom_main.a: $(GTEST_LIB_DIR)/gtest-all.o \
	$(GTEST_LIB_DIR)/gtest_custom_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/gtest_custom_main.o: gtest_custom_main.cpp $(GTEST_SRCS_)
	mkdir -p $(GTEST_LIB_DIR)
	$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
	$(test_dir)/gtest_custom_main.cpp -o $@
