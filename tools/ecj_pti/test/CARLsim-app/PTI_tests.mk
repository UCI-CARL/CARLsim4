# module include file for CARLsim pti

pti_cpp := $(wildcard $(pti_dir)/*.cpp)
pti_objs :=$(patsubst %.cpp,%.o,$(pti_cpp))
output_files += $(patsubst %.o,%.gcda,$(pti_objs))
output_files += $(patsubst %.o,%.gcno,$(pti_objs))


gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a \
	$(GTEST_LIB_DIR)/libgtest_custom_main.a
local_objs := LoggerTest.o ParameterInstancesTest.o PTITest.o

pti_test: $(local_objs) $(pti_objs) $(gtest_deps)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) \
	-lgtest_custom_main $(pti_objs) $(local_objs) -o $@

# recipe to build individual test component .cpp files
%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(pti_dir) -c $< -o $@

$(GTEST_LIB_DIR)/libgtest_custom_main.a: $(GTEST_LIB_DIR)/gtest-all.o \
	$(GTEST_LIB_DIR)/gtest_custom_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/gtest_custom_main.o: gtest_custom_main.cpp $(GTEST_SRCS_)
	mkdir -p $(GTEST_LIB_DIR)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	gtest_custom_main.cpp -o $@
