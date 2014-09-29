# module include file for CARLsim pti 

pti_src = ../../src/CARLsim-app/pti
gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a $(GTEST_LIB_DIR)/libgtest_custom_main.a
pti_deps = pti_config.h pti.h pti.cpp ptiImpl.h ptiImpl.cpp
pti_objs += $(addprefix $(pti_src)/,pti.o ptiImpl.o) 
local_objs := outputTest.o


pti_tests: $(local_objs) $(pti_objs) $(gtest_deps)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) \
	-lgtest_custom_main $(pti_objs) $(local_objs) -o $@

# recipes to build pti dependencies
$(pti_src)/pti.o: pti.cpp pti.h pti_config.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(pti_src) -c $< -o $@

$(pti_src)/ptiImpl.o: ptiImpl.cpp ptiImpl.h pti_config.h pti.cpp pti.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(pti_src) -c $< -o $@

# recipe to build individual test component .cpp files
%.o: %.cpp pti_tests.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(pti_src) -c $< -o $@

$(GTEST_LIB_DIR)/libgtest_custom_main.a: $(GTEST_LIB_DIR)/gtest-all.o $(GTEST_LIB_DIR)/gtest_custom_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/gtest_custom_main.o: gtest_custom_main.cpp $(GTEST_SRCS_)
	mkdir -p $(GTEST_LIB_DIR)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c gtest_custom_main.cpp -o $@
