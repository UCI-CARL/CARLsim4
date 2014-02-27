# module include file for CARLsim pti 


gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a

carlsim_tests: carlsim_tests.o $(gtest_deps)
	$(CXX) $(GTEST_CPPFLAGS) $(GTEST_CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@


carlsim_tests.o: $(test_core_dir)/carlsim_tests.cpp
	$(CXX) $(GTEST_CPPFLAGS) $(GTEST_CXXFLAGS) -c $(test_core_dir)/carlsim_tests.cpp