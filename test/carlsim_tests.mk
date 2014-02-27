# module include file for CARLsim pti 

objects += $(test_dir)/carlsim_tests.o
output_files += $(test_dir)/carlsim_tests

gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a

carlsim_tests: $(test_dir)/carlsim_tests.o $(gtest_deps)
	$(CXX) $(GTEST_CPPFLAGS) $(GTEST_CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $(test_dir)/$@


$(test_dir)/carlsim_tests.o: $(test_dir)/carlsim_tests.cpp
	$(CXX) $(GTEST_CPPFLAGS) $(GTEST_CXXFLAGS) -c $(test_dir)/carlsim_tests.cpp -o $@