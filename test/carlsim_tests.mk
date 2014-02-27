# module include file for CARLsim pti 

objects += $(test_dir)/carlsim_tests.o
output_files += $(test_dir)/carlsim_tests

miau = -I/home/mbeyeler/CARLsim/src -I/home/mbeyeler/CARLsim/interface/include -I/home/mbeyeler/CARLsim/test -I/home/mbeyeler/CARLsim/

gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a

carlsim_tests: $(test_dir)/carlsim_tests.o $(gtest_deps)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS)  $(carlsim_objs) \
	$(GTEST_CPPFLAGS) -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $(test_dir)/$@
#	$(CXX) $(GTEST_CPPFLAGS) $(GTEST_CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $(test_dir)/$@


$(test_dir)/carlsim_tests.o: $(test_dir)/carlsim_tests.cpp
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) $(CARLSIM_FLAGS) \
	$(GTEST_CPPFLAGS) -L$(GTEST_LIB_DIR) -lgtest_main \
	-c $(test_dir)/carlsim_tests.cpp -o $@
#	$(CXX) $(GTEST_CPPFLAGS) $(GTEST_CXXFLAGS) $(miau) -c $(test_dir)/carlsim_tests.cpp -o $@