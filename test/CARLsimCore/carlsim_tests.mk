# module include file for CARLsim pti 

#pti_src = ../../src/CARLsim-app/pti
carlsim_src = ../../src
interface_dir = ../../interface/include

NVCC = nvcc
NVCC_STUFF = -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib \
				-lcutil_x86_64 -D__CUDA3__ -arch sm_13

gtest_deps = $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a

carlsim_tests: carlsim_tests.o $(gtest_deps) 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main \
			$^ -o $@

carlsim_tests.o: carlsim_tests.cpp 
	$(NVCC) $(CPPFLAGS) -I$(carlsim_src) -I$(interface_dir) $(NVCC_STUFF) \
			-c carlsim_tests.cpp
