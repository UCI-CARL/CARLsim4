##----------------------------------------------------------------------------##
##
##   CARLsim5 Project Makefile
##   -------------------------
##
##   Authors:   Michael Beyeler <mbeyeler@uci.edu>
##              Kristofor Carlson <kdcarlso@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   03/31/2016
##
##----------------------------------------------------------------------------##

################################################################################
# Start of user-modifiable section
################################################################################

# In this section, specify all files that are part of the project.

# Name of the binary file to be created.
# NOTE: There must be a corresponding .cpp file named main_$(proj_target).cpp!
proj_target    := random_spnet

# Directory where all include files reside. The Makefile will automatically
# detect and include all .h files within that directory.
proj_inc_dir   := inc

# Directory where all source files reside. The Makefile will automatically
# detect and include all .cpp and .cu files within that directory.
proj_src_dir   := src

################################################################################
# End of user-modifiable section
################################################################################


#------------------------------------------------------------------------------
# Include configuration file
#------------------------------------------------------------------------------

# NOTE: If your CARLsim5 installation does not reside in the default path, make
# sure the environment variable CARLSIM5_INSTALL_DIR is set.
ifdef CARLSIM5_INSTALL_DIR
	CARLSIM5_INC_DIR  := $(CARLSIM5_INSTALL_DIR)/include
else
	CARLSIM5_INC_DIR  := $(HOME)/CARL/include
endif

# include compile flags etc.
include $(CARLSIM5_INC_DIR)/configure.mk


#------------------------------------------------------------------------------
# Build local variables
#------------------------------------------------------------------------------

main_src_file := $(proj_src_dir)/main_$(proj_target).cpp

# build list of all .cpp, .cu, and .h files (but don't include main_src_file)
cpp_files  := $(wildcard $(proj_src_dir)/*.cpp)
cpp_files  := $(filter-out $(main_src_file),$(cpp_files))
inc_files  := $(wildcard $(proj_inc_dir)/*.h)

# compile .cpp files to -cpp.o, and .cu files to -cu.o
obj_files  := $(patsubst %.cpp, %-cpp.o, $(cpp_files))

# handled by clean and distclean
clean_files := $(obj_files) $(proj_target)
distclean_files := $(clean_files) results/* *.dot *.dat *.csv *.log


#------------------------------------------------------------------------------
# Project targets and rules
#------------------------------------------------------------------------------

.PHONY: $(proj_target) nocuda clean distclean help
default: $(proj_target)


$(proj_target): $(main_src_file) $(inc_files) $(cpp_files) $(obj_files)
	$(eval CARLSIM5_FLG += -Wno-deprecated-gpu-targets)
	$(eval CARLSIM5_LIB += -lcurand)
	$(NVCC) $(CARLSIM5_FLG) $(obj_files) $< -o $(proj_target) $(CARLSIM5_LIB) 

nocuda: $(main_src_file) $(inc_files) $(obj_cpp)
	$(eval CARLSIM5_FLG += -D __NO_CUDA__)
	$(CXX) $(CARLSIM5_FLG) $(obj_cpp) $(obj_cu) $< -o $(proj_target) $(CARLSIM5_LIB)

$(proj_src_dir)/%-cpp.o: $(proj_src_dir)/%.cpp $(inc_files)
	$(CXX) -c $(CXXINCFL) $(CXXFL) $< -o $@

clean:
	$(RM) $(clean_files)

distclean:
	$(RM) $(distclean_files)

help:
	$(info CARLsim5 Project options:)
	$(info )
	$(info make               Compiles this project)
	$(info make no_cuda       Compiles this project without CUDA library)
	$(info make clean         Cleans out all object files)
	$(info make distclean     Cleans out all object and output files)
	$(info make help          Brings up this message)
