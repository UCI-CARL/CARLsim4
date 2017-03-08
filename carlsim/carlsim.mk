##----------------------------------------------------------------------------##
##
##   CARLsim4 Kernel
##   ---------------
##
##   Authors:   Michael Beyeler <mbeyeler@uci.edu>
##              Kristofor Carlson <kdcarlso@uci.edu>
##              Ting-Shuo Chou <tingshuc@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   12/31/2016
##
##----------------------------------------------------------------------------##

#------------------------------------------------------------------------------
# CARLsim4 Interface
#------------------------------------------------------------------------------

intf_dir          := carlsim/interface
intf_inc_files    := $(wildcard $(intf_dir)/inc/*.h)
intf_cpp_files    := $(wildcard $(intf_dir)/src/*.cpp)
intf_cu_files     := $(wildcard $(intf_dir)/src/*.cu)
intf_cpp_obj      := $(patsubst %.cpp, %-cpp.o, $(intf_cpp_files))
intf_cu_obj       := $(patsubst %.cu, %-cu.o, $(intf_cu_files))
SIMINCFL          += -I$(intf_dir)/inc


#------------------------------------------------------------------------------
# CARLsim4 Kernel
#------------------------------------------------------------------------------

krnl_dir          := carlsim/kernel
krnl_inc_files    := $(wildcard $(krnl_dir)/inc/*.h)
krnl_cpp_files    := $(wildcard $(krnl_dir)/src/*.cpp)
krnl_cu_files     := $(wildcard $(krnl_dir)/src/gpu_module/*.cu)
krnl_cpp_obj      := $(patsubst %.cpp, %-cpp.o, $(krnl_cpp_files))
krnl_cu_obj       := $(patsubst %.cu, %-cu.o, $(krnl_cu_files))
SIMINCFL          += -I$(krnl_dir)/inc


#------------------------------------------------------------------------------
# CARLsim4 Utilities
#------------------------------------------------------------------------------

mon_dir          := carlsim/monitor
mon_inc_files    := $(wildcard $(mon_dir)/*.h)
mon_cpp_files    := $(wildcard $(mon_dir)/*.cpp)
mon_cu_files     := $(wildcard $(mon_dir)/src/*.cu)
mon_cpp_obj      := $(patsubst %.cpp, %-cpp.o, $(mon_cpp_files))
mon_cu_obj       := $(patsubst %.cu, %-cu.o, $(mon_cu_files))
SIMINCFL         += -I$(mon_dir)


#------------------------------------------------------------------------------
# CARLsim4 Tools
#------------------------------------------------------------------------------

tools_dir        := tools
tools_cpp_obj    :=
tools_cu_obj     :=

# simple weight tuner
swt_dir          := $(tools_dir)/simple_weight_tuner
swt_inc_files    := $(wildcard $(swt_dir)/*h)
swt_cpp_files    := $(wildcard $(swt_dir)/*.cpp)
swt_cu_files     := $(wildcard $(swt_dir)/*.cu)
swt_cpp_obj      := $(patsubst %.cpp, %.o, $(swt_cpp_files))
tools_cpp_obj    += $(swt_cpp_obj)
SIMINCFL         += -I$(swt_dir)

# spike generators
spkgen_dir       := $(tools_dir)/spike_generators
spkgen_inc_files := $(wildcard $(spkgen_dir)/*.h)
spkgen_cpp_files := $(wildcard $(spkgen_dir)/*.cpp)
spkgen_cpp_obj   := $(patsubst %.cpp, %.o, $(spkgen_cpp_files))
tools_cpp_obj    += $(spkgen_cpp_obj)
SIMINCFL         += -I$(spkgen_dir)

# stopwatch
stp_dir          := $(tools_dir)/stopwatch
stp_inc_files    := $(wildcard $(stp_dir)/*h)
stp_cpp_files    := $(wildcard $(stp_dir)/*.cpp)
stp_cpp_obj      := $(patsubst %.cpp, %.o, $(stp_cpp_files))
tools_cpp_obj    += $(stp_cpp_obj)
SIMINCFL         += -I$(stp_dir)


#------------------------------------------------------------------------------
# CARLsim4 Common
#------------------------------------------------------------------------------

output          += carlsim4
objects_cpp     += $(krnl_cpp_obj) $(intf_cpp_obj) $(mon_cpp_obj) $(tools_cpp_obj)
objects_cu      += $(krnl_cu_obj) $(intf_cu_obj) $(mon_cu_obj) $(tools_cu_obj)

ifeq ($(CARLSIM4_COVERAGE),1)
output          += $(addprefix $(intf_dir)/*/,*.gcda *.gcno)
output          += $(addprefix $(krnl_dir)/*/,*.gcda *.gcno)
output          += $(addprefix $(mon_dir)/,*.gcda *.gcno)
output          += $(addprefix $(tools_dir)/*/,*.gcda *.gcno)
endif

# additional files that need to be installed
add_files       := $(addprefix carlsim/,configure.mk)


#------------------------------------------------------------------------------
# CARLsim4 Targets
#------------------------------------------------------------------------------

.PHONY: prepare_cuda prepare_nocuda release debug archive archive_nocuda

# These have to be replicated for every user-called target.
# In CUDA mode:
prepare_cuda: CXXFL  += -O3 -ffast-math
prepare_cuda: NVCCFL += --compiler-options "-O3 -ffast-math"
ifeq ($(CARLSIM4_COVERAGE),1)
preapre_cuda: CARLSIM4_FLG += --compiler-options "-fprofile-arcs -ftest-coverage"
prepare_cuda: CARLSIM4_LIB += -lgcov
endif
prepare_cuda: CARLSIM4_FLG += -Wno-deprecated-gpu-targets
prepare_cuda: CARLSIM4_LIB += -lcurand

# These have to be replicated for every user-called target.
# In CUDA mode:
prepare_nocuda: CXXFL  += -D__NO_CUDA__
prepare_nocuda: CARLSIM4_FLG += -D __NO_CUDA__
prepare_nocuda: NVCCFL += --compiler-options "-O3 -ffast-math"
ifeq ($(CARLSIM4_COVERAGE),1)
prepare_nocuda: CARLSIM4_FLG += -fprofile-arcs -ftest-coverage
prepare_nocuda: CARLSIM4_LIB += -lgcov
endif
prepare_nocuda: NVCC := $(CXX)
prepare_nocuda: NVCCINCFL := $(CXXINCFL)
prepare_nocuda: NVCCLDFL := $(CXXLIBFL)
prepare_nocuda: NVCCFL := $(CXXFL)


# Actual release targets
release: prepare_cuda
release: $(objects_cpp) $(objects_cu)
release: archive

release_nocuda: prepare_nocuda
release_nocuda: $(objects_cpp)
release_nocuda: archive_nocuda


# Actual debug targets

# debug: CXXFL    += -g -Wall -O0
# ifeq ($(CARLSIM4_NO_CUDA),1)
# debug: NVCCFL   += -g -Wall -O0
# else
# debug: NVCCFL   += -g -G --compiler-options "-Wall -O0"
# endif
# debug: $(objects)
# debug: archive



#------------------------------------------------------------------------------
# CARLsim4 Rules
#------------------------------------------------------------------------------

# rule to compile local cpps
$(intf_dir)/src/%-cpp.o: $(intf_dir)/src/%.cpp $(intf_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(intf_dir)/src/%-cu.o: $(intf_dir)/src/%.cu $(intf_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(krnl_dir)/src/%-cpp.o: $(krnl_dir)/src/%.cpp $(krnl_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(krnl_dir)/src/gpu_module/%-cu.o: $(krnl_dir)/src/gpu_module/%.cu $(krnl_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@

# utilities
$(mon_dir)/%-cpp.o: $(mon_dir)/%.cpp $(mon_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@

# tools
$(swt_dir)/%.o: $(swt_dir)/%.cpp $(swt_inc_files)
	$(CXX) $(CXXSHRFL) -c $(CXXINCFL) $(SIMINCFL) $(CXXFL) $< -o $@
$(spkgen_dir)/%.o: $(spkgen_dir)/%.cpp $(spkgen_inc_files)
	$(CXX) $(CXXSHRFL) -c $(CXXINCFL) $(SIMINCFL) $(CXXFL) $< -o $@
$(stp_dir)/%.o: $(stp_dir)/%.cpp $(stp_inc_files)
	$(CXX) $(CXXSHRFL) -c $(CXXINCFL) $(SIMINCFL) $(CXXFL) $< -o $@

# archive
archive: $(objects_cpp) $(objects_cu) prepare_cuda
	ar rcs $(lib_name).$(lib_ver) $(objects_cpp) $(objects_cu)

archive_nocuda: $(objects_cpp) prepare_nocuda
	ar rcs $(lib_name).$(lib_ver) $(objects_cpp)
