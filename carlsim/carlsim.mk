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

# visual stimulus
vs_dir          := $(tools_dir)/visual_stimulus
vs_inc_files    := $(wildcard $(vs_dir)/*h)
vs_cpp_files    := $(wildcard $(vs_dir)/*.cpp)
vs_cpp_obj      := $(patsubst %.cpp, %.o, $(vs_cpp_files))
tools_cpp_obj    += $(vs_cpp_obj)
SIMINCFL         += -I$(vs_dir)

# prepare clean-up
output += *.gcda *.gcno *.gcov coverage.info
output += $(addprefix $(intf_dir)/*/,*.gcda *.gcno)
output += $(addprefix $(krnl_dir)/*/,*.gcda *.gcno)
output += $(addprefix $(krnl_dir)/*/gpu_module/,*.gcda *.gcno)
output += $(addprefix $(mon_dir)/,*.gcda *.gcno)
output += $(addprefix $(tools_dir)/*/,*.gcda *.gcno)


#------------------------------------------------------------------------------
# CARLsim4 Common
#------------------------------------------------------------------------------

output          += carlsim4
objects_cpp     += $(krnl_cpp_obj) $(intf_cpp_obj) $(mon_cpp_obj) $(tools_cpp_obj)
objects_cu      += $(krnl_cu_obj) $(intf_cu_obj) $(mon_cu_obj) $(tools_cu_obj)

# additional files that need to be installed
add_files       := $(addprefix carlsim/,configure.mk)


#------------------------------------------------------------------------------
# CARLsim4 Targets
#------------------------------------------------------------------------------

.PHONY: release release_nocuda release_coverage release_nocuda_coverage debug debug_nocuda archive archive_nocuda prepare prep_nocuda

prep_cuda:
	$(eval CARLSIM4_FLG += -Wno-deprecated-gpu-targets)
	$(eval CARLSIM4_LIB += -lcurand)

prep_nocuda:
	$(eval CXXFL += -D__NO_CUDA__)
	$(eval CARLSIM4_FLG += -D __NO_CUDA__)
	$(eval NVCCFL += --compiler-options "-O3 -ffast-math")
	$(eval NVCC := $(CXX))
	$(eval NVCCINCFL := $(CXXINCFL))
	$(eval NVCCLDFL := $(CXXLIBFL))
	$(eval NVCCFL := $(CXXFL))

prep_release:
	$(eval CXXFL += -O3 -ffast-math)
	$(eval NVCCFL += --compiler-options "-O3 -ffast-math")

prep_debug:
	$(eval CXXFL += -g -Wall -O0)
	$(eval NVCCFL += -g -G --compiler-options "-Wall -O0")

prep_coverage:
	$(eval CXXFL += -fprofile-arcs -ftest-coverage)
	$(eval CXXLIBFL += -lgcov)
	$(eval NVCCFL += --compiler-options "-fprofile-arcs -ftest-coverage")
	$(eval NVCCLDFL += -lgcov)
	$(eval CARLSIM4_FLG += -fprofile-arcs -ftest-coverage)
	$(eval CARLSIM4_LIB += -lgcov)


release: prep_release prep_cuda $(objects_cpp) $(objects_cu) archive
release_nocuda: prep_release prep_nocuda $(objects_cpp) archive_nocuda
release_coverage: prep_release prep_coverage prep_cuda $(objects_cpp) $(objects_cu) archive
release_nocuda_coverage: prep_release prep_coverage prep_nocuda $(objects_cpp) archive_nocuda


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
$(vs_dir)/%.o: $(vs_dir)/%.cpp $(vs_inc_files)
	$(CXX) $(CXXSHRFL) -c $(CXXINCFL) $(SIMINCFL) $(CXXFL) $< -o $@


# archive
archive: prep_cuda $(objects_cpp) $(objects_cu)
	ar rcs $(lib_name).$(lib_ver) $(objects_cpp) $(objects_cu)

archive_nocuda: prep_nocuda $(objects_cpp)
	ar rcs $(lib_name).$(lib_ver) $(objects_cpp)
