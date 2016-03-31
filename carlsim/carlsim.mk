##----------------------------------------------------------------------------##
##
##   CARLsim4 Kernel
##   ---------------
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

#------------------------------------------------------------------------------
# CARLsim4 Interface
#------------------------------------------------------------------------------

intf_dir        := carlsim/interface
intf_inc_files  := $(wildcard $(intf_dir)/inc/*.h)
intf_cpp_files  := $(wildcard $(intf_dir)/src/*.cpp)
intf_cu_files   := $(wildcard $(intf_dir)/src/*.cu)
intf_obj_files  := $(patsubst %.cpp, %-cpp.o, $(intf_cpp_files))
intf_obj_files  += $(patsubst %.cu, %-cu.o, $(intf_cu_files))
SIMINCFL    += -I$(intf_dir)/inc


#------------------------------------------------------------------------------
# CARLsim4 Kernel
#------------------------------------------------------------------------------

krnl_dir        := carlsim/kernel
krnl_inc_files  := $(wildcard $(krnl_dir)/inc/*.h)
krnl_cpp_files  := $(wildcard $(krnl_dir)/src/*.cpp)
krnl_cu_files   := $(wildcard $(krnl_dir)/src/*.cu)
krnl_obj_files  := $(patsubst %.cpp, %-cpp.o, $(krnl_cpp_files))
krnl_obj_files  += $(patsubst %.cu, %-cu.o, $(krnl_cu_files))
SIMINCFL    += -I$(krnl_dir)/inc


#------------------------------------------------------------------------------
# CARLsim4 Utilities
#------------------------------------------------------------------------------

conn_dir        := carlsim/connection_monitor
conn_inc_files  := $(wildcard $(conn_dir)/*.h)
conn_cpp_files  := $(wildcard $(conn_dir)/*.cpp)
conn_cu_files   := $(wildcard $(conn_dir)/src/*.cu)
conn_obj_files  := $(patsubst %.cpp, %-cpp.o, $(conn_cpp_files))
conn_obj_files  += $(patsubst %.cu, %-cu.o, $(conn_cu_files))
SIMINCFL    += -I$(conn_dir)

grps_dir        := carlsim/group_monitor
grps_inc_files  := $(wildcard $(grps_dir)/*.h)
grps_cpp_files  := $(wildcard $(grps_dir)/*.cpp)
grps_cu_files   := $(wildcard $(grps_dir)/src/*.cu)
grps_obj_files  := $(patsubst %.cpp, %-cpp.o, $(grps_cpp_files))
grps_obj_files  += $(patsubst %.cu, %-cu.o, $(grps_cu_files))
SIMINCFL    += -I$(grps_dir)

spks_dir        := carlsim/spike_monitor
spks_inc_files  := $(wildcard $(spks_dir)/*.h)
spks_cpp_files  := $(wildcard $(spks_dir)/*.cpp)
spks_cu_files   := $(wildcard $(spks_dir)/src/*.cu)
spks_obj_files  := $(patsubst %.cpp, %-cpp.o, $(spks_cpp_files))
spks_obj_files  += $(patsubst %.cu, %-cu.o, $(spks_cu_files))
SIMINCFL    += -I$(spks_dir)


#------------------------------------------------------------------------------
# CARLsim4 Common
#------------------------------------------------------------------------------

targets         += carlsim4
objects         += $(krnl_obj_files) $(intf_obj_files) $(conn_obj_files) $(grps_obj_files) $(spks_obj_files)
add_files       := $(addprefix carlsim/,configure.mk)


#------------------------------------------------------------------------------
# CARLsim4 Targets
#------------------------------------------------------------------------------

.PHONY: release debug carlsim4

# release build
release: CXXFL  += -O3 -ffast-math
release: NVCCFL += --compiler-options "-O3 -ffast-math"
release: $(targets)

# debug build
debug: CXXFL    += -g -Wall -O0
debug: NVCCFL   += -g -G --compiler-options "-O0"
debug: $(targets)

# all CARLsim4 targets
carlsim4: $(objects)


#------------------------------------------------------------------------------
# CARLsim4 Rules
#------------------------------------------------------------------------------

# rule to compile local cpps
$(intf_dir)/src/%-cpp.o: $(intf_dir)/src/%.cpp $(intf_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(intf_dir)/src/%-cu.o: $(intf_dir)/src/%.cu $(intf_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(krnl_dir)/src/%-cpp.o: $(krnl_dir)/src/%.cpp $(krnl_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(krnl_dir)/src/%-cu.o: $(krnl_dir)/src/%.cu $(krnl_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@

$(conn_dir)/%-cpp.o: $(conn_dir)/%.cpp $(conn_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(grps_dir)/%-cpp.o: $(grps_dir)/%.cpp $(grps_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
$(spks_dir)/%-cpp.o: $(spks_dir)/%.cpp $(spks_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(SIMINCFL) $(NVCCFL) $< -o $@
