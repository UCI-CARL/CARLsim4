#------------------------------------------------------------------------------
# CARLsim Engine Makefile
#
# Note: This file depends on variables set in configure.mk, thus must be run
# after importing those others.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# CARLsim Kernel
#------------------------------------------------------------------------------

# kernel_dir       := carlsim/kernel
# kernel_obj_dir   := carlsim/kernel
# kernel_inc_files :=
# kernel_src_files :=
# kernel_obj_files :=
# KERNELINCFL   :=
# include $(kernel_dir)/kernel.mk


# # kernel has all of core and all backends
# kernel_dir     := carlsim/kernel
# kernel_inc_dir := $(sort $(wildcard $(kernel_dir)/*/inc))
# KRNLINCFL   := $(addprefix -I,$(kernel_inc_dir))

# # core
# core_dir       := $(kernel_dir)
# core_obj_dir   := $(kernel_dir)
# core_inc_files :=
# core_src_files :=
# core_obj_files :=
# COREINCFL   :=
# include $(kernel_dir)/core.mk

# # connection_monitor
# conn_dir       := $(kernel_dir)/core
# conn_obj_dir   := $(kernel_dir)/core
# conn_inc_files :=
# conn_src_files :=
# conn_obj_files :=
# CONNINCFL   :=
# include $(conn_dir)/core.mk



# # backends
# back_so_files  :=
# back_inc_files :=
# back_src_files :=
# BACKINCFL   :=
# include $(kernel_dir)/cpu/cpu.mk


#------------------------------------------------------------------------------
# CARLsim Interface
#------------------------------------------------------------------------------
intf_dir       := carlsim/interface
intf_inc_files := $(wildcard $(intf_dir)/inc/*.h)
intf_cpp_files := $(wildcard $(intf_dir)/src/*.cpp)
intf_cu_files  := $(wildcard $(intf_dir)/src/*.cu)
intf_obj_files := $(patsubst %.cpp, %.cpp.o, $(intf_cpp_files))
intf_obj_files += $(patsubst %.cu, %.cu.o, $(intf_cu_files))
INTFINCFL      := -I$(intf_dir)/inc

krnl_dir       := carlsim/kernel
krnl_inc_files := $(wildcard $(krnl_dir)/inc/*.h)
krnl_cpp_files := $(wildcard $(krnl_dir)/src/*.cpp)
krnl_cu_files  := $(wildcard $(krnl_dir)/src/*.cu)
krnl_obj_files := $(patsubst %.cpp, %.cpp.o, $(krnl_cpp_files))
krnl_obj_files += $(patsubst %.cu, %.cu.o, $(krnl_cu_files))
KRNLINCFL      := -I$(krnl_dir)/inc

conn_dir       := carlsim/connection_monitor
conn_inc_files := $(wildcard $(conn_dir)/*.h)
conn_cpp_files := $(wildcard $(conn_dir)/*.cpp)
conn_cu_files  := $(wildcard $(conn_dir)/src/*.cu)
conn_obj_files := $(patsubst %.cpp, %.cpp.o, $(conn_cpp_files))
conn_obj_files += $(patsubst %.cu, %.cu.o, $(conn_cu_files))
CONNINCFL      := -I$(conn_dir)

grps_dir       := carlsim/group_monitor
grps_inc_files := $(wildcard $(grps_dir)/*.h)
grps_cpp_files := $(wildcard $(grps_dir)/*.cpp)
grps_cu_files  := $(wildcard $(grps_dir)/src/*.cu)
grps_obj_files := $(patsubst %.cpp, %.cpp.o, $(grps_cpp_files))
grps_obj_files += $(patsubst %.cu, %.cu.o, $(grps_cu_files))
GRPSINCFL      := -I$(grps_dir)

spks_dir       := carlsim/spike_monitor
spks_inc_files := $(wildcard $(spks_dir)/*.h)
spks_cpp_files := $(wildcard $(spks_dir)/*.cpp)
spks_cu_files  := $(wildcard $(spks_dir)/src/*.cu)
spks_obj_files := $(patsubst %.cpp, %.cpp.o, $(spks_cpp_files))
spks_obj_files += $(patsubst %.cu, %.cu.o, $(spks_cu_files))
SPKSINCFL      := -I$(spks_dir)



#------------------------------------------------------------------------------
# CARLsim Common
#------------------------------------------------------------------------------

CARLSIMINCFL := $(INTFINCFL) $(KRNLINCFL) $(SPKSINCFL) $(CONNINCFL) \
	$(GRPSINCFL)
targets += carlsim4
objects += $(krnl_obj_files) $(intf_obj_files) $(conn_obj_files) \
	$(grps_obj_files) $(spks_obj_files)

.PHONY: release debug carlsim4

# release build
release: CXXFLAGS += -O3 -ffast-math
release: NVCCFLAGS += --compiler-options "-O3 -ffast-math"
release: $(targets)

# debug build
debug: CXXFLAGS += -g -Wall
debug: NVCCFLAGS += -g -G
debug: $(targets)

# uninstall library
uninstall:
	$(RM) $(CARLSIM_INSTALL_DIR)

# all CARLsim4 targets
carlsim4: $(objects)

test: $(intf_obj_files)
	@ echo "hello"
	@ echo $(intf_obj_files)



# rule to compile local cpps
$(intf_dir)/src/%.cpp.o: $(intf_dir)/src/%.cpp $(intf_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(CARLSIMINCFL) $(NVCCFL) $< -o $@
$(intf_dir)/src/%.cu.o: $(intf_dir)/src/%.cu $(intf_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(CARLSIMINCFL) $(NVCCFL) $< -o $@
$(krnl_dir)/src/%.cpp.o: $(krnl_dir)/src/%.cpp $(krnl_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(CARLSIMINCFL) $(NVCCFL) $< -o $@
$(krnl_dir)/src/%.cu.o: $(krnl_dir)/src/%.cu $(krnl_inc_files)
	$(NVCC) $(NVCCSHRFL)-c $(NVCCINCFL) $(CARLSIMINCFL) $(NVCCFL) $< -o $@

$(conn_dir)/%.cpp.o: $(conn_dir)/%.cpp $(conn_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(CARLSIMINCFL) $(NVCCFL) $< -o $@
$(grps_dir)/%.cpp.o: $(grps_dir)/%.cpp $(grps_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(CARLSIMINCFL) $(NVCCFL) $< -o $@
$(spks_dir)/%.cpp.o: $(spks_dir)/%.cpp $(spks_inc_files)
	$(NVCC) $(NVCCSHRFL) -c $(NVCCINCFL) $(CARLSIMINCFL) $(NVCCFL) $< -o $@
