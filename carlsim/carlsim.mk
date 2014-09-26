# module include file for CARLsim core

#---------------------------------------------------------------------------
# CARLsim kernel variables
#---------------------------------------------------------------------------
# kernel variables
kernel_inc := $(addprefix $(kernel_dir)/include/, snn.h mtrand.h gpu.h \
	snn_definitions.h snn_datastructures.h \
	gpu_random.h propagated_spike_buffer.h poisson_rate.h \
	error_code.h cuda_version_control.h)
kernel_cpp := $(addprefix $(kernel_dir)/src/, snn_cpu.cpp mtrand.cpp \
	propagated_spike_buffer.cpp poisson_rate.cpp print_snn_info.cpp)
kernel_cu := $(addprefix $(kernel_dir)/src/, gpu_random.cu snn_gpu.cu)
kernel_src := $(kernel_cpp) $(kernel_cu)
kernel_cpp_objs := $(patsubst %.cpp, %.o, $(kernel_cpp))
kernel_cu_objs := $(patsubst %.cu, %.o, $(kernel_cu))
kernel_objs := $(kernel_cpp_objs) $(kernel_cu_objs)

# interface variables
interface_inc := $(addprefix $(interface_dir)/include/, carlsim.h \
	user_errors.h callback.h callback_core.h carlsim_definitions.h \
	carlsim_datastructures.h)
interface_src := $(addprefix $(interface_dir)/src/,carlsim.cpp \
	user_errors.cpp callback_core.cpp)
interface_objs := $(patsubst %.cpp, %.o, $(interface_src))

# spike monitor variables
spike_mon_inc := $(addprefix $(spike_mon_dir)/,spike_monitor.h \
	spike_monitor_core.h)
spike_mon_src := $(addprefix $(spike_mon_dir)/, spike_monitor.cpp \
	spike_monitor_core.cpp)
spike_mon_objs := $(patsubst %.cpp, %.o, $(spike_mon_src))
spike_mon_flags := -I$(spike_mon_dir)

# motion energy objects
util_2_0_objs := $(addprefix $(kernel_dir)/,v1ColorME.2.0.o)

# carlsim variables all together in one place
carlsim_inc += $(kernel_inc) $(interface_inc) $(spike_mon_inc)
carlsim_objs += $(kernel_objs) $(interface_objs) $(spike_mon_objs)
carlsim_sources += $(kernel_src) $(interface_src) $(spike_mon_src)
objects += $(carlsim_objs) $(interface_objs) $(spike_mon_objs)

default_targets += carlsim libCARLsim

#---------------------------------------------------------------------------
# CARLsim rules
#---------------------------------------------------------------------------
# put libcuda stuff here
.PHONY: carlsim libCARLsim install
carlsim: $(carlsim_sources) $(carlsim_inc) $(carlsim_objs)

# interface
$(interface_dir)/src/%.o: $(interface_dir)/src/%.cpp $(interface_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

# spike_monitor
$(spike_mon_dir)/%.o: $(spike_mon_dir)/%.cpp $(spike_mon_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(spike_mon_flags) \
$< -o $@

# kernel carlsim cpps
$(kernel_dir)/src/%.o: $(kernel_dir)/src/%.cpp $(kernel_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

# kernel carlsim cuda
$(kernel_dir)/src/%.o: $(kernel_dir)/src/%.cu $(kernel_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

num_ver := $(carlsim_major_num).$(carlsim_minor_num)

lib_ver := $(num_ver).$(carlsim_build_num)

lib_name := libCARLsim.a

carlsim_lib := $(addprefix carlsim/,$(lib_name))
# keep track of this so we can delete it later on distclean
libraries += $(carlsim_lib)

libCARLsim: $(carlsim_lib)

$(carlsim_lib): $(carlsim_sources) $(carlsim_inc) $(carlsim_objs)
	ar rcs $@.$(lib_ver) $(carlsim_objs)

# TODO: in the readme tell the user how to uninstall the library (just delete the
# TODO: Consider using the library naming convention on the CARLsim wiki to name the
# library and use a symlink to it. Maybe add this as an issue.
# $(CARLSIM_LIB_INSTALL_DIR))
# TODO: remove all the libraries in the src directory upon distclean
install: $(carlsim_lib)
	@test -d $(CARLSIM_LIB_INSTALL_DIR) || \
		mkdir -p $(CARLSIM_LIB_INSTALL_DIR)
	@test -d $(CARLSIM_LIB_INSTALL_DIR)/lib || mkdir \
		$(CARLSIM_LIB_INSTALL_DIR)/lib
	@test -d $(CARLSIM_LIB_INSTALL_DIR)/include || mkdir \
		$(CARLSIM_LIB_INSTALL_DIR)/include
	@test -d $(CARLSIM_LIB_INSTALL_DIR)/include/kernel || mkdir \
		$(CARLSIM_LIB_INSTALL_DIR)/include/kernel
	@test -d $(CARLSIM_LIB_INSTALL_DIR)/include/interface || mkdir \
		$(CARLSIM_LIB_INSTALL_DIR)/include/interface
	@test -d $(CARLSIM_LIB_INSTALL_DIR)/include/spike_monitor || mkdir \
		$(CARLSIM_LIB_INSTALL_DIR)/include/spike_monitor
	@test -d $(CARLSIM_LIB_INSTALL_DIR)/include/spike_generators || mkdir \
		$(CARLSIM_LIB_INSTALL_DIR)/include/spike_generators
	@install -m 0755 $(carlsim_lib) $(CARLSIM_LIB_INSTALL_DIR)/lib
	@ln -fs $(CARLSIM_LIB_INSTALL_DIR)/lib/$(lib_name).$(lib_ver) \
		$(CARLSIM_LIB_INSTALL_DIR)/lib/$(lib_name).$(num_ver)
	@ln -fs $(CARLSIM_LIB_INSTALL_DIR)/lib/$(lib_name).$(num_ver) \
		$(CARLSIM_LIB_INSTALL_DIR)/lib/$(lib_name)
	@install -m 0644 $(kernel_dir)/include/cuda_version_control.h \
		$(kernel_dir)/include/poisson_rate.h $(CARLSIM_LIB_INSTALL_DIR)/include/kernel
	@install -m 0644 $(interface_dir)/include/callback.h \
		$(interface_dir)/include/carlsim_datastructures.h \
		$(interface_dir)/include/carlsim_definitions.h \
		$(interface_dir)/include/carlsim.h $(interface_dir)/include/user_errors.h \
		$(CARLSIM_LIB_INSTALL_DIR)/include/interface
	@install -m 0644 $(spike_mon_dir)/spike_monitor.h \
	$(CARLSIM_LIB_INSTALL_DIR)/include/spike_monitor
	@install -m 0644 $(spike_gen_dir)/periodic_spikegen.h \
		$(spike_gen_dir)/spikegen_from_file.h \
		$(spike_gen_dir)/spikegen_from_vector.h $(CARLSIM_LIB_INSTALL_DIR)/include/spike_generators
