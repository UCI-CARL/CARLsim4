# module include file for CARLsim core

#---------------------------------------------------------------------------
# CARLsim kernel variables
#---------------------------------------------------------------------------
# kernel variables
kernel_inc := $(addprefix $(kernel_dir)/include/, snn.h gpu.h \
	snn_definitions.h snn_datastructures.h \
	gpu_random.h propagated_spike_buffer.h \
	error_code.h cuda_version_control.h)
kernel_cpp := $(addprefix $(kernel_dir)/src/, snn_cpu.cpp \
	propagated_spike_buffer.cpp print_snn_info.cpp)
kernel_cu := $(addprefix $(kernel_dir)/src/, gpu_random.cu snn_gpu.cu)
kernel_src := $(kernel_cpp) $(kernel_cu)
kernel_cpp_objs := $(patsubst %.cpp, %.o, $(kernel_cpp))
kernel_cu_objs := $(patsubst %.cu, %.o, $(kernel_cu))
kernel_objs := $(kernel_cpp_objs) $(kernel_cu_objs)

# interface variables
interface_inc := $(addprefix $(interface_dir)/include/, carlsim.h \
	user_errors.h callback.h callback_core.h carlsim_definitions.h \
	carlsim_datastructures.h poisson_rate.h linear_algebra.h)
interface_src := $(addprefix $(interface_dir)/src/,carlsim.cpp \
	user_errors.cpp callback_core.cpp poisson_rate.cpp linear_algebra.cpp)
interface_objs := $(patsubst %.cpp, %.o, $(interface_src))

# spike monitor variables
spike_mon_inc := $(addprefix $(spike_mon_dir)/,spike_monitor.h \
	spike_monitor_core.h)
spike_mon_src := $(addprefix $(spike_mon_dir)/, spike_monitor.cpp \
	spike_monitor_core.cpp)
spike_mon_objs := $(patsubst %.cpp, %.o, $(spike_mon_src))

# tools spikegen variables
tools_spikegen_inc  := $(addprefix $(tools_spikegen_dir)/,interactive_spikegen.h \
	periodic_spikegen.h spikegen_from_file.h spikegen_from_vector.h)
tools_spikegen_src  := $(addprefix $(tools_spikegen_dir)/,interactive_spikegen.cpp \
	periodic_spikegen.cpp spikegen_from_file.cpp spikegen_from_vector.cpp)
tools_spikegen_objs := $(patsubst %.cpp, %.o, $(tools_spikegen_src))

# tools inputstim variables
tools_inputstim_inc  := $(addprefix $(tools_inputstim_dir)/,input_stimulus.h)
tools_inputstim_src  := $(addprefix $(tools_inputstim_dir)/,input_stimulus.cpp)
tools_inputstim_objs := $(patsubst %.cpp, %.o, $(tools_inputstim_src))

# tools simple weight tuner variables
tools_swt_inc  := $(addprefix $(tools_swt_dir)/,simple_weight_tuner.h)
tools_swt_src  := $(addprefix $(tools_swt_dir)/,simple_weight_tuner.cpp)
tools_swt_objs := $(patsubst %.cpp, %.o, $(tools_swt_src))

# motion energy objects
util_2_0_objs := $(addprefix $(kernel_dir)/,v1ColorME.2.0.o)

# carlsim variables all together in one place
carlsim_inc += $(kernel_inc) $(interface_inc) $(spike_mon_inc) $(tools_spikegen_inc) \
	$(tools_inputstim_inc) $(tools_swt_inc)
carlsim_objs += $(kernel_objs) $(interface_objs) $(spike_mon_objs) $(tools_spikegen_objs) \
	$(tools_inputstim_objs) $(tools_swt_objs)
carlsim_sources += $(kernel_src) $(interface_src) $(spike_mon_src) $(tools_spikegen_src) \
	$(tools_inputstim_src) $(tools_swt_src)
objects += $(carlsim_objs) $(interface_objs) $(spike_mon_objs) $(tools_spikegen_objs) \
	$(tools_inputstim_objs) $(tools_swt_objs)

default_targets += carlsim

#---------------------------------------------------------------------------
# CARLsim rules
#---------------------------------------------------------------------------
# put libcuda stuff here
.PHONY: carlsim
carlsim: $(carlsim_sources) $(carlsim_inc) $(carlsim_objs)

# interface
$(interface_dir)/src/%.o: $(interface_dir)/src/%.cpp $(interface_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

# spike_monitor
$(spike_mon_dir)/%.o: $(spike_mon_dir)/%.cpp $(spike_mon_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS)	$< -o $@

# tools/spikegen
$(tools_spikegen_dir)/%.o: $(tools_spikegen_src) $(tools_spikegen_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS)	$< -o $@

# tools/input_stimulus
$(tools_inputstim_dir)/%.o: $(tools_inputstim_src) $(tools_inputstim_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

# tools/simple_weight_tuner
$(tools_swt_dir)/%.o: $(tools_swt_src) $(tools_swt_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

# kernel carlsim cpps
$(kernel_dir)/src/%.o: $(kernel_dir)/src/%.cpp $(kernel_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

# kernel carlsim cuda
$(kernel_dir)/src/%.o: $(kernel_dir)/src/%.cu $(kernel_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@
