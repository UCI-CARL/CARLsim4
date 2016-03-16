# module include file for CARLsim core

#---------------------------------------------------------------------------
# CARLsim kernel variables
#---------------------------------------------------------------------------
# kernel variables
kernel_inc := $(addprefix $(kernel_dir)/include/, snn.h \
	snn_definitions.h snn_datastructures.h \
	propagated_spike_buffer.h \
	error_code.h cuda_version_control.h)
kernel_cpp := $(addprefix $(kernel_dir)/src/, snn_manager.cpp snn_cpu_module.cpp \
	propagated_spike_buffer.cpp print_snn_info.cpp)
kernel_cu := $(addprefix $(kernel_dir)/src/, snn_gpu_module.cu)
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

# connection monitor variables
conn_mon_inc := $(addprefix $(conn_mon_dir)/,connection_monitor_core.h \
	connection_monitor.h)
conn_mon_src := $(addprefix $(conn_mon_dir)/, connection_monitor_core.cpp \
	connection_monitor.cpp)
conn_mon_objs := $(patsubst %.cpp, %.o, $(conn_mon_src))
conn_mon_flags := -I$(conn_mon_dir)

# spike monitor variables
spike_mon_inc := $(addprefix $(spike_mon_dir)/,spike_monitor.h \
	spike_monitor_core.h)
spike_mon_src := $(addprefix $(spike_mon_dir)/, spike_monitor.cpp \
	spike_monitor_core.cpp)
spike_mon_objs := $(patsubst %.cpp, %.o, $(spike_mon_src))

# group monitor variables
group_mon_inc := $(addprefix $(group_mon_dir)/, group_monitor.h \
	group_monitor_core.h)
group_mon_src := $(addprefix $(group_mon_dir)/, group_monitor.cpp \
	group_monitor_core.cpp)
group_mon_objs := $(patsubst %.cpp, %.o, $(group_mon_src))

# tools spikegen variables
tools_spikegen_inc  := $(addprefix $(tools_spikegen_dir)/,interactive_spikegen.h \
	periodic_spikegen.h spikegen_from_file.h spikegen_from_vector.h)
tools_spikegen_src  := $(addprefix $(tools_spikegen_dir)/,interactive_spikegen.cpp \
	periodic_spikegen.cpp spikegen_from_file.cpp spikegen_from_vector.cpp pre_post_group_spikegen.cpp)
tools_spikegen_objs := $(patsubst %.cpp, %.o, $(tools_spikegen_src))

# tools visualstim variables
tools_visualstim_inc  := $(addprefix $(tools_visualstim_dir)/,visual_stimulus.h)
tools_visualstim_src  := $(addprefix $(tools_visualstim_dir)/,visual_stimulus.cpp)
tools_visualstim_objs := $(patsubst %.cpp, %.o, $(tools_visualstim_src))

# tools simple weight tuner variables
tools_swt_inc  := $(addprefix $(tools_swt_dir)/,simple_weight_tuner.h)
tools_swt_src  := $(addprefix $(tools_swt_dir)/,simple_weight_tuner.cpp)
tools_swt_objs := $(patsubst %.cpp, %.o, $(tools_swt_src))

tools_stopwatch_inc  := $(addprefix $(tools_stopwatch_dir)/,stopwatch.h)
tools_stopwatch_src  := $(addprefix $(tools_stopwatch_dir)/,stopwatch.cpp)
tools_stopwatch_objs := $(patsubst %.cpp, %.o, $(tools_stopwatch_src))

# motion energy objects
util_2_0_objs := $(addprefix $(kernel_dir)/,v1ColorME.2.0.o)

# carlsim variables all together in one place
carlsim_inc += $(kernel_inc) $(interface_inc) $(conn_mon_inc) $(spike_mon_inc) $(group_mon_inc) \
	$(tools_spikegen_inc) $(tools_visualstim_inc) $(tools_swt_inc) $(tools_stopwatch_inc)
carlsim_objs += $(kernel_objs) $(interface_objs) $(conn_mon_objs) $(spike_mon_objs) $(group_mon_objs) \
	$(tools_spikegen_objs) $(tools_visualstim_objs) $(tools_swt_objs) $(tools_stopwatch_objs)
carlsim_sources += $(kernel_src) $(interface_src) $(conn_mon_src) $(spike_mon_src) $(group_mon_src) \
	$(tools_spikegen_src) $(tools_visualstim_src) $(tools_swt_src) $(tools_stopwatch_src)
objects += $(carlsim_objs) $(interface_objs) $(conn_mon_objs) $(spike_mon_objs) $(group_mon_objs) \
	$(tools_spikegen_objs) $(tools_visualstim_objs) $(tools_swt_objs) $(tools_stopwatch_objs)

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

# connection monitor
$(conn_mon_dir)/%.o: $(conn_mon_dir)/%.cpp $(conn_mon_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(conn_mon_flags) $< -o $@

# spike_monitor
$(spike_mon_dir)/%.o: $(spike_mon_dir)/%.cpp $(spike_mon_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS)	$< -o $@

# group_monitor
$(group_mon_dir)/%.o: $(group_mon_dir)/%.cpp $(group_mon_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS)	$< -o $@

# tools/spikegen
$(tools_spikegen_dir)/%.o: $(tools_spikegen_src) $(tools_spikegen_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS)	$(@D)/$*.cpp -o $@

# tools/visual_stimulus
$(tools_visualstim_dir)/%.o: $(tools_visualstim_src) $(tools_visualstim_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(@D)/$*.cpp -o $@

# tools/simple_weight_tuner
$(tools_swt_dir)/%.o: $(tools_swt_src) $(tools_swt_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(@D)/$*.cpp -o $@

# tools/stopwatch
$(tools_stopwatch_dir)/%.o: $(tools_stopwatch_src) $(tools_stopwatch_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(@D)/$*.cpp -o $@

# kernel carlsim cpps
$(kernel_dir)/src/%.o: $(kernel_dir)/src/%.cpp $(kernel_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@

# kernel carlsim cuda
$(kernel_dir)/src/%.o: $(kernel_dir)/src/%.cu $(kernel_inc)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@
