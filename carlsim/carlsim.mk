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
	carlsim_datastructures.h linear_algebra.h)
interface_src := $(addprefix $(interface_dir)/src/,carlsim.cpp \
	user_errors.cpp callback_core.cpp linear_algebra.cpp)
interface_objs := $(patsubst %.cpp, %.o, $(interface_src))

# connection monitor variables
conn_mon_inc := $(addprefix $(conn_mon_dir)/,connection_monitor.h \
	connection_monitor_core.h)
conn_mon_src := $(addprefix $(conn_mon_dir)/, connection_monitor.cpp \
	connection_monitor_core.cpp)
conn_mon_objs := $(patsubst %.cpp, %.o, $(conn_mon_src))
conn_mon_flags := -I$(conn_mon_dir)

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
carlsim_inc += $(kernel_inc) $(interface_inc) $(spike_mon_inc) $(conn_mon_inc)
carlsim_objs += $(kernel_objs) $(interface_objs) $(spike_mon_objs) $(conn_mon_objs)
carlsim_sources += $(kernel_src) $(interface_src) $(spike_mon_src) $(conn_mon_src)
objects += $(carlsim_objs) $(interface_objs) $(spike_mon_objs) $(conn_mon_objs)

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
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(conn_mon_flags) \
$< -o $@

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
