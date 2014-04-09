# module include file for CARLsim core
# edited by KDC

#---------------------------------------------------------------------------
# CARLsim flags
#---------------------------------------------------------------------------

ifeq (${strip ${CARLSIM_CUDAVER}},5)
	CARLSIM_INCLUDES = -I/usr/local/cuda/samples/common/inc/
	CARLSIM_LFLAGS =
	CARLSIM_LIBS =
	CARLSIM_FLAGS = -D__CUDA5__ -arch sm_20
else
	CARLSIM_INCLUDES = -I${NVIDIA_SDK}/C/common/inc/
	CARLSIM_LFLAGS = -L${NVIDIA_SDK}/C/lib
	CARLSIM_LIBS = -lcutil_x86_64
	CARLSIM_FLAGS = -D__CUDA3__ -arch sm_13
endif

# use fast math
ifeq (${strip ${CARLSIM_FASTMATH}},1)
	CARLSIM_FLAGS += -use_fast_math
endif

# use CUDA optimization level
ifneq (${strip ${CARLSIM_CUOPTLEVEL}},1)
	CARLSIM_FLAGS += -O${CARLSIM_CUOPTLEVEL}
endif

# append include path to CARLSIM_FLAGS
CARLSIM_FLAGS += -I$(CURDIR)/$(carlsim_dir)/include \
	-I$(CURDIR)/$(interface_dir)/include

#---------------------------------------------------------------------------
# CARLsim local variables
#---------------------------------------------------------------------------
local_dir := $(carlsim_dir)
local_deps := $(addprefix $(local_dir)/include/, snn.h mtrand.h gpu.h \
	snn_definitions.h snn_datastructures.h \
	gpu_random.h propagated_spike_buffer.h poisson_rate.h \
	error_code.h cuda_version_control.h)
local_src := $(addprefix $(local_dir)/src/, snn_cpu.cpp mtrand.cpp \
	propagated_spike_buffer.cpp poisson_rate.cpp \
	print_snn_info.cpp gpu_random.cu \
	snn_gpu.cu)
local_objs := $(addprefix $(local_dir)/src/,snn_cpu.o mtrand.o \
	propagated_spike_buffer.o poisson_rate.o print_snn_info.o \
	gpu_random.o snn_gpu.o)


interface_deps := carlsim.h carlsim.cpp user_errors.h user_errors.cpp \
	callback.h callback_core.h callback_core.cpp carlsim_definitions.h \
	carlsim_datastructures.h
interface_src := $(interface_dir)/src/carlsim.cpp \
	$(interface_dir)/src/user_errors.cpp \
	$(interface_dir)/src/callback_core.cpp
interface_objs := $(interface_dir)/src/carlsim.o \
	$(interface_dir)/src/user_errors.o \
	$(interface_dir)/src/callback_core.o


# motion energy objects
util_2_0_objs := $(addprefix $(local_dir)/,v1ColorME.2.0.o)

sources += $(local_src) $(interface_src)
carlsim_deps += $(local_deps) $(interface_deps)
carlsim_objs += $(local_objs) $(interface_objs)
carlsim_sources += $(local_src) $(interface_src)
objects += $(carlsim_objs) $(interface_objs)
all_targets += carlsim

#---------------------------------------------------------------------------
# CARLsim rules
#---------------------------------------------------------------------------
.PHONY: carlsim
carlsim: $(local_src) $(interface_src) $(local_objs) $(interface_objs)

# interface
$(interface_dir)/src/%.o: $(interface_dir)/src/%.cpp $(interface_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@

# local cpps
$(local_dir)/src/%.o: $(local_dir)/src/%.cpp $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@

# local cuda
$(local_dir)/src/%.o: $(local_dir)/src/%.cu $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@
