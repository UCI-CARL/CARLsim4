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
	CARLSIM_FLAGS = -D__CUDA3__ -arch sm_20
endif

# use fast math
ifeq (${strip ${CARLSIM_FASTMATH}},1)
	CARLSIM_FLAGS += -use_fast_math
endif

# use CUDA optimization level
ifneq (${strip ${CARLSIM_CUOPTLEVEL}},1)
	CARLSIM_FLAGS += -O${CARLSIM_CUOPTLEVEL}
endif

# set debug flag
ifeq (${strip ${CARLSIM_DEBUG}},1)
	CARLSIM_FLAGS += -g
endif

# set regression flag
ifeq (${strip ${CARLSIM_TEST}},1)
	CARLSIM_FLAGS += -I$(CURDIR)/$(test_dir) -D__REGRESSION_TESTING__
endif


# append include path to CARLSIM_FLAGS
CARLSIM_FLAGS += -I$(CURDIR)/$(carlsim_dir)/include \
	-I$(CURDIR)/$(interface_dir)/include

spike_info_flags := -I$(spike_info_dir)

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


spike_info_deps := spike_info.h spike_info.cpp
spike_info_src := spike_info.cpp
spike_info_objs := $(spike_info_dir)/spike_info.o
spike_info_flags := -I$(spike_info_dir)
# motion energy objects
util_2_0_objs := $(addprefix $(local_dir)/,v1ColorME.2.0.o)

sources += $(local_src) $(interface_src)
carlsim_deps += $(local_deps) $(interface_deps) $(spike_info_deps)
carlsim_objs += $(local_objs) $(interface_objs) $(spike_info_objs)
carlsim_sources += $(local_src) $(interface_src) $(spike_info_src)
objects += $(carlsim_objs) $(interface_objs) $(spike_info_objs)
all_targets += carlsim

#---------------------------------------------------------------------------
# CARLsim rules
#---------------------------------------------------------------------------
.PHONY: carlsim
carlsim: $(local_src) $(interface_src) $(local_objs) $(interface_objs) \
	$(spike_info_objs)

# interface
$(interface_dir)/src/%.o: $(interface_dir)/src/%.cpp $(interface_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(spike_info_flags) $< -o $@

#util
$(spike_info_dir)/%.o: $(spike_info_dir)/%.cpp $(spike_info_deps)
	$(NVCC) -c $(spike_info_flags) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) \
	$< -o $@

# local cpps
$(local_dir)/src/%.o: $(local_dir)/src/%.cpp $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(spike_info_flags) $< -o $@

# local cuda
$(local_dir)/src/%.o: $(local_dir)/src/%.cu $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(spike_info_flags) $< -o $@
