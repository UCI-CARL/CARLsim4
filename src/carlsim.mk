# module include file for CARLsim core
# edited by KDC

#-------------------------------------------------------------------------------
# CARLsim flags
#-------------------------------------------------------------------------------

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

# append include path to CARLSIM_FLAGS
CARLSIM_FLAGS += -I$(CURDIR)/$(src_dir) -I$(CURDIR)/$(interface_dir)/include

#-------------------------------------------------------------------------------
# CARLsim local variables
#-------------------------------------------------------------------------------
local_dir := $(src_dir)
local_deps := snn.h mtrand.h gpu.h gpu_random.h config.h \
	propagated_spike_buffer.h poisson_rate.h \
	errorCode.h CUDAVersionControl.h
local_src := $(addprefix $(local_dir)/,$(local_deps) snn_cpu.cpp mtrand.cpp \
	propagated_spike_buffer.cpp poisson_rate.cpp \
	printSNNInfo.cpp gpu_random.cu \
	snn_gpu.cu v1ColorME.2.0.cu v1ColorME.2.1.cu)
local_objs := $(addprefix $(local_dir)/,snn_cpu.o  mtrand.o \
	propagated_spike_buffer.o poisson_rate.o printSNNInfo.o \
	gpu_random.o snn_gpu.o)


interface_deps := carlsim.h carlsim.cpp user_errors.h user_errors.cpp
interface_src := $(interface_dir)/include/carlsim.h \
	$(interface_dir)/include/user_errors.h \
	$(interface_dir)/src/carlsim.cpp $(interface_dir)/src/user_errors.cpp
interface_objs := $(interface_dir)/src/carlsim.o \
	$(interface_dir)/src/user_errors.o



# motion energy objects
util_2_0_objs := $(addprefix $(local_dir)/,v1ColorME.2.0.o)
util_2_1_objs := $(addprefix $(local_dir)/,v1ColorME.2.1.o)

sources += $(local_src) $(interface_src)
carlsim_deps += $(local_deps) $(interface_deps)
carlsim_objs += $(local_objs) $(interface_objs)
carlsim_sources += $(local_src) $(interface_src)
objects += $(carlsim_objs) $(interface_objs) $(util_2_0_objs) $(util_2_1_objs)
all_targets += CARLsim

#-------------------------------------------------------------------------------
# CARLsim rules
#-------------------------------------------------------------------------------
.PHONY: carlsim
carlsim: $(local_src) $(interface_src) $(local_objs) $(interface_objs)

# interface
$(interface_dir)/src/%.o: $(interface_dir)/src/%.cpp $(interface_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@

# local cpps
$(local_dir)/%.o: $(local_dir)/%.cpp $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@

# local cuda
$(local_dir)/%.o: $(local_dir)/%.cu $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@
