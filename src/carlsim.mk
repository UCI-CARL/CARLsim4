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
CARLSIM_FLAGS += -I$(CURDIR)/$(src_dir)

#-------------------------------------------------------------------------------
# CARLsim local variables
#-------------------------------------------------------------------------------
local_dir := src
local_deps := snn.h mtrand.h PropagatedSpikeBuffer.h errorCode.h gpu.h \
	gpu_random.h CUDAVersionControl.h config.h	
local_src := $(addprefix $(local_dir)/,$(local_deps) snn_cpu.cpp mtrand.cpp \
	PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu \
	v1ColorME.2.0.cu v1ColorME.2.1.cu)
local_objs := $(addprefix $(local_dir)/,snn_cpu.o  mtrand.o \
	PropagatedSpikeBuffer.o printSNNInfo.o gpu_random.o snn_gpu.o)

# motion energy objects
util_2_0_objs := $(addprefix $(local_dir)/,v1ColorME.2.0.o)
util_2_1_objs := $(addprefix $(local_dir)/,v1ColorME.2.1.o)

sources += $(local_src)
carlsim_deps += $(local_deps)
carlsim_objs += $(local_objs)
carlsim_sources += $(local_src)
objects += $(carlsim_objs) $(util_2_0_objs) $(util_2_1_objs)
all_targets += CARLsim

#-------------------------------------------------------------------------------
# CARLsim rules
#-------------------------------------------------------------------------------
.PHONY: CARLsim
CARLsim: $(local_src) $(local_objs)

$(local_dir)/%.o: $(local_dir)/%.cpp $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@

$(local_dir)/%.o: $(local_dir)/%.cu $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@
