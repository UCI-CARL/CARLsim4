#------------------------------------------------------------------------------
# USER-MODIFIABLE COMPONENT OF MAKEFILE
#
# Note: all paths should be absolute (start with /)
#------------------------------------------------------------------------------
# desired installation path of libcarlsim and headers
CARLSIM_LIB_DIR ?= /opt/CARL/CARLsim

# cuda capability major version number for GPU device
CUDA_MAJOR_NUM ?= 2
# cuda capability minor version number for GPU device
CUDA_MINOR_NUM ?= 0
# cuda toolkit version (major number only)
CARLSIM_CUDAVER ?= 5

# CUDA Installation location. If your CUDA installation is not /usr/local/cuda,
# please set CUDA_INSTALL_PATH to point to the correct location or set it as
# an environment variable.
CUDA_INSTALL_PATH ?= /usr/local/cuda

# $(OPT_LEVEL): set to 1, 2, or 3 if you want to use optimization.  Default: 0.
# $(DEBUG_INFO): set to 1 to include debug info, set to 0 to not include
# debugging info.  Default: 0.
CARLSIM_FASTMATH ?= 0
CARLSIM_CUOPTLEVEL ?= 0
CARLSIM_DEBUG ?= 0


#------------------------------------------------------------------------------
# OPTIONAL FEATURES:
#
# Note: These features aren't necessary for a functioning CARLsim installation.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Deprecated: CARLsim/Evolving Objects Parameter Tuning Interface Options
#------------------------------------------------------------------------------
# path of evolving objects installation for EO-PTI CARLsim support (deprecated)
EO_DIR ?= /opt/eo
EO_PTI_DIR ?= /opt/CARL/carlsim_eo_pti

#------------------------------------------------------------------------------
# CARLsim/ECJ Parameter Tuning Interface Options
#------------------------------------------------------------------------------
# path of evolutionary computation in java installation for ECJ-PTI CARLsim
# support
ECJ_DIR ?= /opt/ecj
ECJ_PTI_DIR ?= /opt/CARL/carlsim_ecj_pti

#------------------------------------------------------------------------------
# CARLsim Developer Features: Running tests and compiling from sources
#------------------------------------------------------------------------------

# path of installation of Google test framework
GTEST_DIR ?= /opt/gtest

# whether to include flag for regression testing
CARLSIM_TEST ?= 0

#------------------------------------------------------------------------------
# END OF USER-MODIFIABLE SECTION
#------------------------------------------------------------------------------

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit).
OS_SIZE 	:=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_LOWER 	:=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
OS_UPPER 	:=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN  	:=$(strip $(findstring DARWIN, $(OS_UPPER)))

# variable defintions
CXX = g++
CC  = g++
NVCC = nvcc
CPPFLAGS = $(DEBUG_FLAG) $(OPT_FLAG) -Wall -std=c++0x

MV := mv -f
RM := rm -rf

# if Mac OS X, include these flags
ifeq ($(DARWIN),DARWIN)
	CARLSIM_FLAGS +=-Xlinker -lstdc++ -lc++
endif

# add compute capability to compile flags
CARLSIM_FLAGS += -arch sm_$(CUDA_MAJOR_NUM)$(CUDA_MINOR_NUM)
ifeq (${strip ${CUDA_MAJOR_NUM}},1)
	CARLSIM_FLAGS += -D__NO_ATOMIC_ADD__
endif

# add CUDA version to compile flags
CARLSIM_FLAGS += -D__CUDA$(CARLSIM_CUDAVER)__

# load appropriate CUDA flags
ifneq (,$(filter $(CARLSIM_CUDAVER),3 4))
	CARLSIM_INCLUDES = -I${NVIDIA_SDK}/C/common/inc/
	CARLSIM_LFLAGS = -L${NVIDIA_SDK}/C/lib
	CARLSIM_LIBS = -lcutil_x86_64
else
	CARLSIM_INCLUDES = -I$(CUDA_INSTALL_PATH)/samples/common/inc/
	CARLSIM_LFLAGS =
	CARLSIM_LIBS =
endif

# use fast math
ifeq ($(strip $(CARLSIM_FASTMATH)),1)
	CARLSIM_FLAGS += -use_fast_math
endif

# use CUDA optimization level
ifneq ($(strip $(CARLSIM_CUOPTLEVEL)),1)
	CARLSIM_FLAGS += -O${CARLSIM_CUOPTLEVEL}
endif

# set debug flag
ifeq ($(strip $(CARLSIM_DEBUG)),1)

endif

# location of .h files
vpath %.h $(kernel_dir)/include $(ex_dir)/common $(interface_dir)/include \
	$(spike_mon_dir) $(tools_spikegen_dir) $(ECJ_PTI_DIR)/include \
	$(input_stim_dir) $(EO_DIR)/src

# location of .cpp files
vpath %.cpp $(kernel_dir)/src $(interface_dir)/src \
	$(spike_info_dir) $(tools_spikegen_dir) $(input_stim_dir) $(ex_dir)/common/

# location of .cu files
vpath %.cu $(kernel_dir)/src
