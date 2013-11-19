########################################################################################################################
#
# CARLsim Makefile
#
# CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
# Ver 11/05/2013
#
########################################################################################################################


########################################################################################################################
# HOW TO USE
########################################################################################################################
#
# In order to display all available example experiments, open up a terminal in the current directory and type
# $ make help
# In order to clean up object files, executables, etc., type
# $ make clean
# 
# CARLsim simulation examples follow a common rule: Some experiment (e.g., called "test") has a source file named
# "main_test.cpp" in subdirectory "examples/test/", and should direct all of its output files to a subdirectory
# "Results/test/". If Matlab analysis scripts are available for this particular experiment, they can be found in a
# subdirectory "scripts/test/".
# The experiment can be compiled by typing:
# $ make test
# This will create an executable called "test". In order to run it, type:
# $ ./test
# This naming convention is true for all example simulations.
#
# Required environment variables:
#	${NVIDIA_SDK}: path to NVIDIA GPU Computing SDK
#		no longer required in combination with CUDA 5
#
# Optional environment variables:
#	${CARLSIM_CUDAVER}: which CUDA version to use
#		set to version number you want to use (int). Default: 3
#	${CARLSIM_FASTMATH}: whether to use fast math flags
#		set to 1 if you want to use fast math. Default: 0


########################################################################################################################
# COMMON BUILD
########################################################################################################################

# if optional env vars do not exist, assign default values
CARLSIM_CUDAVER ?= 3
CARLSIM_FASTMATH ?= 0

ifeq (${strip ${CARLSIM_CUDAVER}},5)
	INCLUDES = -I/usr/local/cuda/samples/common/inc/
	LFLAGS =
	LIBS =
	CFLAGS = -D__CUDA5__ -arch sm_20
else
	INCLUDES = -I${NVIDIA_SDK}/C/common/inc/
	LFLAGS = -L${NVIDIA_SDK}/C/lib
	LIBS = -lcutil_x86_64
	CFLAGS = -D__CUDA3__ -arch sm_13
endif

ifeq (${strip ${CARLSIM_FASTMATH}},1)
	CFLAGS += -O3 -use_fast_math
endif

CC = nvcc
SRCS = snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu v1ColorME.2.0.cu v1ColorME.2.1.cu
DEP = snn.h PropagatedSpikeBuffer.h gpu.h gpu_random.h mtrand.h config.h CUDAVersionControl.h

#OBJS = ${SRCS:.cpp=.o}

CORE_OBJS = snn_cpu.o \
            snn_gpu.o \
            mtrand.o \
            PropagatedSpikeBuffer.o \
            printSNNInfo.o \
            gpu_random.o \

UTIL_2_0_OBJS = v1ColorME.2.0.o
UTIL_2_1_OBJS = v1ColorME.2.1.o

# examples are split according to the version of v1ColorME they are using
EXE_CU_NONE = random
EXE_CU_20 = colorblind colorcycle orientation rdk v1v4PFC
EXE_CU_21 = v1MTLIP



########################################################################################################################
# RULES
########################################################################################################################

all: ${EXE_CU_21} ${EXE_CU_20} ${EXE_CU_NONE}

.SECONDEXPANSION:
# using none of the v1ColorME.cu
${EXE_CU_NONE}: ${DEP} ${CORE_OBJS} examples/$$@/main_$$@.cpp
	${CC} ${INCLUDES} ${LFLAGS} ${LIBS} ${CFLAGS} ${CORE_OBJS} examples/$@/main_$@.cpp -o $@

# using v1ColorME.2.0.cu
${EXE_CU_20}: ${DEP} ${CORE_OBJS} ${UTIL_2_0_OBJS} examples/$$@/main_$$@.cpp
	${CC} ${INCLUDES} ${LFLAGS} ${LIBS} ${CFLAGS} ${CORE_OBJS} examples/$@/main_$@.cpp ${UTIL_2_0_OBJS} -o $@

# using v1ColorME.2.1.cu
${EXE_CU_21}: ${DEP} ${CORE_OBJS} ${UTIL_2_1_OBJS} examples/$$@/main_$$@.cpp
	${CC} ${INCLUDES} ${LFLAGS} ${LIBS} ${CFLAGS} ${CORE_OBJS} examples/$@/main_$@.cpp ${UTIL_2_1_OBJS} -o $@

# object files
%.o: %.cpp ${DEP}
	${CC} -c ${INCLUDES} ${LFLAGS} ${CFLAGS} $< -o $@

%.o: %.cu ${DEP}
	${CC} -c ${INCLUDES} ${LFLAGS} ${CFLAGS} $< -o $@


########################################################################################################################
# MAINTENANCE AND SPECIAL RULES
########################################################################################################################

clean:
	rm -f *.o *~ *.dot param.txt *.log ${EXE_CU} ${EXE_CU_21} ${EXE_CU_20} ${EXE_CU_NONE}

help:
	@echo Choose from the following example networks: ${EXE_CU} ${EXE_CU_21} ${EXE_CU_20} ${EXE_CU_NONE}.
	@echo Create env var USECUDAVER to specify which CUDA version to use \(Default: 5\)
