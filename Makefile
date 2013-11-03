# CARLsim Makefile
# CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
# Ver 07/13/2013

# source files of the sim core
SOURCES = snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu
DEP = snn.h PropagatedSpikeBuffer.h errorCode.h config.h gpu_random.hpp
LIBCUTIL = -lcutil_x86_64

colorblind: ${SOURCES} ${DEP} examples/colorblind/main_colorblind.cpp v1ColorMEold.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib ${LIBCUTIL} -D__CUDA3__ -arch sm_13 ${SOURCES} examples/colorblind/main_colorblind.cpp v1ColorMEold.cu -o colorblind

colorcycle: ${SOURCES} ${DEP} examples/colorcycle/main_colorcycle.cpp v1ColorMEold.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib ${LIBCUTIL} -D__CUDA3__ -arch sm_13 ${SOURCES} examples/colorcycle/main_colorcycle.cpp v1ColorMEold.cu -o colorcycle

orientation: ${SOURCES} ${DEP} examples/orientation/main_orientation.cpp v1ColorMEold.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib ${LIBCUTIL} -D__CUDA3__ -arch sm_13 ${SOURCES} examples/orientation/main_orientation.cpp v1ColorMEold.cu -o orientation

random: ${SOURCES} ${DEP} examples/random/main_random.cpp v1ColorMEold.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib ${LIBCUTIL} -D__CUDA3__ -arch sm_13 ${SOURCES} examples/random/main_random.cpp v1ColorMEold.cu -o random

rdk: ${SOURCES} ${DEP} examples/rdk/main_rdk.cpp v1ColorMEold.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib ${LIBCUTIL} -D__CUDA3__ -arch sm_13 ${SOURCES} examples/rdk/main_rdk.cpp v1ColorMEold.cu -o rdk

v1MTLIP: ${SOURCES} ${DEP} examples/v1MTLIP/main_v1MTLIP.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib ${LIBCUTIL} -D__CUDA3__ -arch sm_13 ${SOURCES} examples/v1MTLIP/main_v1MTLIP.cpp v1ColorME.cu -o v1MTLIP

v1v4PFC: ${SOURCES} ${DEP} examples/v1v4PFC/main_v1v4PFC.cpp v1ColorMEold.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib ${LIBCUTIL} -D__CUDA3__ -arch sm_13 ${SOURCES} examples/v1v4PFC/main_v1v4PFC.cpp v1ColorMEold.cu -o v1v4PFC
