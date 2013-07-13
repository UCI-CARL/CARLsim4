# CARLsim Makefile
# CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
# Ver 07/13/2013

# source files of the sim core
SOURCES = snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNinfo.cpp gpu_random.cu snn_gpu.cu

colorblind: ${SOURCES} examples/main_colorblind.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 ${SOURCES} examples/main_colorblind.cpp v1ColorME.cu -o colorblind

colorcycle: ${SOURCES} examples/main_colorcycle.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 ${SOURCES} examples/main_colorcycle.cpp v1ColorME.cu -o colorblind

orientation: ${SOURCES} examples/main_orientation.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 ${SOURCES} examples/main_orientation.cpp v1ColorME.cu -o colorblind

random: ${SOURCES} examples/main_random.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 ${SOURCES} examples/main_random.cpp v1ColorME.cu -o colorblind

rdk: ${SOURCES} examples/main_rdk.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 ${SOURCES} examples/main_rdk.cpp v1ColorME.cu -o colorblind

v1MTLIP: ${SOURCES} examples/main_v1MTLIP.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 ${SOURCES} examples/main_v1MTLIP.cpp v1ColorME.cu -o colorblind

v1v4PFC: ${SOURCES} examples/main_v1v4PFC.cpp v1ColorME.cu
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 ${SOURCES} examples/main_v1v4PFC.cpp v1ColorME.cu -o colorblind