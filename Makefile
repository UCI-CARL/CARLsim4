random:    
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib/ -lcutil -arch sm_13 snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu examples/main_random.cpp

v1v4PFC:
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu examples/main_v1v4PFC.cpp v1ColorME.cu

colorcycle:
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu examples/main_colorcycle.cpp v1ColorME.cu

orientation:
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu examples/main_orientation.cpp v1ColorME.cu

colorblind:
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu examples/main_colorblind.cpp v1ColorME.cu

rdk:
	nvcc -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 snn_cpu.cpp mtrand.cpp PropagatedSpikeBuffer.cpp printSNNInfo.cpp gpu_random.cu snn_gpu.cu examples/main_rdk.cpp v1ColorME.cu
