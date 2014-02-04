
#Simulator source code
#------------------------------------------------------------
snn_cpu.o: snn_cpu.cpp
	nvcc -c -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 snn_cpu.cpp -o snn_cpu.o 

mtrand.o: mtrand.cpp
	g++ -c mtrand.cpp -o mtrand.o

PropagatedSpikeBuffer.o: PropagatedSpikeBuffer.cpp
	g++ -c PropagatedSpikeBuffer.cpp -o PropagatedSpikeBuffer.o

printSNNInfo.o: printSNNInfo.cpp
	nvcc -c -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 printSNNInfo.cpp -o printSNNInfo.o

gpu_random.o: gpu_random.cu
	nvcc -c -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 gpu_random.cu -o gpu_random.o

snn_gpu.o: snn_gpu.cu
	nvcc -c -I${NVIDIA_SDK}/C/common/inc/ -L${NVIDIA_SDK}/C/lib -lcutil -arch sm_13 snn_gpu.cu -o snn_gpu.o

#Various clean functions
#--------------------------------------------------------------

#distclean deletes all object files and data files so a complete recompile is required.
clean: 
	rm -rf *.o 
