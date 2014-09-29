# module include makefile so CARLsim file can delete objects created by this program
izk_deps := ../izk/network/Network.c ../izk/network/ConnectionGroup.c ../izk/network/CustomConnectionScheme.c ../izk/network/FullConnectionScheme.c ../izk/network/RandomConnectionScheme.c ../izk/network/NeuronGroup.c ../izk/neuron/Neuron.c ../izk/neuron/Izhikevich4Neuron.c ../izk/neuron/PoissonNeuron.c
izk_objs := $(izk_deps:.c=.o)

izk_build_files := $(izk_objs) ../izk/libizk.a
