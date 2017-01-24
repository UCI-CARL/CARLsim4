# module include makefile so CARLsim file can delete objects created by this program
izk_deps := $(izk_dir)/network/Network.c $(izk_dir)/network/ConnectionGroup.c $(izk_dir)/network/CustomConnectionScheme.c $(izk_dir)/network/FullConnectionScheme.c $(izk_dir)/network/RandomConnectionScheme.c $(izk_dir)/network/NeuronGroup.c $(izk_dir)/neuron/Neuron.c $(izk_dir)/neuron/Izhikevich4Neuron.c $(izk_dir)/neuron/PoissonNeuron.c
izk_objs := $(izk_deps:.c=.o)

izk_build_files := $(izk_objs) $(izk_dir)/libizk.a
