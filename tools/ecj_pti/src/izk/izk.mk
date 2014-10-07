# module include makefile so CARLsim file can delete objects created by this program
izk_deps := $(iz_dir)/network/Network.c $(iz_dir)/network/ConnectionGroup.c $(iz_dir)/network/CustomConnectionScheme.c $(iz_dir)/network/FullConnectionScheme.c $(iz_dir)/network/RandomConnectionScheme.c $(iz_dir)/network/NeuronGroup.c $(iz_dir)/neuron/Neuron.c $(iz_dir)/neuron/Izhikevich4Neuron.c $(iz_dir)/neuron/PoissonNeuron.c
izk_objs := $(izk_deps:.c=.o)

izk_build_files := $(izk_objs) $(iz_dir)/libizk.a
