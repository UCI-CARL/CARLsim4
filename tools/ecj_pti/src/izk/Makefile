izk_deps := network/Network.c network/ConnectionGroup.c network/CustomConnectionScheme.c network/FullConnectionScheme.c network/RandomConnectionScheme.c network/NeuronGroup.c neuron/Neuron.c neuron/Izhikevich4Neuron.c neuron/PoissonNeuron.c
izk_objs := $(izk_deps:.c=.o)

izk: libizk.a IzkExample
	
%.o: %.c $(izk_deps)
	gcc -g -c $< -o $@
libizk.a: $(izk_objs)
	ar rcs $@ $(izk_objs)
	
IzkExample: IzkExample.c libizk.a
	gcc -g $< libizk.a -o $@ -lm
clean:
	rm $(izk_objs) IzkExample libizk.a
