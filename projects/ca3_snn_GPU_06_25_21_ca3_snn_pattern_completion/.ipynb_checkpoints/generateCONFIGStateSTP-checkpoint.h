int CA3_LMR_Targeting = sim.createGroup("CA3_LMR_Targeting", 1099.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_QuadD_LM = sim.createGroup("CA3_QuadD_LM", 415.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 671.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Basket = sim.createGroup("CA3_Basket", 500.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_BC_CCK = sim.createGroup("CA3_BC_CCK", 658.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Bistratified = sim.createGroup("CA3_Bistratified", 861.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Ivy = sim.createGroup("CA3_Ivy", 43.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Trilaminar = sim.createGroup("CA3_Trilaminar", 8601.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 135.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 74389.0,
                              EXCITATORY_NEURON, 0, GPU_CORES);
                              
sim.setNeuronParameters(CA3_LMR_Targeting, 54.0, 0.0, 1.103614344, 0,
                                                -67.7253072, 0.0, -24.08892339, 0.0, 0.000994,
                                                0.0, -61.26447822, 0.0, 10.38028985,
                                                0.0, -42.7378584, 0.0,
                                                43.0, 0.0);
                     
sim.setNeuronParameters(CA3_QuadD_LM, 186.0, 0.0, 1.77600861583782, 0,
                                                -73.4821116922868, 0.0, -54.9369058996129, 0.0, 0.00584332072216318,
                                                0.0, -3.44873648365723, 0.0, 7.06631328236041,
                                                0.0, -64.4037157222031, 0.0,
                                                52.0, 0.0);
                     
sim.setNeuronParameters(CA3_Axo_Axonic, 165.0, 0.0, 3.96146287759279, 0,
                                                -57.099782869594, 0.0, -51.7187562820223, 0.0, 0.00463860807187154,
                                                0.0, 8.68364493653417, 0.0, 27.7986355932787,
                                                0.0, -73.9685042125372, 0.0,
                                                15.0, 0.0);
                     
sim.setNeuronParameters(CA3_Basket, 45.0, 0.0, 0.9951729, 0,
                                                -57.506126, 0.0, -23.378766, 0.0, 0.003846186,
                                                0.0, 9.2642765, 0.0, 18.454934,
                                                0.0, -47.555661, 0.0,
                                                -6.0, 0.0);
                     
sim.setNeuronParameters(CA3_BC_CCK, 135.0, 0.0, 0.583005186, 0,
                                                -58.99667734, 0.0, -39.39832097, 0.0, 0.00574483,
                                                0.0, -1.244845715, 0.0, 18.27458854,
                                                0.0, -42.7711851, 0.0,
                                                54.0, 0.0);
                     
sim.setNeuronParameters(CA3_Bistratified, 107.0, 0.0, 3.935030495, 0,
                                                -64.67262808, 0.0, -58.74397154, 0.0, 0.001952449,
                                                0.0, 16.57957046, 0.0, -9.928793958,
                                                0.0, -59.70326258, 0.0,
                                                19.0, 0.0);
                     
sim.setNeuronParameters(CA3_Ivy, 364.0, 0.0, 1.91603822942046, 0,
                                                -70.4345135750261, 0.0, -40.8589263758355, 0.0, 0.009151158130158,
                                                0.0, 1.90833702318966, 0.0, -6.91973671560226,
                                                0.0, -53.3998503336009, 0.0,
                                                45.0, 0.0);
                     
sim.setNeuronParameters(CA3_Trilaminar, 251.0, 0.0, 0.930715938803403, 0,
                                                -63.1326022655518, 0.0, -55.6396712684413, 0.0, 0.000441258674088799,
                                                0.0, -18.7576541248722, 0.0, 17.0068586049703,
                                                0.0, -52.6177192420144, 0.0,
                                                74.0, 0.0);
                     
sim.setNeuronParameters(CA3_MFA_ORDEN, 209.0, 0.0, 1.37980713457205, 0,
                                                -57.076423571379, 0.0, -39.1020427841762, 0.0, 0.00783805979364104,
                                                0.0, 12.9332855397722, 0.0, 16.3132681887705,
                                                0.0, -40.6806648852695, 0.0,
                                                0.0, 0.0);
                     
sim.setNeuronParameters(CA3_Pyramidal, 366.0, 0.0, 0.792338703789581, 0,
                                                -63.2044008171655, 0.0, -33.6041733124267, 0.0, 0.00838350334098279,
                                                0.0, -42.5524776883928, 0.0, 35.8614648558726,
                                                0.0, -38.8680990294091, 0.0,
                                                588.0, 0.0);
                     
sim.connect(CA3_LMR_Targeting, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1f, 2.0f), 0.09293346755589336f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.532175289f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.00198176910683873f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.853173174f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.06120334840172132f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.157262353f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.06886172965035817f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.915319854f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.053727881406546704f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.636826932f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.003119978345877652f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.261849976f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Ivy, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0037527187005270226f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.636833939f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.00354607713195609f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.912538909f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0720967942651253f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.453369267f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0845151502137485f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.808425786f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.08914817647808601f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.435590533f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.10301783949200871f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.243828988f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.06724984842486549f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.683531036f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.11882477878628f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.551854362f, 0.0f);
                                       
sim.connect(CA3_Axo_Axonic, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.015f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.644261648f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.713553453f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.791348857f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 4.061631276f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.359227353f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.135048131f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.01f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.590947878f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.015f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.219421572f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.707546905f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.708213783f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.567591248f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.502536236f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.064846506f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.686699431f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.015f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.428510448f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1f, 2.0f), 0.009132361030652542f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.240526844f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0280863730109791f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.607538323f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03239322987696437f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.647204471f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03175234779094391f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.600565008f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.024401738130525888f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.512926018f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.03257852082077668f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.910823391f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Ivy, "random", RangeWeight(0.0f, 1f, 2.0f), 0.023036995573319972f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.497423138f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Trilaminar, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0176805299769247f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.458291731f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0327445340521029f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.65833298f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0278686093547943f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.35829133f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1f, 2.0f), 0.005345891855303468f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.911396994f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0150318717716677f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.098330268f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.01675141468120584f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.44606939f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.016300617068683923f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.382462716f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.010742278540606838f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.187767719f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.01703156100118382f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.082025068f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Ivy, "random", RangeWeight(0.0f, 1f, 2.0f), 0.009774873774462175f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.039964505f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Trilaminar, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0127828148201104f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.379810422f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0169285421706311f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.138762321f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0136019039233742f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.919023303f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1f, 2.0f), 0.00809719719382231f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.593825841f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0275947109307687f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.994803494f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.022591416547506484f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.986305421f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.022204873915356138f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.999298414f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.01619942522378886f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.822856489f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.02288864081313128f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.190768694f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_Ivy, "random", RangeWeight(0.0f, 1f, 2.0f), 0.015107005305090272f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.505691098f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_Trilaminar, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0168899430303076f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.894212283f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0244999224201649f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.948625008f, 0.0f);
                                       
sim.connect(CA3_Trilaminar, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0192358752949603f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.658146232f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1f, 2.0f), 0.017385797155141058f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.715436388f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0349029677603613f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 4.169599367f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.06362608114049711f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 5.558918905f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.05992576040511249f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 4.228090316f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.039686162634668426f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.306807773f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.06487491202885205f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.721738064f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Ivy, "random", RangeWeight(0.0f, 1f, 2.0f), 0.038204625456942344f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.783020174f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0474702019783161f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 4.02104594f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0417555599977689f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.375850669f, 0.0f);
                                       
sim.connect(CA3_Pyramidal, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1f, 2.0f), 0.00206533635093884f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.615513653f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0133672243607345f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.783552885f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0197801419188772f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.840844055f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Basket, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0197417562762975f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.700200939f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_BC_CCK, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0172994236281402f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.599853514f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Bistratified, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0203823774703507f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.32170445f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Ivy, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0125783953956972f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.479679571f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Trilaminar, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0163947446999513f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.789924513f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0209934225689348f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.85563014f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Pyramidal, "random", RangeWeight(0.0f, 1f, 2.0f), 0.0250664662231983f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 2.368715389f, 0.0f);
                                   
sim.setSTP(CA3_LMR_Targeting, CA3_LMR_Targeting, true, STPu(0.102218328f, 0.0f),
                                         STPtauU(66.57693323f, 0.0f),
                                         STPtauX(508.331419f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(12.27059944f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_QuadD_LM, true, STPu(0.153890489f, 0.0f),
                                         STPtauU(65.18673165f, 0.0f),
                                         STPtauX(429.843055f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(12.0464793f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Axo_Axonic, true, STPu(0.180888581f, 0.0f),
                                         STPtauU(51.3279926f, 0.0f),
                                         STPtauX(550.9949908f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.81716065f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Basket, true, STPu(0.130889563f, 0.0f),
                                         STPtauU(25.92482599f, 0.0f),
                                         STPtauX(641.8299594f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.823262941f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_BC_CCK, true, STPu(0.14914348f, 0.0f),
                                         STPtauU(51.56911788f, 0.0f),
                                         STPtauX(527.7664356f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.14299646f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Bistratified, true, STPu(0.070774287f, 0.0f),
                                         STPtauU(23.79219334f, 0.0f),
                                         STPtauX(591.537059f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.47037695f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Ivy, true, STPu(0.147154877f, 0.0f),
                                         STPtauU(20.44877538f, 0.0f),
                                         STPtauX(850.2357302f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.65219486f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_MFA_ORDEN, true, STPu(0.178577768f, 0.0f),
                                         STPtauU(67.01256464f, 0.0f),
                                         STPtauX(385.6138994f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.35966736f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Pyramidal, true, STPu(0.141749854f, 0.0f),
                                         STPtauU(66.64159269f, 0.0f),
                                         STPtauX(363.0553799f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(13.34728338f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_LMR_Targeting, true, STPu(0.152046951f, 0.0f),
                                         STPtauU(54.70579097f, 0.0f),
                                         STPtauX(504.4469131f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(12.82096169f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Axo_Axonic, true, STPu(0.227291526f, 0.0f),
                                         STPtauU(39.73410017f, 0.0f),
                                         STPtauX(525.2333483f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.19992774f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Basket, true, STPu(0.174447661f, 0.0f),
                                         STPtauU(23.72875566f, 0.0f),
                                         STPtauX(634.710816f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.14329235f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_BC_CCK, true, STPu(0.17682376f, 0.0f),
                                         STPtauU(39.96207846f, 0.0f),
                                         STPtauX(483.8834589f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.56150272f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Pyramidal, true, STPu(0.188797554f, 0.0f),
                                         STPtauU(50.75642594f, 0.0f),
                                         STPtauX(331.3275611f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(13.675978f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Axo_Axonic, CA3_Pyramidal, true, STPu(0.259914361f, 0.0f),
                                         STPtauU(17.20004939f, 0.0f),
                                         STPtauX(435.8103009f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.71107251f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_QuadD_LM, true, STPu(0.229910977f, 0.0f),
                                         STPtauU(18.25210028f, 0.0f),
                                         STPtauX(620.0839267f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.923152659f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Axo_Axonic, true, STPu(0.259550011f, 0.0f),
                                         STPtauU(21.55840998f, 0.0f),
                                         STPtauX(728.7609299f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.179717354f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Basket, true, STPu(0.202909789f, 0.0f),
                                         STPtauU(16.91556274f, 0.0f),
                                         STPtauX(712.0540919f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.600964509f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_BC_CCK, true, STPu(0.215686662f, 0.0f),
                                         STPtauU(15.50819894f, 0.0f),
                                         STPtauX(748.3233594f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.661495263f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Bistratified, true, STPu(0.116935135f, 0.0f),
                                         STPtauU(9.936730429f, 0.0f),
                                         STPtauX(610.8062676f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.316993527f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_MFA_ORDEN, true, STPu(0.239996508f, 0.0f),
                                         STPtauU(22.06307668f, 0.0f),
                                         STPtauX(584.4592928f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.76215774f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Pyramidal, true, STPu(0.216204016f, 0.0f),
                                         STPtauU(17.52835258f, 0.0f),
                                         STPtauX(533.5284513f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.34964374f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_QuadD_LM, true, STPu(0.196788544f, 0.0f),
                                         STPtauU(22.82022867f, 0.0f),
                                         STPtauX(779.6541624f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.2959873f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Axo_Axonic, true, STPu(0.187726511f, 0.0f),
                                         STPtauU(22.95739819f, 0.0f),
                                         STPtauX(831.280778f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.05633843f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Basket, true, STPu(0.131089615f, 0.0f),
                                         STPtauU(19.10146603f, 0.0f),
                                         STPtauX(696.9988518f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.643109947f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_BC_CCK, true, STPu(0.157336639f, 0.0f),
                                         STPtauU(17.34413771f, 0.0f),
                                         STPtauX(917.3953648f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.66589323f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Bistratified, true, STPu(0.061779984f, 0.0f),
                                         STPtauU(12.2147064f, 0.0f),
                                         STPtauX(507.334051f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.474339813f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_MFA_ORDEN, true, STPu(0.207573084f, 0.0f),
                                         STPtauU(27.72250844f, 0.0f),
                                         STPtauX(749.8373852f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.06113787f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Pyramidal, true, STPu(0.203567174f, 0.0f),
                                         STPtauU(25.03255541f, 0.0f),
                                         STPtauX(653.5301443f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(12.69947786f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_LMR_Targeting, true, STPu(0.122948332f, 0.0f),
                                         STPtauU(37.28962162f, 0.0f),
                                         STPtauX(608.2464892f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.4708284f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_QuadD_LM, true, STPu(0.172951581f, 0.0f),
                                         STPtauU(41.54957197f, 0.0f),
                                         STPtauX(569.7892129f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.30947514f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Axo_Axonic, true, STPu(0.181065594f, 0.0f),
                                         STPtauU(34.29954214f, 0.0f),
                                         STPtauX(659.8911124f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.44073202f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Basket, true, STPu(0.129240835f, 0.0f),
                                         STPtauU(22.64080308f, 0.0f),
                                         STPtauX(640.2428024f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.665705274f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_BC_CCK, true, STPu(0.156903502f, 0.0f),
                                         STPtauU(39.2576303f, 0.0f),
                                         STPtauX(605.6495433f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.38837176f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Bistratified, true, STPu(0.042258508f, 0.0f),
                                         STPtauU(10.79281828f, 0.0f),
                                         STPtauX(431.6435473f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.069849106f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Ivy, true, STPu(0.154671807f, 0.0f),
                                         STPtauU(21.03208306f, 0.0f),
                                         STPtauX(848.7570227f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.962538125f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Trilaminar, true, STPu(0.116408248f, 0.0f),
                                         STPtauU(22.492265f, 0.0f),
                                         STPtauX(496.2191667f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.6983811f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_MFA_ORDEN, true, STPu(0.188516356f, 0.0f),
                                         STPtauU(49.45405723f, 0.0f),
                                         STPtauX(515.6774021f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.8227429f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Pyramidal, true, STPu(0.173165863f, 0.0f),
                                         STPtauU(39.75945564f, 0.0f),
                                         STPtauX(497.9877062f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.70630159f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_LMR_Targeting, true, STPu(0.017980751f, 0.0f),
                                         STPtauU(8.587541043f, 0.0f),
                                         STPtauX(71.28807307f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.44452431f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_QuadD_LM, true, STPu(0.054801761f, 0.0f),
                                         STPtauU(20.24614139f, 0.0f),
                                         STPtauX(134.3179824f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.136610859f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Axo_Axonic, true, STPu(0.042875836f, 0.0f),
                                         STPtauU(13.58698624f, 0.0f),
                                         STPtauX(93.06277339f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.197848812f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Basket, true, STPu(0.018759865f, 0.0f),
                                         STPtauU(6.097950156f, 0.0f),
                                         STPtauX(73.37723543f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.10502528f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_BC_CCK, true, STPu(0.093226807f, 0.0f),
                                         STPtauU(21.53090397f, 0.0f),
                                         STPtauX(316.402153f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.86801419f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Bistratified, true, STPu(0.010497385f, 0.0f),
                                         STPtauU(3.167714285f, 0.0f),
                                         STPtauX(35.10660443f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.061782985f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Ivy, true, STPu(0.059234373f, 0.0f),
                                         STPtauU(12.53347413f, 0.0f),
                                         STPtauX(355.5423068f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.868419767f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Trilaminar, true, STPu(0.016348367f, 0.0f),
                                         STPtauU(5.379474386f, 0.0f),
                                         STPtauX(59.31411695f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.045296287f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_MFA_ORDEN, true, STPu(0.053176879f, 0.0f),
                                         STPtauU(21.05536743f, 0.0f),
                                         STPtauX(121.0601087f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.578761219f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Pyramidal, true, STPu(0.045297728f, 0.0f),
                                         STPtauU(18.28338527f, 0.0f),
                                         STPtauX(114.5641985f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.14648277f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_LMR_Targeting, true, STPu(0.146654936f, 0.0f),
                                         STPtauU(17.64852105f, 0.0f),
                                         STPtauX(754.5615069f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(13.11516333f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_QuadD_LM, true, STPu(0.199917672f, 0.0f),
                                         STPtauU(19.95673334f, 0.0f),
                                         STPtauX(735.2090137f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(12.39083767f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_Axo_Axonic, true, STPu(0.207010531f, 0.0f),
                                         STPtauU(21.77781735f, 0.0f),
                                         STPtauX(850.4583346f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.24488382f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_Basket, true, STPu(0.155568101f, 0.0f),
                                         STPtauU(18.39933858f, 0.0f),
                                         STPtauX(782.8465541f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.96631573f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_BC_CCK, true, STPu(0.176791527f, 0.0f),
                                         STPtauU(16.02320263f, 0.0f),
                                         STPtauX(848.480161f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.68358918f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_Bistratified, true, STPu(0.085752021f, 0.0f),
                                         STPtauU(11.47529511f, 0.0f),
                                         STPtauX(729.7519546f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.43078517f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_Ivy, true, STPu(0.17773949f, 0.0f),
                                         STPtauU(19.01110146f, 0.0f),
                                         STPtauX(946.4011299f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(12.12434366f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_Trilaminar, true, STPu(0.14706638f, 0.0f),
                                         STPtauU(18.30484384f, 0.0f),
                                         STPtauX(640.1235581f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.43258895f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_MFA_ORDEN, true, STPu(0.216108872f, 0.0f),
                                         STPtauU(24.65039949f, 0.0f),
                                         STPtauX(705.071041f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(12.14533121f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Trilaminar, CA3_Pyramidal, true, STPu(0.192964925f, 0.0f),
                                         STPtauU(20.34984037f, 0.0f),
                                         STPtauX(630.1862229f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(13.71240857f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_LMR_Targeting, true, STPu(0.115540787f, 0.0f),
                                         STPtauU(63.59574072f, 0.0f),
                                         STPtauX(720.1229791f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(10.02937439f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_QuadD_LM, true, STPu(0.170840239f, 0.0f),
                                         STPtauU(63.65929616f, 0.0f),
                                         STPtauX(673.2589246f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.908508332f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Axo_Axonic, true, STPu(0.185743417f, 0.0f),
                                         STPtauU(47.81236753f, 0.0f),
                                         STPtauX(787.8075387f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.79316684f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Basket, true, STPu(0.123564968f, 0.0f),
                                         STPtauU(23.25849737f, 0.0f),
                                         STPtauX(782.0880356f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.051224933f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_BC_CCK, true, STPu(0.164267618f, 0.0f),
                                         STPtauU(45.40855401f, 0.0f),
                                         STPtauX(760.26965f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.286459094f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Bistratified, true, STPu(0.052837f, 0.0f),
                                         STPtauU(17.241103f, 0.0f),
                                         STPtauX(619.649148f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(8.061113457f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Ivy, true, STPu(0.146934947f, 0.0f),
                                         STPtauU(21.01255478f, 0.0f),
                                         STPtauX(935.2711213f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.939570581f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_MFA_ORDEN, true, STPu(0.193620378f, 0.0f),
                                         STPtauU(67.39183116f, 0.0f),
                                         STPtauX(601.714189f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.315215294f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Pyramidal, true, STPu(0.157847979f, 0.0f),
                                         STPtauU(65.61972279f, 0.0f),
                                         STPtauX(549.3580329f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(11.2564576f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Pyramidal, CA3_LMR_Targeting, true, STPu(0.130176816f, 0.0f),
                                     STPtauU(37.68735406f, 0.0f),
                                     STPtauX(728.8787787f, 0.0f),
                                     STPtdAMPA(8.970320044f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_QuadD_LM, true, STPu(0.170544117f, 0.0f),
                                     STPtauU(37.89155876f, 0.0f),
                                     STPtauX(640.2646697f, 0.0f),
                                     STPtdAMPA(8.817565535f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Axo_Axonic, true, STPu(0.17816195f, 0.0f),
                                     STPtauU(33.42820332f, 0.0f),
                                     STPtauX(817.0342011f, 0.0f),
                                     STPtdAMPA(8.167547377f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Basket, true, STPu(0.137496865f, 0.0f),
                                     STPtauU(25.73406064f, 0.0f),
                                     STPtauX(864.5013776f, 0.0f),
                                     STPtdAMPA(7.678347856f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_BC_CCK, true, STPu(0.15118797f, 0.0f),
                                     STPtauU(28.32182342f, 0.0f),
                                     STPtauX(735.1808146f, 0.0f),
                                     STPtdAMPA(8.156513136f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Bistratified, true, STPu(0.105352106f, 0.0f),
                                     STPtauU(21.2664278f, 0.0f),
                                     STPtauX(944.132605f, 0.0f),
                                     STPtdAMPA(8.357584961f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Ivy, true, STPu(0.156189969f, 0.0f),
                                     STPtauU(22.85063359f, 0.0f),
                                     STPtauX(999.0103599f, 0.0f),
                                     STPtdAMPA(8.466011695f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Trilaminar, true, STPu(0.15681333f, 0.0f),
                                     STPtauU(25.9420068f, 0.0f),
                                     STPtauX(695.5097455f, 0.0f),
                                     STPtdAMPA(8.630718152f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_MFA_ORDEN, true, STPu(0.19845995f, 0.0f),
                                     STPtauU(45.21337748f, 0.0f),
                                     STPtauX(547.766236f, 0.0f),
                                     STPtdAMPA(8.286345583f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Pyramidal, true, STPu(0.177173426f, 0.0f),
                                     STPtauU(39.41409758f, 0.0f),
                                     STPtauX(493.6195595f, 0.0f),
                                     STPtdAMPA(10.81725865f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
