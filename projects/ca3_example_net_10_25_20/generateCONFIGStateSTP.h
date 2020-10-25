int CA3_LMR_Targeting = sim.createGroup("CA3_LMR_Targeting", 107.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_QuadD_LM = sim.createGroup("CA3_QuadD_LM", 328.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 190.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Basket = sim.createGroup("CA3_Basket", 51.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_BC_CCK = sim.createGroup("CA3_BC_CCK", 66.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Bistratified = sim.createGroup("CA3_Bistratified", 58.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Ivy = sim.createGroup("CA3_Ivy", 43.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 152.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 743.0,
                              EXCITATORY_NEURON, 0, GPU_CORES);
                              
sim.setNeuronParameters(CA3_LMR_Targeting, 54.0, 0.0, 1.103614344, 0,
                                                -67.7253072, 0.0, -24.08892339, 0.0, 0.000994,
                                                0.0, -61.26447822, 0.0, 10.38028985,
                                                0.0, -42.7378584, 0.0,
                                                43.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_QuadD_LM, 186.0, 0.0, 1.77600861583782, 0,
                                                -73.4821116922868, 0.0, -54.9369058996129, 0.0, 0.00584332072216318,
                                                0.0, -3.44873648365723, 0.0, 7.06631328236041,
                                                0.0, -64.4037157222031, 0.0,
                                                52.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_Axo_Axonic, 165.0, 0.0, 3.96146287759279, 0,
                                                -57.099782869594, 0.0, -51.7187562820223, 0.0, 0.00463860807187154,
                                                0.0, 8.68364493653417, 0.0, 27.7986355932787,
                                                0.0, -73.9685042125372, 0.0,
                                                15.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_Basket, 45.0, 0.0, 0.9951729, 0,
                                                -57.506126, 0.0, -23.378766, 0.0, 0.003846186,
                                                0.0, 9.2642765, 0.0, 18.454934,
                                                0.0, -47.555661, 0.0,
                                                -6.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_BC_CCK, 135.0, 0.0, 0.583005186, 0,
                                                -58.99667734, 0.0, -39.39832097, 0.0, 0.00574483,
                                                0.0, -1.244845715, 0.0, 18.27458854,
                                                0.0, -42.7711851, 0.0,
                                                54.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_Bistratified, 107.0, 0.0, 3.935030495, 0,
                                                -64.67262808, 0.0, -58.74397154, 0.0, 0.001952449,
                                                0.0, 16.57957046, 0.0, -9.928793958,
                                                0.0, -59.70326258, 0.0,
                                                19.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_Ivy, 364.0, 0.0, 1.91603822942046, 0,
                                                -70.4345135750261, 0.0, -40.8589263758355, 0.0, 0.009151158130158,
                                                0.0, 1.90833702318966, 0.0, -6.91973671560226,
                                                0.0, -53.3998503336009, 0.0,
                                                45.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_MFA_ORDEN, 209.0, 0.0, 1.37980713457205, 0,
                                                -57.076423571379, 0.0, -39.1020427841762, 0.0, 0.00783805979364104,
                                                0.0, 12.9332855397722, 0.0, 16.3132681887705,
                                                0.0, -40.6806648852695, 0.0,
                                                0.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_Pyramidal, 366.0, 0.0, 0.792338703789581, 0,
                                                -63.2044008171655, 0.0, -33.6041733124267, 0.0, 0.00838350334098279,
                                                0.0, -42.5524776883928, 0.0, 35.8614648558726,
                                                0.0, -38.8680990294091, 0.0,
                                                588.0, 0.0, 1);
                     
sim.connect(CA3_LMR_Targeting, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.09293346755589336f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.272970021f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00198176910683873f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.165772827f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.06120334840172132f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.314644754f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.06886172965035817f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.670292951f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.053727881406546704f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.180464832f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Bistratified, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.003119978345877652f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.248292203f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Ivy, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0037527187005270226f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.713627115f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00354607713195609f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.222129972f, 0.0f);
                                       
sim.connect(CA3_LMR_Targeting, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0720967942651253f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 0.995517226f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0845151502137485f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.422489311f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.08914817647808601f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.473054473f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.10301783949200871f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.815288206f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.06724984842486549f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.308152788f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.11882477878628f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.183249644f, 0.0f);
                                       
sim.connect(CA3_Axo_Axonic, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.015f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.869561088f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.750808378f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.02478806f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.281611994f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.686238357f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Bistratified, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.770407646f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.01f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.808726221f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.015f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.572405696f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.334971075f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.493830393f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.745239175f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 0.965567683f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Bistratified, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.371483576f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.355214118f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.015f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.306303671f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.009132361030652542f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.622700161f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0280863730109791f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.490477376f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03239322987696437f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.655173699f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03175234779094391f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.99431849f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.024401738130525888f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.442647868f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Bistratified, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.03257852082077668f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.547036443f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Ivy, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.023036995573319972f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.061179756f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0327445340521029f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.567727407f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0278686093547943f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.431148109f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005345891855303468f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.715424042f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0150318717716677f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.567261924f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.01675141468120584f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.758382851f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.016300617068683923f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.111359644f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.010742278540606838f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.54009769f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Bistratified, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.01703156100118382f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.660111909f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Ivy, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.009774873774462175f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.142741525f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0169285421706311f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.687214907f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0136019039233742f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.540661646f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.017385797155141058f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.570007077f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0349029677603613f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.471995564f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.06362608114049711f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.628518636f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.05992576040511249f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.972333716f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.039686162634668426f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.415044453f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Bistratified, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.06487491202885205f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.536494845f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Ivy, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.038204625456942344f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.081976802f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0474702019783161f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.552656079f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0417555599977689f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.360315289f, 0.0f);
                                       
sim.connect(CA3_Pyramidal, CA3_LMR_Targeting, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00206533635093884f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 0.920761083f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0133672243607345f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 0.874424964f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0197801419188772f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 0.932621138f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0197417562762975f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 1.172460639f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0172994236281402f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 0.847532877f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Bistratified, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0203823774703507f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 0.883682094f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Ivy, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0125783953956972f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 1.314331038f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0209934225689348f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 0.88025265f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0250664662231983f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, 0.553062478f, 0.0f);
                                   
sim.setSTP(CA3_LMR_Targeting, CA3_LMR_Targeting, true, STPu(0.180887108f, 0.0f),
                                         STPtauU(25.36700667f, 0.0f),
                                         STPtauX(616.1615903f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.401901155f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_QuadD_LM, true, STPu(0.187982165f, 0.0f),
                                         STPtauU(24.76630928f, 0.0f),
                                         STPtauX(565.2542465f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.598700398f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Axo_Axonic, true, STPu(0.193262347f, 0.0f),
                                         STPtauU(24.06662425f, 0.0f),
                                         STPtauX(667.9216703f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.369185719f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Basket, true, STPu(0.212408847f, 0.0f),
                                         STPtauU(17.22154219f, 0.0f),
                                         STPtauX(675.4733154f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.448164795f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_BC_CCK, true, STPu(0.187083044f, 0.0f),
                                         STPtauU(19.39871323f, 0.0f),
                                         STPtauX(616.7020115f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.96606175f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Bistratified, true, STPu(0.196224293f, 0.0f),
                                         STPtauU(20.4330606f, 0.0f),
                                         STPtauX(668.3485536f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.888205882f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Ivy, true, STPu(0.181173458f, 0.0f),
                                         STPtauU(15.5120885f, 0.0f),
                                         STPtauX(696.283731f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.058393836f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_MFA_ORDEN, true, STPu(0.181805939f, 0.0f),
                                         STPtauU(26.23291354f, 0.0f),
                                         STPtauX(573.427672f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.627208525f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_LMR_Targeting, CA3_Pyramidal, true, STPu(0.147893166f, 0.0f),
                                         STPtauU(26.56605695f, 0.0f),
                                         STPtauX(453.1318699f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.563225249f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_LMR_Targeting, true, STPu(0.207021564f, 0.0f),
                                         STPtauU(22.87045361f, 0.0f),
                                         STPtauX(591.964003f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.127949766f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Axo_Axonic, true, STPu(0.21783904f, 0.0f),
                                         STPtauU(22.34022321f, 0.0f),
                                         STPtauX(635.0122846f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.168513359f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Basket, true, STPu(0.231900972f, 0.0f),
                                         STPtauU(16.42005923f, 0.0f),
                                         STPtauX(663.2470985f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.293130764f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_BC_CCK, true, STPu(0.207810376f, 0.0f),
                                         STPtauU(17.78433468f, 0.0f),
                                         STPtauX(596.5048064f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.830825741f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Pyramidal, true, STPu(0.193598093f, 0.0f),
                                         STPtauU(24.78801373f, 0.0f),
                                         STPtauX(382.1372881f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.107305544f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Axo_Axonic, CA3_Pyramidal, true, STPu(0.236806614f, 0.0f),
                                         STPtauU(12.93029701f, 0.0f),
                                         STPtauX(361.0287219f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.623472774f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_QuadD_LM, true, STPu(0.245404293f, 0.0f),
                                         STPtauU(19.30519205f, 0.0f),
                                         STPtauX(589.2002267f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.162447367f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Axo_Axonic, true, STPu(0.272878555f, 0.0f),
                                         STPtauU(23.20689302f, 0.0f),
                                         STPtauX(725.0291555f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.800283876f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Basket, true, STPu(0.268625017f, 0.0f),
                                         STPtauU(11.19042564f, 0.0f),
                                         STPtauX(689.5059466f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.007016545f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_BC_CCK, true, STPu(0.235943711f, 0.0f),
                                         STPtauU(16.71506093f, 0.0f),
                                         STPtauX(636.7647263f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.20676528f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Bistratified, true, STPu(0.250623533f, 0.0f),
                                         STPtauU(16.7158668f, 0.0f),
                                         STPtauX(680.3279514f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.720497985f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_MFA_ORDEN, true, STPu(0.24148518f, 0.0f),
                                         STPtauU(19.60369075f, 0.0f),
                                         STPtauX(581.9355018f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.230610278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Pyramidal, true, STPu(0.227671739f, 0.0f),
                                         STPtauU(16.73589406f, 0.0f),
                                         STPtauX(384.3363321f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.63862234f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_QuadD_LM, true, STPu(0.174996467f, 0.0f),
                                         STPtauU(17.34307333f, 0.0f),
                                         STPtauX(398.1521614f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.480095841f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Axo_Axonic, true, STPu(0.176470108f, 0.0f),
                                         STPtauU(18.502335f, 0.0f),
                                         STPtauX(477.4286298f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.43847192f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Basket, true, STPu(0.195589215f, 0.0f),
                                         STPtauU(14.85568316f, 0.0f),
                                         STPtauX(505.1237763f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.688537876f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_BC_CCK, true, STPu(0.118630661f, 0.0f),
                                         STPtauU(23.37589368f, 0.0f),
                                         STPtauX(283.28138f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.886667725f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Bistratified, true, STPu(0.180014988f, 0.0f),
                                         STPtauU(15.24644944f, 0.0f),
                                         STPtauX(478.3138872f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.9732847f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_MFA_ORDEN, true, STPu(0.1684458f, 0.0f),
                                         STPtauU(17.83862927f, 0.0f),
                                         STPtauX(421.419837f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.538700998f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Pyramidal, true, STPu(0.145898121f, 0.0f),
                                         STPtauU(13.7603965f, 0.0f),
                                         STPtauX(376.8743946f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.101015633f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_LMR_Targeting, true, STPu(0.230491012f, 0.0f),
                                         STPtauU(18.173225f, 0.0f),
                                         STPtauX(649.8282077f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.330598235f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_QuadD_LM, true, STPu(0.23618509f, 0.0f),
                                         STPtauU(17.89475495f, 0.0f),
                                         STPtauX(594.3278762f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.529521921f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Axo_Axonic, true, STPu(0.239224482f, 0.0f),
                                         STPtauU(19.16221108f, 0.0f),
                                         STPtauX(686.2791734f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.568545011f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Basket, true, STPu(0.252137216f, 0.0f),
                                         STPtauU(14.59835713f, 0.0f),
                                         STPtauX(695.2132647f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.857230209f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_BC_CCK, true, STPu(0.221085653f, 0.0f),
                                         STPtauU(17.68670462f, 0.0f),
                                         STPtauX(592.1864649f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.577797723f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Bistratified, true, STPu(0.246886553f, 0.0f),
                                         STPtauU(13.59791022f, 0.0f),
                                         STPtauX(775.0419417f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.576515291f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Ivy, true, STPu(0.225317729f, 0.0f),
                                         STPtauU(13.54144463f, 0.0f),
                                         STPtauX(706.9023888f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.380229451f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_MFA_ORDEN, true, STPu(0.232956061f, 0.0f),
                                         STPtauU(18.29762673f, 0.0f),
                                         STPtauX(605.2466713f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.537380416f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Pyramidal, true, STPu(0.213196513f, 0.0f),
                                         STPtauU(16.60665423f, 0.0f),
                                         STPtauX(481.8508503f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.48628716f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_LMR_Targeting, true, STPu(0.232739881f, 0.0f),
                                         STPtauU(26.59062988f, 0.0f),
                                         STPtauX(615.1682504f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.613516473f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_QuadD_LM, true, STPu(0.23909629f, 0.0f),
                                         STPtauU(26.14661454f, 0.0f),
                                         STPtauX(563.468709f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.894190356f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Axo_Axonic, true, STPu(0.244663105f, 0.0f),
                                         STPtauU(25.51127047f, 0.0f),
                                         STPtauX(651.6350355f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.673753656f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Basket, true, STPu(0.255201515f, 0.0f),
                                         STPtauU(19.12313634f, 0.0f),
                                         STPtauX(665.1591417f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.745602473f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_BC_CCK, true, STPu(0.229677262f, 0.0f),
                                         STPtauU(20.97804705f, 0.0f),
                                         STPtauX(614.0072327f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.398921413f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Bistratified, true, STPu(0.246096207f, 0.0f),
                                         STPtauU(22.68666918f, 0.0f),
                                         STPtauX(660.479446f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.240503496f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Ivy, true, STPu(0.232071036f, 0.0f),
                                         STPtauU(17.71732982f, 0.0f),
                                         STPtauX(675.5443943f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.513908651f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_MFA_ORDEN, true, STPu(0.236346724f, 0.0f),
                                         STPtauU(28.44917201f, 0.0f),
                                         STPtauX(578.8951805f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.961017332f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Pyramidal, true, STPu(0.219560529f, 0.0f),
                                         STPtauU(23.01118394f, 0.0f),
                                         STPtauX(439.5040963f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.007078916f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_LMR_Targeting, true, STPu(0.225271179f, 0.0f),
                                         STPtauU(21.22211644f, 0.0f),
                                         STPtauX(712.2728097f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.386534271f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_QuadD_LM, true, STPu(0.231630824f, 0.0f),
                                         STPtauU(21.00936153f, 0.0f),
                                         STPtauX(637.949947f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.517195135f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Axo_Axonic, true, STPu(0.235379179f, 0.0f),
                                         STPtauU(21.44649708f, 0.0f),
                                         STPtauX(762.6005365f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.549658496f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Basket, true, STPu(0.249546896f, 0.0f),
                                         STPtauU(15.70448009f, 0.0f),
                                         STPtauX(759.1190877f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.896195604f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_BC_CCK, true, STPu(0.22096495f, 0.0f),
                                         STPtauU(17.0758504f, 0.0f),
                                         STPtauX(693.9231434f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.322167425f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Bistratified, true, STPu(0.240575261f, 0.0f),
                                         STPtauU(17.26898777f, 0.0f),
                                         STPtauX(776.5689731f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.958324186f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Ivy, true, STPu(0.220202643f, 0.0f),
                                         STPtauU(13.87512648f, 0.0f),
                                         STPtauX(806.4750758f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.317088488f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_MFA_ORDEN, true, STPu(0.22845699f, 0.0f),
                                         STPtauU(22.52027885f, 0.0f),
                                         STPtauX(642.0975453f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.533747322f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Pyramidal, true, STPu(0.216244394f, 0.0f),
                                         STPtauU(20.61711347f, 0.0f),
                                         STPtauX(496.0484093f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.149050278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Pyramidal, CA3_LMR_Targeting, true, STPu(0.195199069f, 0.0f),
                                     STPtauU(26.73415912f, 0.0f),
                                     STPtauX(552.2685883f, 0.0f),
                                     STPtdAMPA(5.667238706f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_QuadD_LM, true, STPu(0.199701771f, 0.0f),
                                     STPtauU(27.15856439f, 0.0f),
                                     STPtauX(453.2939139f, 0.0f),
                                     STPtdAMPA(5.815770593f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Axo_Axonic, true, STPu(0.197240611f, 0.0f),
                                     STPtauU(26.26018349f, 0.0f),
                                     STPtauX(630.7258001f, 0.0f),
                                     STPtdAMPA(4.92302074f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Basket, true, STPu(0.221350678f, 0.0f),
                                     STPtauU(21.16086172f, 0.0f),
                                     STPtauX(691.4177768f, 0.0f),
                                     STPtdAMPA(3.97130389f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_BC_CCK, true, STPu(0.204890451f, 0.0f),
                                     STPtauU(22.45458108f, 0.0f),
                                     STPtauX(530.4003851f, 0.0f),
                                     STPtdAMPA(4.2935729f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Bistratified, true, STPu(0.203609518f, 0.0f),
                                     STPtauU(23.85082513f, 0.0f),
                                     STPtauX(569.1504567f, 0.0f),
                                     STPtdAMPA(5.367704956f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Ivy, true, STPu(0.191829256f, 0.0f),
                                     STPtauU(18.84783629f, 0.0f),
                                     STPtauX(623.453684f, 0.0f),
                                     STPtdAMPA(4.481241177f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_MFA_ORDEN, true, STPu(0.196218723f, 0.0f),
                                     STPtauU(29.01335489f, 0.0f),
                                     STPtauX(444.9925289f, 0.0f),
                                     STPtdAMPA(5.948303553f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Pyramidal, true, STPu(0.192566137f, 0.0f),
                                     STPtauU(21.44820657f, 0.0f),
                                     STPtauX(318.510891f, 0.0f),
                                     STPtdAMPA(10.21893984f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));