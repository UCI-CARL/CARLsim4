int CA3_Basket = sim.createGroup("CA3_Basket", 3089,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 11771,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 74366,
                              EXCITATORY_NEURON, 0, GPU_CORES);
                              
sim.setNeuronParameters(CA3_Basket, 45.0, 0.0, 0.9951729, 0.0,
                                                -57.506126, 0.0, -23.378766, 0.0, 0.003846186,
                                                0.0, 9.2642765, 0.0, 18.454934,
                                                0.0, -47.555661, 0.0,
                                                -6.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_MFA_ORDEN, 209.0, 0.0, 1.37980713457205, 0.0,
                                                -57.076423571379, 0.0, -39.1020427841762, 0.0, 0.00783805979364104,
                                                0.0, 12.9332855397722, 0.0, 16.3132681887705,
                                                0.0, -40.6806648852695, 0.0,
                                                0.0, 0.0, 1);
                     
sim.setNeuronParameters(CA3_Pyramidal, 366.0, 0.0, 0.792338703789581, 0.0,
                                                -63.2044008171655, 0.0, -33.6041733124267, 0.0, 0.00838350334098279,
                                                0.0, -42.5524776883928, 0.0, 35.8614648558726,
                                                0.0, -38.8680990294091, 0.0,
                                                588.0, 0.0, 1);
                     
sim.connect(CA3_Basket, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00663136909224044f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.139606233f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00573675602167279f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.681117626f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.14999999521149f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.463203583f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0117345087001957f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.450174133f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00913901938344389f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.828413006f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0637836805073049f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.915443373f, 0.0f);
                                       
sim.connect(CA3_Pyramidal, CA3_Basket, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0161746579069214f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.910763922f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0176731417548347f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 1.118971391f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0250664662231983f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.304184363f, 0.0f);
                                   
sim.setSTP(CA3_Basket, CA3_Basket, true, STPu(0.162517426384018f, 0.0f),
                                         STPtauU(11.19042564f, 0.0f),
                                         STPtauX(689.5059466f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.007016545f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_MFA_ORDEN, true, STPu(0.177068034944573f, 0.0f),
                                         STPtauU(19.60369075f, 0.0f),
                                         STPtauX(581.9355018f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.230610278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Pyramidal, true, STPu(0.118642073007157f, 0.0f),
                                         STPtauU(16.73589406f, 0.0f),
                                         STPtauX(384.3363321f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.63862234f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Basket, true, STPu(0.24814064689538f, 0.0f),
                                         STPtauU(15.70448009f, 0.0f),
                                         STPtauX(759.1190877f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.896195604f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_MFA_ORDEN, true, STPu(0.206752592891043f, 0.0f),
                                         STPtauU(22.52027885f, 0.0f),
                                         STPtauX(642.0975453f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.533747322f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Pyramidal, true, STPu(0.112589304970516f, 0.0f),
                                         STPtauU(20.61711347f, 0.0f),
                                         STPtauX(496.0484093f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.149050278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Pyramidal, CA3_Basket, true, STPu(0.217145987458849f, 0.0f),
                                     STPtauU(21.16086172f, 0.0f),
                                     STPtauX(691.4177768f, 0.0f),
                                     STPtdAMPA(3.97130389f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_MFA_ORDEN, true, STPu(0.182858254313729f, 0.0f),
                                     STPtauU(29.01335489f, 0.0f),
                                     STPtauX(444.9925289f, 0.0f),
                                     STPtdAMPA(5.948303553f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Pyramidal, true, STPu(0.27922089865f, 0.0f),
                                     STPtauU(21.44820657f, 0.0f),
                                     STPtauX(318.510891f, 0.0f),
                                     STPtdAMPA(10.21893984f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setNeuronMonitor(CA3_Basket, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_MFA_ORDEN, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_Pyramidal, "DEFAULT");
                                 
