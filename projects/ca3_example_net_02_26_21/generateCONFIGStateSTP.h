int CA3_QuadD_LM = sim.createGroup("CA3_QuadD_LM", 328.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 190.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Basket = sim.createGroup("CA3_Basket", 51.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_BC_CCK = sim.createGroup("CA3_BC_CCK", 66.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Bistratified = sim.createGroup("CA3_Bistratified", 463.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Ivy = sim.createGroup("CA3_Ivy", 233.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 152.0,
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              
int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 743.0,
                              EXCITATORY_NEURON, 0, GPU_CORES);
                                                   
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

sim.connect(CA3_QuadD_LM, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.00529496670795314f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.473054473f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 1.55f), 0.0667209642727206f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.815288206f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0495724663599669f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.308152788f, 0.0f);
                                       
sim.connect(CA3_QuadD_LM, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.11882477878628f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.183249644f, 0.0f);
                                       
sim.connect(CA3_Axo_Axonic, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.15f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.869561088f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_QuadD_LM, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.750808378f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.025f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.02478806f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 1.55f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.281611994f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.686238357f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Bistratified, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.025f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.770407646f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.808726221f, 0.0f);
                                       
sim.connect(CA3_Basket, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.15f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.572405696f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_QuadD_LM, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.025f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.334971075f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.025f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.493830393f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 1.55f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.745239175f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 0.965567683f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Bistratified, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.025f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.371483576f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.355214118f, 0.0f);
                                       
sim.connect(CA3_BC_CCK, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.15f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.306303671f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_QuadD_LM, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.00775423637622817f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.490477376f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.00659307293350001f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.655173699f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 1.55f), 0.00926382440444106f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.99431849f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00431732093529376f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.442647868f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Bistratified, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.0325785208207766f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.547036443f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Ivy, "random", RangeWeight(0.0f, 0.65f, 1.65f), 0.00385580973430123f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.061179756f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00862472688778918f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.567727407f, 0.0f);
                                       
sim.connect(CA3_Bistratified, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.0278686093547943f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.431148109f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_QuadD_LM, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.00198176910683873f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.567261924f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.00424634524060682f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.758382851f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 1.55f), 0.016300617068683923f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.111359644f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.010742278540606838f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.54009769f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Bistratified, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.01703156100118382f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.660111909f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Ivy, "random", RangeWeight(0.0f, 0.65f, 1.65f), 0.00375271870052702f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.142741525f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.0169285421706311f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.687214907f, 0.0f);
                                       
sim.connect(CA3_Ivy, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.0720967942651253f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.540661646f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_QuadD_LM, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.00375566003798539f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.471995564f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.00437941786544469f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.628518636f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 1.55f), 0.0072882240621001f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.972333716f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.00526662735486614f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.415044453f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Bistratified, "random", RangeWeight(0.0f, 1.3f, 2.3f), 0.00480204193183194f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.536494845f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Ivy, "random", RangeWeight(0.0f, 0.65f, 1.65f), 0.00267981747242981f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 2.081976802f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 0.75f, 1.75f), 0.00210548528014741f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.552656079f, 0.0f);
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.0417555599977689f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.360315289f, 0.0f);
                                       
sim.connect(CA3_Pyramidal, CA3_QuadD_LM, "random", RangeWeight(0.0f, 1.25f, 2.25f), 0.0133672243607345f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.874424964f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Axo_Axonic, "random", RangeWeight(0.0f, 0.7f, 1.7f), 0.0148205394733136f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.932621138f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Basket, "random", RangeWeight(0.0f, 1.45f, 2.45f), 0.0197417562762975f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 1.172460639f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_BC_CCK, "random", RangeWeight(0.0f, 1.0f, 2.0f), 0.0172994236281402f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.847532877f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Bistratified, "random", RangeWeight(0.0f, 0.7f, 1.7f), 0.015857085924813f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.883682094f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Ivy, "random", RangeWeight(0.0f, 1.35f, 2.35f), 0.0251567907913944f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 1.314331038f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.25f, 2.25f), 0.0209934225689348f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.88025265f, 0.0f);
                                   
sim.connect(CA3_Pyramidal, CA3_Pyramidal, "random", RangeWeight(0.0f, 0.55f, 1.55f), 0.0250664662231983f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.553062478f, 0.0f);
                                   
sim.setSTP(CA3_QuadD_LM, CA3_Axo_Axonic, true, STPu(0.152487328f, 0.0f),
                                         STPtauU(22.34022321f, 0.0f),
                                         STPtauX(635.0122846f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.168513359f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Basket, true, STPu(0.3362564094f, 0.0f),
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
                                     
sim.setSTP(CA3_QuadD_LM, CA3_Pyramidal, true, STPu(0.10647895115000001f, 0.0f),
                                         STPtauU(24.78801373f, 0.0f),
                                         STPtauX(382.1372881f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.107305544f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Axo_Axonic, CA3_Pyramidal, true, STPu(0.1302436377f, 0.0f),
                                         STPtauU(12.93029701f, 0.0f),
                                         STPtauX(361.0287219f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.623472774f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_QuadD_LM, true, STPu(0.30675536625f, 0.0f),
                                         STPtauU(19.30519205f, 0.0f),
                                         STPtauX(589.2002267f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.162447367f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Axo_Axonic, true, STPu(0.1910149885f, 0.0f),
                                         STPtauU(23.20689302f, 0.0f),
                                         STPtauX(725.0291555f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.800283876f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Basket, true, STPu(0.38950627465000004f, 0.0f),
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
                                     
sim.setSTP(CA3_Basket, CA3_Bistratified, true, STPu(0.1754364731f, 0.0f),
                                         STPtauU(16.7158668f, 0.0f),
                                         STPtauX(680.3279514f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.720497985f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_MFA_ORDEN, true, STPu(0.301856475f, 0.0f),
                                         STPtauU(19.60369075f, 0.0f),
                                         STPtauX(581.9355018f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.230610278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Basket, CA3_Pyramidal, true, STPu(0.12521945645000002f, 0.0f),
                                         STPtauU(16.73589406f, 0.0f),
                                         STPtauX(384.3363321f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.63862234f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_QuadD_LM, true, STPu(0.21874558374999997f, 0.0f),
                                         STPtauU(17.34307333f, 0.0f),
                                         STPtauX(398.1521614f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.480095841f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Axo_Axonic, true, STPu(0.12352907559999998f, 0.0f),
                                         STPtauU(18.502335f, 0.0f),
                                         STPtauX(477.4286298f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.43847192f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Basket, true, STPu(0.28360436175000003f, 0.0f),
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
                                     
sim.setSTP(CA3_BC_CCK, CA3_Bistratified, true, STPu(0.1260104916f, 0.0f),
                                         STPtauU(15.24644944f, 0.0f),
                                         STPtauX(478.3138872f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.9732847f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_MFA_ORDEN, true, STPu(0.21055725f, 0.0f),
                                         STPtauU(17.83862927f, 0.0f),
                                         STPtauX(421.419837f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.538700998f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_BC_CCK, CA3_Pyramidal, true, STPu(0.08024396655f, 0.0f),
                                         STPtauU(13.7603965f, 0.0f),
                                         STPtauX(376.8743946f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.101015633f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_QuadD_LM, true, STPu(0.2952313625f, 0.0f),
                                         STPtauU(17.89475495f, 0.0f),
                                         STPtauX(594.3278762f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.529521921f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Axo_Axonic, true, STPu(0.16745713739999998f, 0.0f),
                                         STPtauU(19.16221108f, 0.0f),
                                         STPtauX(686.2791734f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.568545011f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Basket, true, STPu(0.36559896319999996f, 0.0f),
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
                                     
sim.setSTP(CA3_Bistratified, CA3_Bistratified, true, STPu(0.1728205871f, 0.0f),
                                         STPtauU(13.59791022f, 0.0f),
                                         STPtauX(775.0419417f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.576515291f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Ivy, true, STPu(0.30417893415f, 0.0f),
                                         STPtauU(13.54144463f, 0.0f),
                                         STPtauX(706.9023888f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.380229451f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_MFA_ORDEN, true, STPu(0.29119507624999996f, 0.0f),
                                         STPtauU(18.29762673f, 0.0f),
                                         STPtauX(605.2466713f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.537380416f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Bistratified, CA3_Pyramidal, true, STPu(0.11725808215000001f, 0.0f),
                                         STPtauU(16.60665423f, 0.0f),
                                         STPtauX(481.8508503f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.48628716f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_QuadD_LM, true, STPu(0.2988703625f, 0.0f),
                                         STPtauU(26.14661454f, 0.0f),
                                         STPtauX(563.468709f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.894190356f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Axo_Axonic, true, STPu(0.17126417349999998f, 0.0f),
                                         STPtauU(25.51127047f, 0.0f),
                                         STPtauX(651.6350355f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.673753656f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Basket, true, STPu(0.37004219675f, 0.0f),
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
                                     
sim.setSTP(CA3_Ivy, CA3_Bistratified, true, STPu(0.1722673449f, 0.0f),
                                         STPtauU(22.68666918f, 0.0f),
                                         STPtauX(660.479446f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.240503496f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Ivy, true, STPu(0.3132958986f, 0.0f),
                                         STPtauU(17.71732982f, 0.0f),
                                         STPtauX(675.5443943f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.513908651f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_MFA_ORDEN, true, STPu(0.29543340500000004f, 0.0f),
                                         STPtauU(28.44917201f, 0.0f),
                                         STPtauX(578.8951805f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(6.961017332f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Ivy, CA3_Pyramidal, true, STPu(0.12075829095000001f, 0.0f),
                                         STPtauU(23.01118394f, 0.0f),
                                         STPtauX(439.5040963f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(9.007078916f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_QuadD_LM, true, STPu(0.28953853f, 0.0f),
                                         STPtauU(21.00936153f, 0.0f),
                                         STPtauX(637.949947f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.517195135f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Axo_Axonic, true, STPu(0.1647654253f, 0.0f),
                                         STPtauU(21.44649708f, 0.0f),
                                         STPtauX(762.6005365f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.549658496f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Basket, true, STPu(0.36184299919999996f, 0.0f),
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
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Bistratified, true, STPu(0.1684026827f, 0.0f),
                                         STPtauU(17.26898777f, 0.0f),
                                         STPtauX(776.5689731f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.958324186f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Ivy, true, STPu(0.29727356805000005f, 0.0f),
                                         STPtauU(13.87512648f, 0.0f),
                                         STPtauX(806.4750758f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(4.317088488f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_MFA_ORDEN, true, STPu(0.2855712375f, 0.0f),
                                         STPtauU(22.52027885f, 0.0f),
                                         STPtauX(642.0975453f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.533747322f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Pyramidal, true, STPu(0.11893441670000002f, 0.0f),
                                         STPtauU(20.61711347f, 0.0f),
                                         STPtauX(496.0484093f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.149050278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     
sim.setSTP(CA3_Pyramidal, CA3_QuadD_LM, true, STPu(0.14977632825f, 0.0f),
                                     STPtauU(27.15856439f, 0.0f),
                                     STPtauX(453.2939139f, 0.0f),
                                     STPtdAMPA(5.815770593f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Axo_Axonic, true, STPu(0.2564127943f, 0.0f),
                                     STPtauU(26.26018349f, 0.0f),
                                     STPtauX(630.7258001f, 0.0f),
                                     STPtdAMPA(4.92302074f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Basket, true, STPu(0.12174287290000001f, 0.0f),
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
                                 
sim.setSTP(CA3_Pyramidal, CA3_Bistratified, true, STPu(0.2646923734f, 0.0f),
                                     STPtauU(23.85082513f, 0.0f),
                                     STPtauX(569.1504567f, 0.0f),
                                     STPtdAMPA(5.367704956f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_Ivy, true, STPu(0.1246890164f, 0.0f),
                                     STPtauU(18.84783629f, 0.0f),
                                     STPtauX(623.453684f, 0.0f),
                                     STPtdAMPA(4.481241177f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 
sim.setSTP(CA3_Pyramidal, CA3_MFA_ORDEN, true, STPu(0.14716404225000002f, 0.0f),
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
                                 
sim.setNeuronMonitor(CA3_QuadD_LM, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_Axo_Axonic, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_Basket, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_BC_CCK, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_Bistratified, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_Ivy, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_MFA_ORDEN, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_Pyramidal, "DEFAULT");