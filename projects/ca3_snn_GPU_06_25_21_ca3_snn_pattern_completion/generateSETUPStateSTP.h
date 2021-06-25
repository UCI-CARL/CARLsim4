sim.setSpikeMonitor(CA3_Basket, "DEFAULT");

sim.setSpikeMonitor(CA3_MFA_ORDEN, "DEFAULT");
                                 
sim.setSpikeMonitor(CA3_Pyramidal, "DEFAULT");
                                 
int DG_Granule_frate = 100.0f;

PoissonRate DG_Granule_rate(394502, true); // create PoissonRate object for all Granule cells
DG_Granule_rate.setRates(0.4f); // set all mean firing rates for the object to 0.4 Hz
sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // link the object with defined Granule cell group, with refractory period 1 ms
                                      
