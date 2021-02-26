# Import libraries that are necessary for the function library
import pandas as pd
import utilities as ut


def generateSNNSyntax(fileName):

    # Read in the csv and excel spreadsheets containing the parameters
    dfNetList = pd.read_excel(fileName, sheet_name='CA3_netlist');
    dfPopSize = pd.read_excel(fileName,
                              sheet_name='pop_size_neuron_type');
    dfIM = pd.read_excel(fileName,
                         sheet_name='IM_parameters_by_neuron_type').dropna();
    dfConnMat = pd.read_excel(fileName,
                              sheet_name='syn_conn_prob')
    dfSynWeight = pd.read_excel(fileName,
                                sheet_name='syn_conds').dropna();
    dfMultiplier = pd.read_excel(fileName,
                                 sheet_name='syn_weights').dropna();
    dfSTPU = pd.read_excel(fileName,
                           sheet_name='syn_resource_release'). \
                           dropna();
    dfSTPUMultiplier = pd.read_excel(fileName,
                                     sheet_name='syn_resource_release_multiplier'). \
                                     dropna();
    dfSTPTauX = pd.read_excel(fileName,
                              sheet_name='syn_rec_const'). \
                              dropna();
    dfSTPTauNT = pd.read_excel(fileName,
                               sheet_name='syn_decay_const'). \
                               dropna();
    dfSTPTauU = pd.read_excel(fileName,
                              sheet_name='syn_facil_const'). \
                              dropna();
    
    # Clean up the connectivity matrices to retrieve the newest probabilities of
    # connection
    dfConnMat = dfConnMat.iloc[0:8,:-1];
    
    # Rename each of the neuron types to a syntax appropriate for group declaration
    # by placing an underscore between spaces
    ut.cellTypeNameCARL(df = dfNetList, dfName = "dfNetList", colName = "Cell Types");
    ut.cellTypeNameCARL(df = dfIM, dfName = "dfIM", colName = "Neuron Type");
    ut.cellTypeNameCARL(dfPopSize, dfName = "dfPopSize", colName = "Neuron Type (Internal)");
    ut.cellTypeNameCARL(dfConnMat, dfName = "dfConnMat", colName = "Pre-Post");
    
    ut.cellTypeNameCARL(dfSynWeight, dfName = "dfSynWeight", newName = "Wij");
    ut.cellTypeNameCARL(dfMultiplier, dfName = "dfMultiplier", newName = "Mij");
    ut.cellTypeNameCARL(dfSTPUMultiplier, dfName = "dfSTPUMultiplier", newName = "Mij");
    ut.cellTypeNameCARL(dfSTPU, dfName = "dfSTPU", newName = "Uij");
    ut.cellTypeNameCARL(dfSTPTauX, dfName = "dfSTPTauX", newName = "TauXij");
    ut.cellTypeNameCARL(dfSTPTauNT, dfName = "dfSTPTauNT", newName = "TauNTij");
    ut.cellTypeNameCARL(dfSTPTauU, dfName = "dfSTPTauU", newName = "TauUij");
    
    # Clean the internal population size dataframe
    dfPopSize = dfPopSize.loc[0:7];
    
    # Create a data frame that will be used to generate syntax for the groups
    # in the simulation
    dfNeuronTypes = dfNetList[['Cell Types', 'Internal/External', 'Excitatory/Inhibitory']].dropna();
    dfNeuronTypes = dfNeuronTypes.set_index('Cell Types');
    dfNeuronTypes = dfNeuronTypes.reindex(index = dfPopSize['Neuron Type (Internal)']. \
                                                  reset_index(drop=True).dropna());
    dfNeuronTypes = dfNeuronTypes.reset_index();
    dfNeuronTypes['popSizes'] = dfPopSize['Population Size (Internal)'].reset_index(
                                drop=True).dropna();
    dfNeuronTypes = dfNeuronTypes.rename(columns = {'index' : 'Cell Types'});
    
    # For testing purposes, decrease the population sizes by a factor of 50
    #dfNeuronTypes['popSizes'] = dfNeuronTypes['popSizes'] / 50
    #dfNeuronTypes.popSizes = dfNeuronTypes.popSizes.astype(float).round();
    # Generate syntax to define the groups in the simulation
    codeGroups = '';
    
    # Take the number of neuron types minus two as for now we are not going to be
    # considering the MF and stellate input
    for x in range(len(dfNeuronTypes)):
        if (dfNeuronTypes.iloc[x,1] == 'External'):
            if (dfNeuronTypes.iloc[x,2] == 'Excitatory'):
                addCodeGrps = r'''int {grpid} = sim.createSpikeGeneratorGroup("{grpid}", {pop_size},
                              EXCITATORY_NEURON, 0, GPU_CORES);
                              '''.format(grpid = dfNeuronTypes.iloc[x,0],
                                         pop_size = dfNeuronTypes.iloc[x,3]);
            else:
                addCodeGrps = r'''int {grpid} = sim.createSpikeGeneratorGroup("{grpid}", {pop_size},
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              '''.format(grpid = dfNeuronTypes.iloc[x,0],
                                         pop_size = dfNeuronTypes.iloc[x,3]);
        else:
            if (dfNeuronTypes.iloc[x,2] == 'Excitatory'):
                addCodeGrps = r'''int {grpid} = sim.createGroup("{grpid}", {pop_size},
                              EXCITATORY_NEURON, 0, GPU_CORES);
                              '''.format(grpid = dfNeuronTypes.iloc[x,0],
                                         pop_size = dfNeuronTypes.iloc[x,3]);
            else:
                addCodeGrps = r'''int {grpid} = sim.createGroup("{grpid}", {pop_size},
                              INHIBITORY_NEURON, 0, GPU_CORES);
                              '''.format(grpid = dfNeuronTypes.iloc[x,0],
                                         pop_size = dfNeuronTypes.iloc[x,3]);
        codeGroups = codeGroups + addCodeGrps;
        codeGroups = codeGroups + '\n';
    
    # Sort the IM data frame so that it is in the proper order for generating syntax
    dfIM = dfIM[['Neuron Type', 'C_mean', 'C_std', 'k_mean', 'k_std',
                 'Vr_mean', 'Vr_std', 'Vt_mean', 'Vt_std', 'a_mean',
                 'a_std', 'b_mean', 'b_std', 'Vpeak_mean', 'Vpeak_std',
                 'Vmin_mean', 'Vmin_std', 'd_mean', 'd_std']];
    
    
    # Create syntax for the IM parameters through a for loop
    codeIM = '';
    for x in range(len(dfIM.iloc[:,0])):
        addCodeIM = r'''sim.setNeuronParameters({grpid}, {C}, {C_std}, {k}, {k_std},
                                                {Vr}, {Vr_std}, {Vt}, {Vt_std}, {a},
                                                {a_std}, {b}, {b_std}, {Vpeak},
                                                {Vpeak_std}, {Vmin}, {Vmin_std},
                                                {d}, {d_std}, 1);
                     '''.format(grpid = dfIM.iloc[x,0], C = dfIM.iloc[x,1],
                                C_std = dfIM.iloc[x,2], k = dfIM.iloc[x,3],
                                k_std = dfIM.iloc[x,4], Vr = dfIM.iloc[x,5],
                                Vr_std = dfIM.iloc[x,6], Vt = dfIM.iloc[x,7],
                                Vt_std = dfIM.iloc[x,8], a = dfIM.iloc[x,9],
                                a_std = dfIM.iloc[x,10], b = dfIM.iloc[x,11],
                                b_std = dfIM.iloc[x,12], Vpeak = dfIM.iloc[x,13],
                                Vpeak_std = dfIM.iloc[x,14], Vmin = dfIM.iloc[x,15],
                                Vmin_std = dfIM.iloc[x,16], d = dfIM.iloc[x,17],
                                d_std = dfIM.iloc[x,18]);
        codeIM = codeIM + addCodeIM;
        codeIM = codeIM + '\n';
    
    
    sortedColumnsConnMat = list(dfConnMat.iloc[0:8,0]);
    sortedColumnsConnMat.insert(0, 'Pre_Post');
    sortedColumnsTMMat = ['CA3_QuadD_LM_mean', 'CA3_QuadD_LM_std', 'CA3_Axo_Axonic_mean',
                          'CA3_Axo_Axonic_std', 'CA3_Basket_mean', 'CA3_Basket_std',
                          'CA3_BC_CCK_mean', 'CA3_BC_CCK_std', 'CA3_Bistratified_mean',
                          'CA3_Bistratified_std', 'CA3_Ivy_mean', 'CA3_Ivy_std',
                          'CA3_MFA_ORDEN_mean', 'CA3_MFA_ORDEN_std', 
                          'CA3_Pyramidal_mean', 'CA3_Pyramidal_std'
                          ];
    sortedColumnsTMMat.insert(0, 'Wij');
    dfConnMat = dfConnMat.reindex(columns=sortedColumnsConnMat);
    dfSynWeight = dfSynWeight.reindex(columns=sortedColumnsTMMat);
    
    # Re-order columns so that probability of connections are appropriately set
    # within the for loop
    sortedColumnsTMMat[0] = 'Mij';
    dfMultiplier = dfMultiplier.reindex(columns=sortedColumnsTMMat);
    sortedColumnsTMMat[0] = 'Mij';
    dfSTPUMultiplier = dfSTPUMultiplier.reindex(columns=sortedColumnsTMMat);
    sortedColumnsTMMat[0] = 'Uij';
    dfSTPU = dfSTPU.reindex(columns=sortedColumnsTMMat);
    sortedColumnsTMMat[0] = 'TauUij';
    dfSTPTauU = dfSTPTauU.reindex(columns=sortedColumnsTMMat);
    sortedColumnsTMMat[0] = 'TauXij';
    dfSTPTauX = dfSTPTauX.reindex(columns=sortedColumnsTMMat);
    sortedColumnsTMMat[0] = 'TauNTij';
    dfSTPTauNT = dfSTPTauNT.reindex(columns=sortedColumnsTMMat);
    
    # Create syntax for the connection probabilities between groups, and set STP
    # for each of the connection types
    count = 0;
    codeConn = '';
    codeSetConnMonitor = '';
    codeSTP = '';
    k = 0;
    
    # Take the length of the connectivity matrix subtracted by two so that we
    # are not considering the MF and stellate connectivity
    for x in range(len(dfConnMat.iloc[:,0])):
        for y in range(len(dfConnMat.iloc[0,:])-1):
            # Code block allows for excitatory and inhibitory range weights to be
            # assigned (wmin and wmax currently 0.1 minus or plus their value in XL
            # to allow for RangeWeight function to work with plastic synapses. This
            # will change in the future when we have better TM estimates)
            if (dfConnMat.iloc[x,y+1] == 0):
                count = count + 1;
            else:
                addCodeSetConnMonitor = r'''sim.setConnectionMonitor({gin}, {gout}, "DEFAULT");
                                         '''.format(gin = dfConnMat.iloc[x,0],
                                                    gout = dfConnMat.iloc[y,0]
                                                   );
                codeSetConnMonitor = codeSetConnMonitor + addCodeSetConnMonitor;
                codeSetConnMonitor = codeSetConnMonitor + '\n';
                if (dfConnMat.iloc[x,0] == "CA3_Pyramidal"):
                    addCodeConn = r'''sim.connect({gin}, {gout}, "random", RangeWeight(0.0f, {winit}f, {wmax}f), {p}f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, {g}f, 0.0f);
                                   '''.format(gin = dfConnMat.iloc[x,0],
                                              gout = dfConnMat.iloc[y,0],
                                              winit = float(dfMultiplier.iloc[x,k+1]),
                                              wmax = dfMultiplier.iloc[x,k+1] + 1.0,
                                              p = dfConnMat.iloc[x,y+1],
                                              g = dfSynWeight.iloc[x,k+1]
                                              );
                    addCodeSTP = r'''sim.setSTP({gin}, {gout}, true, STPu({U_mean}f, {U_std}f),
                                     STPtauU({tau_u_mean}f, {tau_u_std}f),
                                     STPtauX({tau_x_mean}f, {tau_x_std}f),
                                     STPtdAMPA({tau_d_mean}f, {tau_d_std}f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f));
                                 '''.format(gin = dfSTPU.iloc[x,0],
                                            gout = dfSTPU.iloc[y,0],
                                            U_mean = dfSTPUMultiplier.iloc[x,k+1]*dfSTPU.iloc[x,k+1],
                                            U_std = dfSTPU.iloc[x,k+2],
                                            tau_u_mean = dfSTPTauU.iloc[x,k+1],
                                            tau_u_std = dfSTPTauU.iloc[x,k+2],
                                            tau_x_mean = dfSTPTauX.iloc[x,k+1],
                                            tau_x_std = dfSTPTauX.iloc[x,k+2],
                                            tau_d_mean = dfSTPTauNT.iloc[x,k+1],
                                            tau_d_std = dfSTPTauNT.iloc[x,k+2]
                                            );
                    codeConn = codeConn + addCodeConn;
                    codeConn = codeConn + '\n';
                    codeSTP = codeSTP + addCodeSTP;
                    codeSTP = codeSTP + '\n';
                else:
                    if (dfSynWeight.iloc[x,k+1] == 0):
                        addCodeConn = r'''sim.connect({gin}, {gout}, "random", RangeWeight(1.0f), {p}f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, {g}f, 0.0f);
                                       '''.format(gin = dfConnMat.iloc[x,0],
                                                  gout = dfConnMat.iloc[y,0],
                                                  p = dfConnMat.iloc[x,k+1],
                                                  g = dfSynWeight.iloc[x,k+1]
                                                 );
                        addCodeSTP = r'''sim.setSTP({gin}, {gout}, true, STPu({U_mean}f, {U_std}f),
                                         STPtauU({tau_u_mean}f, {tau_u_std}f),
                                         STPtauX({tau_x_mean}f, {tau_x_std}f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa({tau_d_mean}f, {tau_d_std}f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     '''.format(gin = dfSTPU.iloc[x,0],
                                                gout = dfSTPU.iloc[y,0],
                                                U_mean = dfSTPUMultiplier.iloc[x,k+1]*dfSTPU.iloc[x,k+1],
                                                U_std = dfSTPU.iloc[x,k+2],
                                                tau_u_mean = dfSTPTauU.iloc[x,k+1],
                                                tau_u_std = dfSTPTauU.iloc[x,k+2],
                                                tau_x_mean = dfSTPTauX.iloc[x,k+1],
                                                tau_x_std = dfSTPTauX.iloc[x,k+2],
                                                tau_d_mean = dfSTPTauNT.iloc[x,k+1],
                                                tau_d_std = dfSTPTauNT.iloc[x,k+2]
                                                );
                        codeConn = codeConn + addCodeConn;
                        codeConn = codeConn + '\n';
                        codeSTP = codeSTP + addCodeSTP;
                        codeSTP = codeSTP + '\n';
                    else:
                        addCodeConn = r'''sim.connect({gin}, {gout}, "random", RangeWeight(0.0f, {winit}f, {wmax}f), {p}f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, {g}f, 0.0f);
                                       '''.format(gin = dfConnMat.iloc[x,0],
                                                  gout = dfConnMat.iloc[y,0],
                                                  winit = float(dfMultiplier.iloc[x,k+1]),
                                                  wmax = dfMultiplier.iloc[x,k+1] + 1.0,
                                                  p = dfConnMat.iloc[x,y+1],
                                                  g = dfSynWeight.iloc[x,k+1]
                                                 );
                        addCodeSTP = r'''sim.setSTP({gin}, {gout}, true, STPu({U_mean}f, {U_std}f),
                                         STPtauU({tau_u_mean}f, {tau_u_std}f),
                                         STPtauX({tau_x_mean}f, {tau_x_std}f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa({tau_d_mean}f, {tau_d_std}f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f));
                                     '''.format(gin = dfSTPU.iloc[x,0],
                                                gout = dfSTPU.iloc[y,0],
                                                U_mean = dfSTPUMultiplier.iloc[x,k+1]*dfSTPU.iloc[x,k+1],
                                                U_std = dfSTPU.iloc[x,k+2],
                                                tau_u_mean = dfSTPTauU.iloc[x,k+1],
                                                tau_u_std = dfSTPTauU.iloc[x,k+2],
                                                tau_x_mean = dfSTPTauX.iloc[x,k+1],
                                                tau_x_std = dfSTPTauX.iloc[x,k+2],
                                                tau_d_mean = dfSTPTauNT.iloc[x,k+1],
                                                tau_d_std = dfSTPTauNT.iloc[x,k+2]
                                                );
                        codeConn = codeConn + addCodeConn;
                        codeConn = codeConn + '\n';
                        codeSTP = codeSTP + addCodeSTP;
                        codeSTP = codeSTP + '\n';
            k = k + 2;
        k = 0;

    # Create syntax for spike and neuron monitors for the groups in
    # the SNN
    codeSetNeuronMonitor = '';
    codeSetSpikeMonitor = '';
    for x in range(len(dfNeuronTypes.iloc[:,0])):
        addCodeSetNeuronMonitor = r'''sim.setNeuronMonitor({grpid}, "DEFAULT");
                                 '''.format(grpid = dfNeuronTypes.iloc[x,0]);
        codeSetNeuronMonitor = codeSetNeuronMonitor + addCodeSetNeuronMonitor;
        codeSetNeuronMonitor = codeSetNeuronMonitor + '\n';
        
        addCodeSetSpikeMonitor = r'''sim.setSpikeMonitor({grpid}, "DEFAULT");
                                 '''.format(grpid = dfNeuronTypes.iloc[x,0]);
        codeSetSpikeMonitor = codeSetSpikeMonitor + addCodeSetSpikeMonitor;
        codeSetSpikeMonitor = codeSetSpikeMonitor + '\n';
    
    # Write the syntax generated for the CONFIG State to a header file to be
    # included in the SNN simulation
    f_config_h = "generateCONFIGStateSTP.h";
    with open(f_config_h, 'w') as FOUT:
        FOUT.write(codeGroups)
        FOUT.write(codeIM)
        FOUT.write(codeConn)
        FOUT.write(codeSTP)
        FOUT.write(codeSetNeuronMonitor)
    
    
    # Write the syntax generated for the SETUP State to a header file to be
    # included in the SNN simulation
    f_setup_h = "generateSETUPStateSTP.h";
    with open(f_setup_h, 'w') as FOUT:
        FOUT.write(codeSetSpikeMonitor)


generateSNNSyntax("ca3net_02_16_21_class.xlsm")
