# Import libraries that are necessary for the function library
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import utilities as ut


def generateSNNSyntax(fileName):

    # Read in the csv and excel spreadsheets containing the parameters
    dfNetList = pd.read_excel(fileName, sheet_name='CA3_netlist');
    dfPopSizeInternal = pd.read_excel(fileName,
                                      sheet_name='pop_size_neuron_type_internal');
    dfPopSizeExternal = pd.read_excel(fileName,
                                      sheet_name='pop_size_neuron_type_external'). \
                                      dropna();
    dfIM = pd.read_excel(fileName,
                          sheet_name='IM_parameters_by_neuron_type').dropna();
    dfPoisson = pd.read_excel(fileName,
                              sheet_name='Poisson_rates_external').dropna();
    dfConnMat = pd.read_excel(fileName,
                               sheet_name='internal_to_internal_conn_prob')
    dfConnMatE2I = pd.read_excel(fileName,
                               sheet_name='external_to_internal_conn_prob')
    dfSynWeight = pd.read_excel(fileName,
                                 sheet_name='syn_conds_int_to_int').dropna();
    dfSynWeightE2I = pd.read_excel(fileName,
                                 sheet_name='syn_conds_ext_to_int').dropna();
    dfNumContacts = pd.read_excel(fileName,
                                 sheet_name='num_contacts_int_to_int').dropna();
    dfNumContactsE2I = pd.read_excel(fileName,
                                 sheet_name='num_contacts_ext_to_int').dropna();
    dfMultiplier = pd.read_excel(fileName,
                                 sheet_name='syn_conds_multiplier_int_to_int').dropna();

    dfSTPU = pd.read_excel(fileName,
                            sheet_name='syn_resource_release_int_to_int'). \
                            dropna();
    dfSTPUE2I = pd.read_excel(fileName,
                            sheet_name='syn_resource_release_ext_to_int'). \
                            dropna();
    dfSTPTauX = pd.read_excel(fileName,
                            sheet_name='syn_rec_const_int_to_int'). \
                            dropna();
    dfSTPTauXE2I = pd.read_excel(fileName,
                            sheet_name='syn_rec_const_ext_to_int'). \
                            dropna();
    dfSTPTauNT = pd.read_excel(fileName,
                            sheet_name='syn_decay_const_int_to_int'). \
                            dropna();
    dfSTPTauNTE2I = pd.read_excel(fileName,
                            sheet_name='syn_decay_const_ext_to_int'). \
                            dropna();
    dfSTPTauU = pd.read_excel(fileName,
                            sheet_name='syn_facil_const_int_to_int'). \
                            dropna();
    dfSTPTauUE2I = pd.read_excel(fileName,
                            sheet_name='syn_facil_const_ext_to_int'). \
                            dropna();

    # Clean up the connectivity matrices to retrieve the newest probabilities of
    # connection
    dfConnMat = dfConnMat.iloc[70:81,:];
    dfConnMatE2I = dfConnMatE2I.iloc[0:2,:-2];

    # Rename each of the neuron types to a syntax appropriate for group declaration
    # by placing an underscore between spaces
    ut.cellTypeNameCARL(df = dfNetList, dfName = "dfNetList", colName = "Cell Types");
    ut.cellTypeNameCARL(df = dfIM, dfName = "dfIM", colName = "Neuron Type");
    ut.cellTypeNameCARL(dfPoisson, dfName = "dfPoisson", colName = "Neuron Type (External)");
    ut.cellTypeNameCARL(dfPopSizeInternal, dfName = "dfPopSizeInternal", colName = "Neuron Type (Internal)");
    ut.cellTypeNameCARL(dfPopSizeExternal, dfName = "dfPopSizeExternal", colName = "Neuron Type (External)");
    ut.cellTypeNameCARL(dfConnMat, dfName = "dfConnMat", colName = "Pre-Post");
    ut.cellTypeNameCARL(dfConnMatE2I, dfName = "dfConnMatE2I", colName = "Pre-Post");

    ut.cellTypeNameCARL(dfSynWeight, dfName = "dfSynWeight", newName = "Wij");
    ut.cellTypeNameCARL(dfSynWeightE2I, dfName = "dfSynWeightE2I", newName = "Wij");
    ut.cellTypeNameCARL(dfNumContacts, dfName = "dfNumContacts", newName = "Nij");
    ut.cellTypeNameCARL(dfNumContactsE2I, dfName = "dfNumContacts", newName = "Nij");
    ut.cellTypeNameCARL(dfMultiplier, dfName = "dfMultiplier", newName = "Mij");
    ut.cellTypeNameCARL(dfSTPU, dfName = "dfSTPU", newName = "Uij");
    ut.cellTypeNameCARL(dfSTPUE2I, dfName = "dfSTPUE2I", newName = "Uij");
    ut.cellTypeNameCARL(dfSTPTauX, dfName = "dfSTPTauX", newName = "TauXij");
    ut.cellTypeNameCARL(dfSTPTauXE2I, dfName = "dfSTPTauXE2I", newName = "TauXij");
    ut.cellTypeNameCARL(dfSTPTauNT, dfName = "dfSTPTauNT", newName = "TauNTij");
    ut.cellTypeNameCARL(dfSTPTauNTE2I, dfName = "dfSTPTauNTE2I", newName = "TauNTij");
    ut.cellTypeNameCARL(dfSTPTauU, dfName = "dfSTPTauU", newName = "TauUij");
    ut.cellTypeNameCARL(dfSTPTauUE2I, dfName = "dfSTPTauUE2I", newName = "TauUij");

    # Clean the internal population size dataframe
    dfPopSizeInternal = dfPopSizeInternal.loc[0:10];

    # Add an additional row to each matrix reflecting the new group that will be
    # added that splits the CA3_pyramidal into two separate populations for the
    # generation of groups

    # dfPopSizeInternal = dfPopSizeInternal.append(dfPopSizeInternal.iloc[9,:],
    #                                              ignore_index = True);
    # dfPopSizeInternal.iloc[9,0] = dfPopSizeInternal.iloc[9,0] + '_a';
    # dfPopSizeInternal.iloc[10,0] = dfPopSizeInternal.iloc[10,0] + '_b';
    # dfPopSizeInternal.iloc[9,1] = np.floor(dfPopSizeInternal.iloc[9,1]/2) + 1;
    # dfPopSizeInternal.iloc[10,1] = np.floor(dfPopSizeInternal.iloc[10,1]/2);

    # Create a data frame that will be used to generate syntax for the groups
    # in the simulation
    dfNeuronTypes = dfNetList[['Cell Types', 'Internal/External', 'Excitatory/Inhibitory']].dropna();
    # rowNum = 1;
    # row_val = dfNeuronTypes.iloc[0,:];
    # dfNeuronTypes = ut.Insert_row_(rowNum, dfNeuronTypes, row_val);
    # dfNeuronTypes.iloc[0,0] = dfNeuronTypes.iloc[0,0] + '_a';
    # dfNeuronTypes.iloc[1,0] = dfNeuronTypes.iloc[1,0] + '_b';
    dfNeuronTypes = dfNeuronTypes.set_index('Cell Types');
    dfNeuronTypes = dfNeuronTypes.reindex(index = dfPopSizeInternal['Neuron Type (Internal)'].append( \
                                                  dfPopSizeExternal['Neuron Type (External)']. \
                                                  reset_index(drop=True).dropna()));
    dfNeuronTypes = dfNeuronTypes.reset_index();
    dfNeuronTypes['popSizes'] = dfPopSizeInternal['Population Size (Internal)'].append( \
                                dfPopSizeExternal['Population Size (External)']).reset_index(
                                drop=True).dropna();
    dfNeuronTypes = dfNeuronTypes.rename(columns = {'index' : 'Cell Types'});

    # For testing purposes, decrease the population sizes by a factor of 50
    #dfNeuronTypes['popSizes'] = dfNeuronTypes['popSizes'] / 50
    #dfNeuronTypes.popSizes = dfNeuronTypes.popSizes.astype(float).round();
    # Generate syntax to define the groups in the simulation
    codeGroups = '';

    # Take the number of neuron types minus two as for now we are not going to be
    # considering the MF and stellate input
    for x in range(len(dfNeuronTypes)-2):
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

    # Add an additional row to each matrix reflecting the new group that will be
    # added that splits the CA3_pyramidal into two separate populations with the
    # intrinsic properties
    # dfIM = dfIM.append(dfIM.iloc[9,:], ignore_index = True);
    # dfIM.iloc[9,0] = dfIM.iloc[9,0] + '_a';
    # dfIM.iloc[10,0] = dfIM.iloc[10,0] + '_b';

    # Create syntax for the IM parameters through a for loop
    codeIM = '';
    for x in range(len(dfIM.iloc[:,0])):
        addCodeIM = r'''sim.setNeuronParameters({grpid}, {C}, {C_std}, {k}, {k_std},
                                                {Vr}, {Vr_std}, {Vt}, {Vt_std}, {a},
                                                {a_std}, {b}, {b_std}, {Vpeak},
                                                {Vpeak_std}, {Vmin}, {Vmin_std},
                                                {d}, {d_std});
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

    # Append the external to internal connection types to connection dataframe
    dfConnMat = pd.concat([dfConnMat, dfConnMatE2I], ignore_index = True);

    # Append the external-internal synaptic weights to the dataframe of
    # internal-internal synaptic weights
    dfSynWeight = pd.concat([dfSynWeight, dfSynWeightE2I], ignore_index = True);

    sortedColumnsConnMat = list(dfConnMat.iloc[0:11,0]);
    sortedColumnsConnMat.insert(0, 'Pre_Post');
    sortedColumnsTMMat = ['CA3_LMR_Targeting_mean', 'CA3_LMR_Targeting_std',
                          'CA3_QuadD_LM_mean', 'CA3_QuadD_LM_std', 'CA3_Axo_Axonic_mean',
                          'CA3_Axo_Axonic_std', 'CA3_Basket_mean', 'CA3_Basket_std',
                          'CA3_BC_CCK_mean', 'CA3_BC_CCK_std', 'CA3_Bistratified_mean',
                          'CA3_Bistratified_std', 'CA3_Ivy_mean', 'CA3_Ivy_std',
                          'CA3_Trilaminar_mean', 'CA3_Trilaminar_std', 'CA3_MFA_ORDEN_mean',
                          'CA3_MFA_ORDEN_std', 'CA3_Pyramidal_a_mean',
                          'CA3_Pyramidal_a_std', 'CA3_Pyramidal_b_mean',
                          'CA3_Pyramidal_b_std'
                          ];
    sortedColumnsTMMat.insert(0, 'Wij');
    dfConnMat = dfConnMat.reindex(columns=sortedColumnsConnMat);
    dfSynWeight = dfSynWeight.reindex(columns=sortedColumnsTMMat);

    # Before generating the syntax for STP, append the external-internal connections
    # to the dataframe that contains internal-internal connections
    dfNumContacts = pd.concat([dfNumContacts, dfNumContactsE2I], ignore_index = True);
    dfSTPU = pd.concat([dfSTPU, dfSTPUE2I], ignore_index = True);
    dfSTPTauU = pd.concat([dfSTPTauU, dfSTPTauUE2I], ignore_index = True);
    dfSTPTauX = pd.concat([dfSTPTauX, dfSTPTauXE2I], ignore_index = True);
    dfSTPTauNT = pd.concat([dfSTPTauNT, dfSTPTauNTE2I], ignore_index = True);

    # Re-order columns so that probability of connections are appropriately set
    # within the for loop
    sortedColumnsTMMat[0] = 'Mij';
    dfMultiplier = dfMultiplier.reindex(columns=sortedColumnsTMMat);
    sortedColumnsTMMat[0] = 'Nij';
    dfNumContacts = dfNumContacts.reindex(columns=sortedColumnsTMMat);
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
    w_max = 1.0;

    # Take the length of the connectivity matrix subtracted by two so that we
    # are not considering the MF and stellate connectivity
    for x in range(len(dfConnMat.iloc[:,0])-2):
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
                if ("CA3_Pyramidal" in dfConnMat.iloc[x,0]):
                    addCodeConn = r'''sim.connect({gin}, {gout}, "random", RangeWeight(0.0f, {winit}f, {wmax}f), {p}f,
                                      RangeDelay(2,4), RadiusRF(-1.0), SYN_PLASTIC, {g}f, 0.0f);
                                   '''.format(gin = dfConnMat.iloc[x,0],
                                              gout = dfConnMat.iloc[y,0],
                                              # Non-zero minimum weights not yet supported by CARLsim4
                                              # wmin = 0.0,
                                              # # winit = dfSynWeight.iloc[x,k+1]*dfNumContacts.iloc[x,k+1],
                                              winit = float(dfMultiplier.iloc[x,k+1]),
                                              # winit = dfSynWeight.iloc[x,k+1],
                                              # # wmax = dfSynWeight.iloc[x,k+1]*dfNumContacts.iloc[x,k+1] + 1.0,
                                              wmax = dfMultiplier.iloc[x,k+1] + 1.0,
                                              # wmax = dfSynWeight.iloc[x,k+1] + 1.0,
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
                                            U_mean = dfSTPU.iloc[x,k+1],
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
                                                  # w = dfSynWeight.iloc[x,k+1]*dfNumContacts.iloc[x,k+1],
                                                  # w = dfSynWeight.iloc[x,k+1]*dfMultiplier.iloc[x,k+1],
                                                  # w = dfSynWeight.iloc[x,k+1],
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
                                                U_mean = dfSTPU.iloc[x,k+1],
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
                                                  # Non-zero minimum weights not yet supported by CARLsim4
                                                  # wmin = 0.0,
                                                  # # winit = dfSynWeight.iloc[x,k+1]*dfNumContacts.iloc[x,k+1],
                                                  winit = float(dfMultiplier.iloc[x,k+1]),
                                                  # winit = dfSynWeight.iloc[x,k+1],
                                                  # # wmax = dfSynWeight.iloc[x,k+1]*dfNumContacts.iloc[x,k+1] + 1.0,
                                                  wmax = dfMultiplier.iloc[x,k+1] + 1.0,
                                                  # wmax = dfSynWeight.iloc[x,k+1] + 1.0,
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
                                                U_mean = dfSTPU.iloc[x,k+1],
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
            #print(k)
        k = 0;
        #print(k)


    # Define variables to be used for excitatory and inhibitory spike time
    # dependent plasticity (STDP) for each group (STDP in CARLsim4 is performed
    # postsynaptically) for the standard exponential curve
#    alphaPlus = 0.1;
#    tauPlus = 20.0;
#    alphaMinus = 0.1;
#    tauMinus = 20.0;
#
#    # Create syntax for STP for each group (in the future this will be modified to
#    # produce syntax for STP between groups)
#
#    codeSTDP = '';
#    for x in range(len(dfSTPU.iloc[:,0])-2):
#        if (dfConnMat.iloc[x,0] == "CA3_Pyramidal"):
#            addCodeESTDP = r'''sim.setESTDP({grpid}, true, STANDARD, ExpCurve({alphaPlus}f, {tauPlus}f, {alphaMinus}f, {tauMinus}f));
#                         '''.format(grpid = dfSTPU.iloc[x,0],
#                                    alphaPlus = 0.01,
#                                    tauPlus = 100.0,
#                                    alphaMinus = 0.01,
#                                    tauMinus = 100.0
#                                   );
#            addCodeISTDP = r'''sim.setISTDP({grpid}, true, STANDARD, ExpCurve(-{alphaPlus}f, {tauPlus}f, {alphaMinus}f, {tauMinus}f));
#                         '''.format(grpid = dfSTPU.iloc[x,0],
#                                    alphaPlus = alphaPlus,
#                                    tauPlus = tauPlus,
#                                    alphaMinus = alphaMinus,
#                                    tauMinus = tauMinus
#                                   );
#            codeSTDP = codeSTDP + addCodeESTDP;
#            codeSTDP = codeSTDP + '\n';
#            codeSTDP = codeSTDP + addCodeISTDP;
#            codeSTDP = codeSTDP + '\n';
#        else:
#            addCodeESTDP = r'''sim.setESTDP({grpid}, true, STANDARD, ExpCurve({alphaPlus}f, {tauPlus}f, -{alphaMinus}f, {tauMinus}f));
#                         '''.format(grpid = dfSTPU.iloc[x,0],
#                                    alphaPlus = alphaPlus,
#                                    tauPlus = tauPlus,
#                                    alphaMinus = alphaMinus,
#                                    tauMinus = tauMinus
#                                   );
#            addCodeISTDP = r'''sim.setISTDP({grpid}, true, STANDARD, ExpCurve(-{alphaPlus}f, {tauPlus}f, {alphaMinus}f, {tauMinus}f));
#                         '''.format(grpid = dfSTPU.iloc[x,0],
#                                    alphaPlus = alphaPlus,
#                                    tauPlus = tauPlus,
#                                    alphaMinus = alphaMinus,
#                                    tauMinus = tauMinus
#                                   );
#            codeSTDP = codeSTDP + addCodeESTDP;
#            codeSTDP = codeSTDP + '\n';
#            codeSTDP = codeSTDP + addCodeISTDP;
#            codeSTDP = codeSTDP + '\n';



    # WILL ADD MORE CODE HERE IN THE FUTURE FOR SPECIFICATION OF DECAY
    # TIME CONSTANTS FOR ION CHANNEL RECEPTORS
    # Create syntax to define the decay constants for excitatory and inhibitory
    # ion channel receptors


    # Create syntax for spike monitors and connection monitors for the groups in
    # the SNN

    # Only monitor the neurons in the local circuit, as we are not currently
    # considering granule and stellate cells
    codeSetSpikeMonitor = '';
    for x in range(len(dfNeuronTypes.iloc[:,0])-2):
        addCodeSetSpikeMonitor = r'''sim.setSpikeMonitor({grpid}, "DEFAULT");
                                 '''.format(grpid = dfNeuronTypes.iloc[x,0]);
        codeSetSpikeMonitor = codeSetSpikeMonitor + addCodeSetSpikeMonitor;
        codeSetSpikeMonitor = codeSetSpikeMonitor + '\n';

        # codeSetConnMonitor = '';
        # for x in range(len(dfNeuronTypes.iloc[:,0])):
        #     for y in range(len(dfNeuronTypes.iloc[:,0])-2):
        #         addCodeSetConnMonitor = r'''sim.setConnectionMonitor({gin}, {gout}, "DEFAULT");
        #                                  '''.format(gin = dfNeuronTypes.iloc[x,0],
        #                                             gout = dfNeuronTypes.iloc[y,0]
        #                                            );
        #         codeSetConnMonitor = codeSetConnMonitor + addCodeSetConnMonitor;
        #         codeSetConnMonitor = codeSetConnMonitor + '\n';



        # Create syntax for the generation of mean Poisson firing rates for the inputs
        # to the SNN
        #
        # # Append population sizes to the Poisson dataframe to be used in generating
        # # syntax
        # dfPoisson['popSizes'] = dfNeuronTypes.iloc[10:12,3].values
        #
        # # Generate the syntax
        # codeSetPoissonRates = '';
        # for x in range(len(dfPoisson.iloc[:,0])):
        #     if (dfPoisson.iloc[x,0] == "DG_Granule"):
        #         addCodeSetPoissonRates = r'''int {grpid}_frate = {rate}f;
        #                                      PoissonRate {ratePointer}({popsize}, true);
        #                                      {ratePointer}.setRates(0.0005f);
        #         	                         sim.setSpikeRate({grpid}, &{ratePointer}, 1);
        #                                   '''.format(ratePointer = dfPoisson.iloc[x,0] +
        #                                              '_rate',
        #                                              grpid = dfPoisson.iloc[x,0],
        #                                              popsize = dfPoisson.iloc[x,3],
        #                                              rate = dfPoisson.iloc[x,1]);
        #         codeSetPoissonRates = codeSetPoissonRates + addCodeSetPoissonRates;
        #         codeSetPoissonRates = codeSetPoissonRates + '\n';
        #     else:
        #         addCodeSetPoissonRates = r'''int {grpid}_frate = {rate}f;
        #                                      PoissonRate {ratePointer}({popsize}, true);
        #                                      {ratePointer}.setRates(0.0f);
        #         	                         sim.setSpikeRate({grpid}, &{ratePointer}, 1);
        #                                   '''.format(ratePointer = dfPoisson.iloc[x,0] +
        #                                              '_rate',
        #                                              grpid = dfPoisson.iloc[x,0],
        #                                              popsize = dfPoisson.iloc[x,3],
        #                                              rate = dfPoisson.iloc[x,1]);
        #         codeSetPoissonRates = codeSetPoissonRates + addCodeSetPoissonRates;
        #         codeSetPoissonRates = codeSetPoissonRates + '\n';


    # Write the syntax generated for the CONFIG State to a header file to be
    # included in the SNN simulation
    f_config_h = "generateCONFIGStateSTP.h";
    with open(f_config_h, 'w') as FOUT:
        FOUT.write(codeGroups)
        FOUT.write(codeIM)
        FOUT.write(codeConn)
        FOUT.write(codeSTP)
        #FOUT.write(codeSTDP)


    # Write the syntax generated for the SETUP State to a header file to be
    # included in the SNN simulation
    f_setup_h = "generateSETUPStateSTP.h";
    with open(f_setup_h, 'w') as FOUT:
        FOUT.write(codeSetSpikeMonitor)
        # FOUT.write(codeSetConnMonitor)
        #FOUT.write(codeSetPoissonRates)


generateSNNSyntax("ca3net_06_24_20_hc.xlsm")
