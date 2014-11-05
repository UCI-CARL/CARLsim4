package ecjapp.statistics;

import ec.EvolutionState;
import ec.Statistics;
import ec.simple.SimpleFitness;
import ec.util.Parameter;
import ec.vector.DoubleVectorIndividual;
import java.io.File;
import java.io.IOException;

/**
 * Prints the genomes of all individuals in CSV format at every generation.
 * 
 * @author Eric 'Siggy' Scott
 */
public class DoubleVectorGenomeStatistics extends Statistics {
    public static final String P_STATISTICS_FILE = "file";
    public static final String P_COMPRESS = "compress";
    /* Parameters that specifies the parameter we should take the genome size from. */
    public static final String P_P_LENGTH = "pVectorLength";
    public static final String P_INIT_ONLY = "initOnly";
    
    private int log = 0; // 0 by default means stdout
    private int vectorLength;
    private boolean initOnly;
    
    @Override
    public void setup(final EvolutionState state, final Parameter base) {
        assert(state != null);
        assert(base != null);
        super.setup(state, base);
        final File statisticsFile = state.parameters.getFile(base.push(P_STATISTICS_FILE),null);
        if (statisticsFile!=null) try {
            final boolean compress = state.parameters.getBoolean(base.push(P_COMPRESS),null,false);
            log = state.output.addLog(statisticsFile, !compress, compress);
        }
        catch (IOException i) {
            state.output.fatal("An IOException occurred trying to create the log " + statisticsFile + ":\n" + i);
        }
        // else we’ll just keep the log at 0, which is stdout
        
        initOnly = state.parameters.getBoolean(base.push(P_INIT_ONLY), null, false);
	final String pVectorLength = state.parameters.getString(base.push(P_P_LENGTH), null);
	vectorLength = state.parameters.getInt(new Parameter(pVectorLength), null, 0);
        printHeader(state);
    }
    
    private void printHeader(final EvolutionState state) {
        assert(state != null);
        final StringBuilder sb = new StringBuilder();
        sb.append("generation,subPopulation,fitness");
        for (int i = 0; i < vectorLength; i++)
            sb.append(",V").append(i);
        state.output.println(sb.toString(), log);
    }
    
    @Override
    public void postEvaluationStatistics(final EvolutionState state) {
        assert(state != null);
        if (!initOnly || state.generation == 0)
            doStatistics(state);
    }
    
    private void doStatistics(final EvolutionState state) {
        assert(state != null);
        for (int subPop = 0; subPop < state.population.subpops.length; subPop++) {
            for (int i = 0; i < state.population.subpops[subPop].individuals.length; i++) {
                final DoubleVectorIndividual ind = (DoubleVectorIndividual) state.population.subpops[subPop].individuals[i];
                
                final StringBuilder sb = new StringBuilder();
                sb.append(state.generation).append(",").append(subPop).append(",").append(((SimpleFitness)ind.fitness).fitness());
                for (int j = 0; j < ind.genome.length; j++)
                    sb.append(",").append(Double.toString(ind.genome[j]));
                state.output.println(sb.toString(), log);
            }
        }
    }
}
