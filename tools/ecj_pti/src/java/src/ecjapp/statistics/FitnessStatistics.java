package ecjapp.statistics;

import ec.EvolutionState;
import ec.Fitness;
import ec.Individual;
import ec.Statistics;
import ec.simple.SimpleFitness;
import ec.util.Output;
import ec.util.Parameter;
import ecjapp.util.Statistics.DoubleAttribute;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * A Statistics that records information about population fitness to a file in
 * CSV format.
 * 
 * If the 'individuals' parameter is true, we measure the fitness of every
 * individual in every generation.  Else we measure the mean, std, min, max
 * and best-so-far.
 * 
 * @author Eric 'Siggy' Scott
 */
public class FitnessStatistics extends Statistics {
    /** log file parameter */
    public static final String P_STATISTICS_FILE = "file";
    /** compress? */
    public static final String P_COMPRESS = "gzip";
    public static final String P_INDIVIDUALS = "individuals";
    
    private boolean compress;
    private String filename;
    /** The index of the log we should write to in state.output. */
    private int statisticslog = 0;  // default to stdout
    
    private long startTime;
    private boolean individuals;
    private Fitness bestSoFar;
    
    // <editor-fold defaultstate="collapsed" desc="Accessors">
    public boolean isCompress() {
        return compress;
    }

    public int getStatisticslog() {
        return statisticslog;
    }
    // </editor-fold>

    public void setup(final EvolutionState state, final Parameter base) {
        assert(state != null);
        assert(base != null);
        super.setup(state, base);
        compress = state.parameters.getBoolean(base.push(P_COMPRESS), null, false);
        individuals = state.parameters.getBoolean(base.push(P_INDIVIDUALS), null, false);

        final File statisticsFile = state.parameters.getFile(
                base.push(P_STATISTICS_FILE), null);
        filename = statisticsFile.getAbsolutePath();

        if (silentFile) {
            statisticslog = Output.NO_LOGS;
        } else {
            try {
                statisticslog = state.output.addLog(statisticsFile, !compress, compress); // !compress because we can't appendOnRestart if we're compressing.
            } catch (final IOException i) {
                state.output.fatal("An IOException occurred while trying to create the log " + statisticsFile + ":\n" + i);
            }
        }

        assert(repOK());
    }

    @Override
    public void postInitializationStatistics(final EvolutionState state) {
        assert(state != null);
        assert(state.output.getLog(statisticslog).filename.getAbsolutePath().equals(filename));
        super.postInitializationStatistics(state);
        
        if (((Integer)state.job[0]).intValue() == 0) // Only print the CSV header on the first job.
            if (individuals)
                state.output.println("job, subpopulation, generation, time, fitness", statisticslog);
            else
                state.output.println("job, subpopulation, generation, time, mean, std, min, max, bsf", statisticslog);
        
        this.startTime = System.currentTimeMillis();
    }
     
    @Override
    public void postEvaluationStatistics(final EvolutionState state) {
        assert(state != null);
        assert(state.output.getLog(statisticslog).filename.getAbsolutePath().equals(filename));
        super.postEvaluationStatistics(state);

        for (int subPop = 0; subPop < state.population.subpops.length; subPop++) {
            if (individuals) {
                    for (int j = 0; j < state.population.subpops[subPop].individuals.length; j++) {
                        final Individual ind = state.population.subpops[subPop].individuals[j];
                        final double fitness = ((SimpleFitness)ind.fitness).fitness();
                        final long time = System.currentTimeMillis() - startTime;
                        state.output.println(String.format("%d, %d, %d, %d, %f", state.job[0], subPop, state.generation, time, fitness), statisticslog);
                    }
            }
            else {
                final List<Individual> subPopList = new ArrayList<Individual>();
                subPopList.addAll(Arrays.asList(state.population.subpops[subPop].individuals));
                
                final double mean = ecjapp.util.Statistics.mean(subPopList, new FitnessAttribute());
                final double std = ecjapp.util.Statistics.std(subPopList, mean, new FitnessAttribute());
                final Fitness min = Collections.min(subPopList, new FitnessComparator()).fitness;
                final Fitness max = Collections.max(subPopList, new FitnessComparator()).fitness;
                if (bestSoFar == null)
                    bestSoFar = max;
                else if (max.betterThan(bestSoFar))
                    bestSoFar = max;
                final long time = System.currentTimeMillis() - startTime;
                state.output.println(String.format("%d, %d, %d, %d, %f, %f, %f, %f, %f", state.job[0], subPop, state.generation, time, mean, std, min.fitness(), max.fitness(), bestSoFar.fitness()), statisticslog);
            }
        }
    }
    
    private static class FitnessComparator implements Comparator<Individual> {
        @Override
        public int compare(final Individual t, final Individual t1) {
            if (t.fitness.betterThan(t1.fitness))
                return 1;
            if (t1.fitness.betterThan(t.fitness))
                return -1;
            return 0;
        }
    }
    
    private static class FitnessAttribute implements DoubleAttribute<Individual> {
        @Override
        public double get(final Individual object) {
            return object.fitness.fitness();
        }
    }

    // <editor-fold defaultstate="collapsed" desc="Standard Methods">
    public final boolean repOK() {
        return statisticslog >= 0
                && P_STATISTICS_FILE != null
                && !P_STATISTICS_FILE.isEmpty()
                && P_COMPRESS != null
                && !P_COMPRESS.isEmpty();
    }
    
    @Override
    public boolean equals(final Object o) {
        if (!(o instanceof FitnessStatistics))
            return false;
        final FitnessStatistics ref = (FitnessStatistics) o;
        return statisticslog == ref.statisticslog
                && compress == ref.compress
                && filename.equals(ref.filename);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 53 * hash + (this.compress ? 1 : 0);
        hash = 53 * hash + (this.filename != null ? this.filename.hashCode() : 0);
        hash = 53 * hash + this.statisticslog;
        return hash;
    }
    
    @Override
    public String toString() {
        return String.format("[%s: filename=%s, statisticslog=%d, compress=%b]", this.getClass().getSimpleName(), filename, statisticslog, compress);
    }
    // </editor-fold>
    
}
