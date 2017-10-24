package ecjapp.doubles;

import ecjapp.eval.problem.SimpleGroupedProblemForm;
import ec.EvolutionState;
import ec.Individual;
import ec.Problem;
import ec.simple.SimpleFitness;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Test spy SimpleGroupedProblem.  This allows us to see how the population was
 * chunked during grouped evaluation.
 * 
 * @author Eric 'Siggy' Scott
 */
public class TestSimpleGroupedProblem extends Problem implements SimpleGroupedProblemForm {
    private final List<Set<Individual>> chunks = new ArrayList<Set<Individual>>();
    private final Map<Individual, Integer> occurrences = new HashMap<Individual, Integer>();
    
    @Override
    public void evaluate(final EvolutionState state, final Individual[] individuals, final int from, final int to, final int subpopulation, final int threadnum) {
        final Set<Individual> chunk = new HashSet<Individual>();
        for (int i = from; i < to; i++) {
            assignFitness(state, individuals[i]);
            
            // Record that this individual was evaluated in this chunk (for test spy purposes)
            chunk.add(individuals[i]); 
        }
        synchronized(chunks) {
            chunks.add(chunk);
        }
    }
    
    public List<Set<Individual>> getChunks() {
        return new ArrayList<Set<Individual>>() {{
            for (final Set<Individual> chunk : chunks)
                add(new HashSet<Individual>(chunk));
        }};
    }
    
    /**
     *  Set the fitness of the individual to its trait value plus some increment.
     * The increment is determined by how many times we have seen the individual
     * so far, such that if this is the nth occurrence of ind, then 
     * 
     *      n = 0 => fitness = trait
     *      n = 1 => fitness = trait + 1
     *      n = 2 => fitness = trait + 1 + 2
     *      n = 3 => fitness = trait + 1 + 2 + 3
     * 
     * and so on. That is,
     * 
     *      fitness = trait + 0.5*n*(n+1).
     * 
     * The point is to assign distinct, deterministic fitness values to identical 
     * individuals in such a way that the mean and median fitnesses differ,
     * so we can test SimpleGroupedEvaluator's collapse options.
     * 
     * @param state
     * @param ind 
     */
    private void assignFitness(final EvolutionState state, final Individual ind) {
        assert(ind instanceof TestIndividual);
        final int n = (occurrences.containsKey(ind) ? occurrences.get(ind) : 0);
        final float f = ((TestIndividual)ind).getTrait() + 0.5f*n*(n+1);
        ((SimpleFitness)ind.fitness).setFitness(state, f, true);
        occurrences.put(ind, n + 1);
    }
}
