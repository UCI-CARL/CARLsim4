package ecjapp.eval.problem;

import ec.EvolutionState;
import ec.vector.DoubleVectorIndividual;
import java.util.List;

/**
 *
 * @author Eric O. Scott
 */
public interface MultiPopGroupedProblemForm {
    /** Evaluates a subset of a Population.  The subset may span one or more
     * SubPopulations.
     * 
     * Effects: Sets the fitness attribute of all the individuals between 
     * from and to (not including to), where from and to are interpreted as indices
     * in a *combined* array that includes the individuals from all SubPopulations.
     * 
     * @param state The state of the current evolutionary simulation.
     * @param from The index of the first individual to evaluate.
     * @param to The index of the last individual to evaluate.
     * @param threadnum ID of the thread this method was called from.
     */
    public void evaluate(final EvolutionState state, final List<DoubleVectorIndividual> individuals, final List<Integer> subPopulations, final int threadnum);
}
