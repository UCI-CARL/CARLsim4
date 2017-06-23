package ecjapp.eval.problem;

import ec.EvolutionState;
import ec.Individual;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public interface SimpleGroupedProblemForm {
    /** Evaluates an array of individuals together and updates their fitness.
     * 
     * Effects: Sets the fitness attribute of all the individuals between 
     * from and to (not including to).
     * 
     * @param state The state of the current evolutionary simulation.
     * @param individuals The individuals in the subpopulation currently under evaluation.
     * @param from The index of the first individual to evaluate.
     * @param to The index of the last individual to evaluate.
     * @param subpopulation Index of the subpopulation represented by individuals.
     * @param threadnum ID of the thread this method was called from.
     */
    public void evaluate(final EvolutionState state, final Individual[] individuals, final int from, final int to, final int subpopulation, final int threadnum);
}
