package ecjapp.eval.problem;

import ec.EvolutionState;
import ec.Individual;
import ec.util.Parameter;

/**
 * The P_COMMAND_ARGUMENTS parameter of CommandProblem allows us to pass
 * statically defined CLI arguments to the external command.  An
 * DynamicArguments is used when the arguments we need to pass are dynamic,
 * i.e. they depend in some way on the state of ECJ when the chunk of
 * individuals is being fired off.
 * 
 * @author Eric 'Siggy' Scott
 */
public interface DynamicArguments {
    void setup(final EvolutionState state, final Parameter base);
    String get(final EvolutionState state, final Individual[] individuals, final int from, final int to, final int subpopulation, final int threadnum);
}
