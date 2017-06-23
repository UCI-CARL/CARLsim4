package ecjapp.eval;

import ec.Evaluator;
import ec.EvolutionState;
import ec.Evolve;
import ec.Individual;
import ec.Population;
import ec.Subpopulation;
import ec.simple.SimpleEvolutionState;
import ec.util.Parameter;
import ec.util.ParameterDatabase;
import ec.vector.DoubleVectorIndividual;
import ecjapp.doubles.TestObjective;
import ecjapp.eval.problem.CommandController;
import ecjapp.eval.problem.CommandProblem;
import ecjapp.util.Option;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * An integration test using a CommandProblem with SimpleGroupedEvaluator.
 * 
 * @author Eric 'Siggy' Scott
 */
public class SimpleGroupedEvaluatorCommandProblemTest {
    private EvolutionState state;
    private SimpleGroupedEvaluator evaluator;
    private final static String COMMAND_PROBLEM_NAME = "ecjapp.eval.problem.CommandProblem";
    private final static String TEST_COMMAND = "/bin/cat";
    private final static String TEST_OBJECTIVE = "ecjapp.doubles.TestObjective";
    private final static Parameter BASE = new Parameter("eval");
    
    // <editor-fold defaultstate="collapsed" desc="Fixture">
    @Before
    public void setUp() {
        this.state = getState();
        this.evaluator = new SimpleGroupedEvaluator(); // Each test needs to call setup (perhaps after tweaking the parameters)
    }
    
    private static EvolutionState getState() {
        final EvolutionState state = new SimpleEvolutionState();
        // Set up just the parameters needed for the SUT to initialize itself
        state.parameters = getParams();
        state.evalthreads = 1;
        
        // We need errors to throw exceptions (rather than exit the program) so we can verify them.
        state.output = Evolve.buildOutput();
        state.output.setThrowsErrors(true); 
        
        // Set up a population
        state.population = new Population();
        state.population.subpops = new Subpopulation[] { new Subpopulation() };
        state.population.subpops[0].individuals = getPopulation();
        return state;
    }
    
    private static ParameterDatabase getParams() {
        final ParameterDatabase parameters = new ParameterDatabase();
        // Parameters needed by Evaluator.setup()
        parameters.set(new Parameter("eval." + Evaluator.P_PROBLEM), COMMAND_PROBLEM_NAME);
        //parameters.set(new Parameter("eval." + Evaluator.P_MASTERPROBLEM), null); // None
        //parameters.set(new Parameter("eval." + Evaluator.P_IAMSLAVE), null); // None
        
        // Parameters needed by SimpleGroupedEvaluator.setup()
        parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CLONE_PROBLEM), "false");
        parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "1");
        //parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), null); // None
        //parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), null); // None
        
        // Parameters needed by CommandProblem.setup()
        parameters.set(new Parameter("eval.problem." + CommandProblem.P_SIMULATION_COMMAND_PATH), TEST_COMMAND);
        parameters.set(new Parameter("eval.problem." + CommandProblem.P_OBJECTIVE_FUNCTION), TEST_OBJECTIVE);
        return parameters;
    }

    private static Individual[] getPopulation() {
        return new Individual[] {
            new DoubleVectorIndividual() {{ genome = new double[] {9.18,   4.85,    .19,   8.90,   5.04,   7.39,   1.85,   2.27,   7.79,   2.39 }; }},
            new DoubleVectorIndividual() {{ genome = new double[] {5.49,   3.60,   3.61,   3.70,   7.78,   3.74,   2.26,   3.90,   6.01,   8.87 }; }},
            new DoubleVectorIndividual() {{ genome = new double[] {2.81,   3.56,   9.69,   5.42,   7.31,    .36,   7.87,   2.79,   4.57,   8.13 }; }},
            new DoubleVectorIndividual() {{ genome = new double[] {8.97,   2.02,   7.43,   1.14,   9.79,   6.55,   9.41,   6.73,   4.98,   2.89 }; }},
            new DoubleVectorIndividual() {{ genome = new double[] {5.08,   6.58,   8.29,   6.83,   5.63,   8.69,   7.48,    .34,   1.39,    .88  }; }}
        };
    }
    
    // </editor-fold>
    
    
    // <editor-fold defaultstate="collapsed" desc="Tests">
    @Test
    public void testSetup() {
        System.out.println("setup");
        evaluator.setup(state, BASE);
        
        final SimpleGroupedEvaluatorTest.ExpectedState expectedEvaluator = new SimpleGroupedEvaluatorTest.ExpectedState() {{ p_problem_name = CommandProblem.class.getName(); }};
        expectedEvaluator.assertMatches(evaluator);
        
        assertTrue(evaluator.p_problem.getClass() == CommandProblem.class);
        final CommandProblem problem = (CommandProblem) evaluator.p_problem;
        final CommandController expectedController = new CommandController(TEST_COMMAND, Option.NONE, Option.NONE);
        assertEquals(expectedController, problem.getCommandController());
        assertEquals(new TestObjective(), problem.getObjective());
    }
    
    @Test
    public void testEvaluate() {
        System.out.println("evaluate");
        
        fail("Unwritten test.");
    }
    
    // </editor-fold>
}
