package ecjapp.eval.problem;

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
import ecjapp.util.Option;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class CommandProblemTest {
    private EvolutionState state;
    private CommandProblem sut;
    private final static String TEST_COMMAND = "/bin/cat";
    private final static String TEST_OBJECTIVE = "ecjapp.doubles.TestObjective";
    
    public CommandProblemTest() {
    }
    
    // <editor-fold defaultstate="collapsed" desc="Fixture">
    @Before
    public void setUp() {
        this.state = getState();
        this.sut = new CommandProblem();
        // sut.setup() will be called by each test (perhaps after tweaking the parameters).
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
    
    /** Test of setup method, of class CommandProblem. */
    @Test (expected = IllegalArgumentException.class)
    public void testSetup_IAE1() {
        System.out.println("setup (IAE)");
        sut.setup(null, new Parameter("eval.problem"));
    }
    
    /** Test of setup method, of class CommandProblem. */
    @Test (expected = IllegalArgumentException.class)
    public void testSetup_IAE2() {
        System.out.println("setup (IAE)");
        sut.setup(state, null);
    }
    
    /** Test of setup method, of class CommandProblem. */
    @Test (expected = IllegalArgumentException.class)
    public void testSetup_IAE3() {
        System.out.println("setup (IAE)");
        state.parameters = null;
        sut.setup(state, new Parameter("eval.problem"));
    }
    
    /** Test of setup method, of class CommandProblem. */
    @Test
    public void testSetup_1() {
        System.out.println("setup");
        sut.setup(state, new Parameter("eval.problem"));
        final CommandController expectedController = new CommandController(TEST_COMMAND, Option.NONE, Option.NONE);
        assertEquals(expectedController, sut.getCommandController());
        assertEquals(new TestObjective(), sut.getObjective());
        assertTrue(sut.repOK());
    }
    
    /** Test of setup method, of class CommandProblem. */
    @Test
    public void testSetup_2() {
        System.out.println("setup");
        state.parameters.set(new Parameter("eval.problem." + CommandProblem.P_SIMULATION_COMMAND_ARGUMENTS), "-e");
        sut.setup(state, new Parameter("eval.problem"));
        final CommandController expectedController = new CommandController(TEST_COMMAND, new Option<String>("-e"), Option.NONE);
        assertEquals(expectedController, sut.getCommandController());
        assertEquals(new TestObjective(), sut.getObjective());
        assertTrue(sut.repOK());
    }

    /** Test of evaluate method, of class CommandProblem. */
    @Test
    public void testEvaluate() {
        System.out.println("evaluate");
        sut.setup(state, new Parameter("eval.problem"));
        final int from = 1;
        final int to = 4;
        final int subpopulation = 0;
        final int threadnum = 0;
        
        sut.evaluate(state, state.population.subpops[0].individuals, from, to, subpopulation, threadnum);
        
        assertEquals(state.population.subpops[0].individuals[0].fitness, null);
        assertEquals(state.population.subpops[0].individuals[1].fitness.fitness(), 8.87f, 0.000001);
        assertEquals(state.population.subpops[0].individuals[2].fitness.fitness(), 9.69f, 0.000001);
        assertEquals(state.population.subpops[0].individuals[3].fitness.fitness(), 9.79f, 0.000001);
        assertEquals(state.population.subpops[0].individuals[4].fitness, null);
        
        assertTrue(sut.repOK());
    }
    
    /** Test of toString method, of class CommandProblem. */
    @Test
    public void testToString() {
        System.out.println("evaluate");
        sut.setup(state, new Parameter("eval.problem"));
        final String expected = String.format("[CommandProblem: controller=[%s, objective=[TestObjective]]", sut.getCommandController().toString());
        assertEquals(expected, sut.toString());
        assertTrue(sut.repOK());
    }
    
    @Test
    public void testEqualsAndHashCode() {
        System.out.println("equals and hashCode");
        
        fail("Unimplemented test");
    }
    //</editor-fold>
}