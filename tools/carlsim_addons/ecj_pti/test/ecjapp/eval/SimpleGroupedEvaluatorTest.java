package ecjapp.eval;

import ecjapp.eval.problem.SimpleGroupedProblemForm;
import ecjapp.eval.SimpleGroupedEvaluator;
import ecjapp.doubles.TestIndividual;
import ecjapp.doubles.TestSimpleGroupedProblem;
import ec.Evaluator;
import ec.EvolutionState;
import ec.Evolve;
import ec.Individual;
import ec.Population;
import ec.Subpopulation;
import ec.simple.SimpleEvolutionState;
import ec.util.BadParameterException;
import ec.util.Output;
import ec.util.Parameter;
import ec.util.ParameterDatabase;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import ecjapp.util.Misc;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class SimpleGroupedEvaluatorTest {
    private final static String PROBLEM_DOUBLE_NAME = "ecjapp.doubles.TestSimpleGroupedProblem";
    private final static String BAD_PROBLEM_DOUBLE_NAME = "ecjapp.doubles.TestSimpleProblem";
    private EvolutionState state;
    private SimpleGroupedEvaluator sut;
    
    public SimpleGroupedEvaluatorTest() { }
    
    // <editor-fold defaultstate="collapsed" desc="Fixture">
    @Before
    public void setUp() {
        this.state = getFreshState();
        this.sut = new SimpleGroupedEvaluator(); // Each test needs to call setup (perhaps after tweaking the parameters)
    }
    
    private static EvolutionState getFreshState() {
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
        state.population.subpops[0].individuals = getIndividuals();
        return state;
    }
    
    private static ParameterDatabase getParams() {
        final ParameterDatabase parameters = new ParameterDatabase();
        // Parameters needed by Evaluator.setup()
        parameters.set(new Parameter("eval." + Evaluator.P_PROBLEM), PROBLEM_DOUBLE_NAME);
        //parameters.set(new Parameter("eval." + Evaluator.P_MASTERPROBLEM), null); // None
        //parameters.set(new Parameter("eval." + Evaluator.P_IAMSLAVE), null); // None
        
        // Parameters needed by SimpleGroupedEvaluator.setup()
        parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CLONE_PROBLEM), "false");
        parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "1");
        //parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), null); // None
        //parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), null); // None
        return parameters;
    }
    
    private static Individual[] getIndividuals() {
        final Individual[] individuals = new Individual[4];
        individuals[0] = new TestIndividual(0);
        individuals[1] = new TestIndividual(1);
        individuals[2] = new TestIndividual(2);
        individuals[3] = new TestIndividual(3);
        return individuals;
    }
    
    @After
    public void tearDown() {
        this.state = null;
        this.sut = null;
    }
    //</editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Tests">
    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test
    public void testSetup() {
        System.out.println("setup (default)");
        
        this.sut.setup(state, new Parameter("eval"));
        
        (new ExpectedState()).assertMatches(sut);
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test
    public void testSetupNumTests0() {
        System.out.println("setup (numTests = 0)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "0");
        
        this.sut.setup(state, new Parameter("eval"));
        
        (new ExpectedState()).assertMatches(sut);
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test
    public void testSetupChunkSizeAuto() {
        System.out.println("setup (chunkSize = auto)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), "auto");
        
        this.sut.setup(state, new Parameter("eval"));
        
        (new ExpectedState()).assertMatches(sut);
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test(expected = Output.OutputExitException.class)
    public void testSetupBadChunkSize() {
        System.out.println("setup (chunkSize = 0)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), "0");
        
        this.sut.setup(state, new Parameter("eval"));
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test
    public void testSetupNumTests2Default() {
        System.out.println("setup (numTests = 2, mergeMethod = null)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "2");
        
        this.sut.setup(state, new Parameter("eval"));
        
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 2;
        expected.assertMatches(sut);
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test
    public void testSetupNumTests2Mean() {
        System.out.println("setup (numTests = 2, mergeMethod = mean)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "2");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), String.valueOf(SimpleGroupedEvaluator.V_MEAN));
        
        this.sut.setup(state, new Parameter("eval"));
        
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 2;
        expected.assertMatches(sut);
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test
    public void testSetupNumTests2Median() {
        System.out.println("setup (numTests = 2, mergeMethod = median)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "2");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), String.valueOf(SimpleGroupedEvaluator.V_MEDIAN));
        
        this.sut.setup(state, new Parameter("eval"));
        
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 2;
        expected.mergeForm = SimpleGroupedEvaluator.MERGE_MEDIAN;
        expected.assertMatches(sut);
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test
    public void testSetupNumTests2Best() {
        System.out.println("setup (numTests = 2, mergeMethod = median)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "2");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), String.valueOf(SimpleGroupedEvaluator.V_BEST));
        
        this.sut.setup(state, new Parameter("eval"));
        
        
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 2;
        expected.mergeForm = SimpleGroupedEvaluator.MERGE_BEST;
        expected.assertMatches(sut);
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test(expected = Output.OutputExitException.class)
    public void testSetupNumTests2BadMerge() {
        System.out.println("setup (numTests = 2, mergeMethod = median)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "2");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), "mode");
        
        this.sut.setup(state, new Parameter("eval"));
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test(expected = Output.OutputExitException.class)
    public void testSetupBadProblem() {
        System.out.println("setup (bad problem type)");
        state.parameters.set(new Parameter("eval." + Evaluator.P_PROBLEM), BAD_PROBLEM_DOUBLE_NAME);
        
        this.sut.setup(state, new Parameter("eval"));
    }

    /** Test of setup method, of class SimpleGroupedEvaluator. */
    @Test(expected = Output.OutputExitException.class)
    public void testSetupBadBreedThreads() {
        System.out.println("setup (multiple breeding threads with no cloning)");
        state.breedthreads = 5;
        
        this.sut.setup(state, new Parameter("eval"));
    }

    /** Test of expand method, of class SimpleGroupedEvaluator. */
    @Test
    public void testExpand() {
        System.out.println("expand");
        final Individual[] originalPopulation = state.population.subpops[0].individuals;
        assertEquals(getIndividuals().length, originalPopulation.length);
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "4");
        
        this.sut.setup(state, new Parameter("eval"));
        this.sut.expand(state);
        
        final Map<Individual, Integer> counts = Misc.countOccurrences(state.population.subpops[0].individuals);
        for (Individual ind : originalPopulation)
            assertEquals(4, (int) counts.get(ind));
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 4;
        expected.assertMatches(sut);
    }

    /** Test of contract method, of class SimpleGroupedEvaluator. */
    @Test
    public void testContractMean() {
        System.out.println("contract (mean)");
        final Individual[] originalPopulation = state.population.subpops[0].individuals;
        assertEquals(getIndividuals().length, originalPopulation.length);
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "4");
        
        exerciseContract();
        
        final Map<Individual, Integer> counts = Misc.countOccurrences(state.population.subpops[0].individuals);
        final double meanOffset = 0.25*(0 + 1 + 3 + 6);
        for (Individual ind : originalPopulation) {
            assertEquals(1, (int) counts.get(ind));
            assertEquals(((TestIndividual)ind).getTrait() + meanOffset, ind.fitness.fitness(), 0.000001);
        }
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 4;
        expected.assertMatches(sut);
    }

    /** Test of contract method, of class SimpleGroupedEvaluator. */
    @Test
    public void testContractMedian() {
        System.out.println("contract (median)");
        final Individual[] originalPopulation = state.population.subpops[0].individuals;
        assertEquals(getIndividuals().length, originalPopulation.length);
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "4");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), SimpleGroupedEvaluator.V_MEDIAN);
        
        exerciseContract();
        
        final Map<Individual, Integer> counts = Misc.countOccurrences(state.population.subpops[0].individuals);
        final double medianOffset = 0.5*(1 + 3);
        for (Individual ind : originalPopulation) {
            assertEquals(1, (int) counts.get(ind));
            assertEquals(((TestIndividual)ind).getTrait() + medianOffset, ind.fitness.fitness(), 0.000001);
        }
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 4;
        expected.mergeForm = SimpleGroupedEvaluator.MERGE_MEDIAN;
        expected.assertMatches(sut);
    }

    /** Test of contract method, of class SimpleGroupedEvaluator. */
    @Test
    public void testContractBest() {
        System.out.println("contract (best)");
        final Individual[] originalPopulation = state.population.subpops[0].individuals;
        assertEquals(getIndividuals().length, originalPopulation.length);
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "4");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_MERGE), SimpleGroupedEvaluator.V_BEST);
        
        exerciseContract();
        
        final Map<Individual, Integer> counts = Misc.countOccurrences(state.population.subpops[0].individuals);
        final double bestOffset = 6;
        for (Individual ind : originalPopulation) {
            assertEquals(1, (int) counts.get(ind));
            assertEquals(((TestIndividual)ind).getTrait() + bestOffset, ind.fitness.fitness(), 0.000001);
        }
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.numTests = 4;
        expected.mergeForm = SimpleGroupedEvaluator.MERGE_BEST;
        expected.assertMatches(sut);
    }

    private void exerciseContract() throws BadParameterException {
        this.sut.setup(state, new Parameter("eval"));
        this.sut.expand(state);
        final SimpleGroupedProblemForm problem = (SimpleGroupedProblemForm) sut.p_problem;
        problem.evaluate(state, state.population.subpops[0].individuals, 0, state.population.subpops[0].individuals.length, 0, 0);
        this.sut.contract(state);
    }

    /** Test of evaluatePopulation method, of class SimpleGroupedEvaluator. */
    @Test
    public void testEvaluatePopulationAuto() {
        System.out.println("evaluatePopulation (auto, 1 thread, numtests 1)");
        this.sut.setup(state, new Parameter("eval"));
        
        sut.evaluatePopulation(state);
        
        final List<Set<Individual>> chunks = ((TestSimpleGroupedProblem)sut.p_problem).getChunks();
        assertEquals(1, chunks.size());
        assertEquals(4, chunks.get(0).size());
        assertTrue(chunks.get(0).contains(new TestIndividual(0)));
        assertTrue(chunks.get(0).contains(new TestIndividual(1)));
        assertTrue(chunks.get(0).contains(new TestIndividual(2)));
        assertTrue(chunks.get(0).contains(new TestIndividual(3)));
        final Individual[] inds = state.population.subpops[0].individuals;
        for (final Individual ind : inds)
            assertEquals(((TestIndividual)ind).getTrait(), ind.fitness.fitness(), 0.000001);
        (new ExpectedState()).assertMatches(sut); // Check for side effects
    }

    /** Test of evaluatePopulation method, of class SimpleGroupedEvaluator. */
    @Test
    public void testEvaluatePopulationChunksize2() {
        System.out.println("evaluatePopulation (chunksize 2, 1 thread, numtests 1)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), "2");
        this.sut.setup(state, new Parameter("eval"));
        
        sut.evaluatePopulation(state);
        
        final List<Set<Individual>> chunks = ((TestSimpleGroupedProblem)sut.p_problem).getChunks();
        assertEquals(2, chunks.size());
        assertEquals(2, chunks.get(0).size());
        assertEquals(2, chunks.get(1).size());
        assertTrue(chunks.get(0).contains(new TestIndividual(0)));
        assertTrue(chunks.get(0).contains(new TestIndividual(1)));
        assertTrue(chunks.get(1).contains(new TestIndividual(2)));
        assertTrue(chunks.get(1).contains(new TestIndividual(3)));
        final Individual[] inds = state.population.subpops[0].individuals;
        for (final Individual ind : inds)
            assertEquals(((TestIndividual)ind).getTrait(), ind.fitness.fitness(), 0.000001);
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.chunkSize = 2;
        expected.assertMatches(sut);
    }

    /** Test of evaluatePopulation method, of class SimpleGroupedEvaluator. */
    @Test
    public void testEvaluatePopulationChunksize3() {
        System.out.println("evaluatePopulation (chunksize 3, 1 thread, numtests 1)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), "3");
        this.sut.setup(state, new Parameter("eval"));
        
        sut.evaluatePopulation(state);
        
        final List<Set<Individual>> chunks = ((TestSimpleGroupedProblem)sut.p_problem).getChunks();
        assertEquals(2, chunks.size());
        assertEquals(3, chunks.get(0).size());
        assertEquals(1, chunks.get(1).size());
        assertTrue(chunks.get(0).contains(new TestIndividual(0)));
        assertTrue(chunks.get(0).contains(new TestIndividual(1)));
        assertTrue(chunks.get(0).contains(new TestIndividual(2)));
        assertTrue(chunks.get(1).contains(new TestIndividual(3)));
        final Individual[] inds = state.population.subpops[0].individuals;
        for (final Individual ind : inds)
            assertEquals(((TestIndividual)ind).getTrait(), ind.fitness.fitness(), 0.000001);
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.chunkSize = 3;
        expected.assertMatches(sut);
    }

    /** Test of evaluatePopulation method, of class SimpleGroupedEvaluator. */
    @Test
    public void testEvaluatePopulationChunksize3Numtests2() {
        System.out.println("evaluatePopulation (chunksize 3, 1 thread, numtests 2)");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), "3");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "2");
        this.sut.setup(state, new Parameter("eval"));
        
        sut.evaluatePopulation(state);
        
        // The spied chunks recorded the expanded population
        final List<Set<Individual>> chunks = ((TestSimpleGroupedProblem)sut.p_problem).getChunks();
        assertEquals(3, chunks.size());
        assertEquals(2, chunks.get(0).size());
        assertEquals(2, chunks.get(1).size());
        assertEquals(1, chunks.get(2).size());
        assertTrue(chunks.get(0).contains(new TestIndividual(0)));
        assertTrue(chunks.get(0).contains(new TestIndividual(1)));
        assertTrue(chunks.get(1).contains(new TestIndividual(1)));
        assertFalse(chunks.get(1).contains(new TestIndividual(0)));
        assertTrue(chunks.get(1).contains(new TestIndividual(2)));
        assertFalse(chunks.get(2).contains(new TestIndividual(2)));
        assertTrue(chunks.get(2).contains(new TestIndividual(3)));
        final Individual[] inds = state.population.subpops[0].individuals;
        for (final Individual ind : inds)
            assertEquals(((TestIndividual)ind).getTrait() + 0.5, ind.fitness.fitness(), 0.000001);
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.chunkSize = 3;
        expected.numTests = 2;
        expected.assertMatches(sut);
    }

    /** Test of evaluatePopulation method, of class SimpleGroupedEvaluator. */
    @Test
    public void testEvaluatePopulationChunksize3Evalthreads2Numtests3() {
        System.out.println("evaluatePopulation (chunksize 3, 2 threads, numtests 3)");
        state.evalthreads = 2;
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), "3");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "3");
        this.sut.setup(state, new Parameter("eval"));
        
        sut.evaluatePopulation(state);
        
        // The spied chunks recorded the expanded population
        final List<Set<Individual>> chunks = ((TestSimpleGroupedProblem)sut.p_problem).getChunks();
        assertEquals(4, chunks.size());
        assertEquals(1, chunks.get(0).size());
        assertEquals(1, chunks.get(1).size());
        assertEquals(1, chunks.get(2).size());
        assertEquals(1, chunks.get(3).size());
        final Individual[] inds = state.population.subpops[0].individuals;
        for (final Individual ind : inds)
            assertEquals(((TestIndividual)ind).getTrait() + 4.0/3, ind.fitness.fitness(), 0.000001);
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.chunkSize = 3;
        expected.numTests = 3;
        expected.assertMatches(sut);
    }

    /** Test of evaluatePopulation method, of class SimpleGroupedEvaluator. */
    @Test
    public void testEvaluatePopulationChunksize3Evalthreads2NumTests2() {
        System.out.println("evaluatePopulation (chunksize 3, 2 threads, numtests 2)");
        state.evalthreads = 2;
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CHUNK_SIZE), "3");
        state.parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "2");
        this.sut.setup(state, new Parameter("eval"));
        
        sut.evaluatePopulation(state);
        
        // The spied chunks recorded the expanded population
        final List<Set<Individual>> chunks = ((TestSimpleGroupedProblem)sut.p_problem).getChunks();
        assertEquals(3, chunks.size());
        final Individual[] inds = state.population.subpops[0].individuals;
        for (final Individual ind : inds)
            assertEquals(((TestIndividual)ind).getTrait() + 0.5, ind.fitness.fitness(), 0.000001);
        // Check for side effects
        final ExpectedState expected = new ExpectedState();
        expected.chunkSize = 3;
        expected.numTests = 2;
        expected.assertMatches(sut);
    }

    /** Test of evaluatePopulation method, of class SimpleGroupedEvaluator. */
    @Test(expected = AssertionError.class)
    public void testEvaluatePopulationThreads0() {
        System.out.println("evaluatePopulation (0 threads)");
        state.evalthreads = 0;
        this.sut.setup(state, new Parameter("eval"));
        
        sut.evaluatePopulation(state);
    }
    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Verification Helpers">
    /** A convenience class for doing full state verification.  This is an
     * ad-hoc way around the fact that ECJ objects don't follow the equals() contract.
     */
    public static class ExpectedState {
        String p_problem_name;
        int numTests;
        int mergeForm;
        boolean cloneProblem;
        int chunkSize;
        
        public ExpectedState() {
            this.p_problem_name = PROBLEM_DOUBLE_NAME;
            this.numTests = 1;
            this.mergeForm = SimpleGroupedEvaluator.MERGE_MEAN;
            this.cloneProblem = false;
            this.chunkSize = SimpleGroupedEvaluator.C_AUTO;
        }
    
        public void assertMatches(final SimpleGroupedEvaluator sut) {
            assertEquals(p_problem_name, sut.p_problem.getClass().getCanonicalName());
            assertEquals(numTests, sut.numTests);
            assertEquals(mergeForm, sut.mergeForm);
            assertEquals(cloneProblem, sut.cloneProblem);
            assertEquals(chunkSize, sut.chunkSize);
        }
    }
    //</editor-fold>
}