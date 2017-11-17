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
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Eric O. Scott
 */
public class MultiPopSimpleGroupedEvaluatorTest {
    private final static String PROBLEM_DOUBLE_NAME = "ecjapp.doubles.TestSimpleGroupedProblem";
    private final static Parameter BASE = new Parameter("eval");
    
    private EvolutionState state;
    private SimpleGroupedEvaluator evaluator;
    
    public MultiPopSimpleGroupedEvaluatorTest() {
    }
    
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
        state.population = getPopulation();
        return state;
    }
    
    private static ParameterDatabase getParams() {
        final ParameterDatabase parameters = new ParameterDatabase();
        // Parameters needed by Evaluator.setup()
        parameters.set(new Parameter("eval." + Evaluator.P_PROBLEM), PROBLEM_DOUBLE_NAME);
        
        // Parameters needed by SimpleGroupedEvaluator.setup()
        parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_CLONE_PROBLEM), "false");
        parameters.set(new Parameter("eval." + SimpleGroupedEvaluator.P_NUM_TESTS), "1");
        return parameters;
    }

    private static Population getPopulation() {
        final Population p = new Population();
        p.subpops = new Subpopulation[] { new Subpopulation(), new Subpopulation(), new Subpopulation() };
        p.subpops[0].individuals =  getSubPopulation(1, 5);
        p.subpops[1].individuals =  getSubPopulation(6, 5);
        p.subpops[2].individuals =  getSubPopulation(11, 5);
        return p;
    }
    
    private static Individual[] getSubPopulation(final int startGene, final int numInds) {
        assert(numInds > 0);
        final DoubleVectorIndividual[] inds = new DoubleVectorIndividual[numInds];
        int gene = startGene;
        for (int i = 0; i < numInds; i++) {
            inds[i] = new DoubleVectorIndividual();
            inds[i].genome = new double[] { gene };
            gene++;
        }
        return inds;
    }
    
    // </editor-fold>

    /** Test of getPopChunk method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetPopChunk1() {
        System.out.println("getPopChunk");
        Population population = getPopulation();
        final int from = 0;
        final int to = 15;
        final List<DoubleVectorIndividual> expResult = new ArrayList<DoubleVectorIndividual>();
        for (int i = 1; i < 16; i++) {
            final DoubleVectorIndividual ind = new DoubleVectorIndividual();
            ind.genome = new double[] { i };
            expResult.add(ind);
        }
        final List<DoubleVectorIndividual> result = MultiPopSimpleGroupedEvaluator.getPopChunk(population, from, to);
        assertEquals(expResult, result);
    }

    /** Test of getPopChunk method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetPopChunk2() {
        System.out.println("getPopChunk");
        Population population = getPopulation();
        final int from = 0;
        final int to = 14;
        final List<DoubleVectorIndividual> expResult = new ArrayList<DoubleVectorIndividual>();
        for (int i = 1; i < 15; i++) {
            final DoubleVectorIndividual ind = new DoubleVectorIndividual();
            ind.genome = new double[] { i };
            expResult.add(ind);
        }
        final List<DoubleVectorIndividual> result = MultiPopSimpleGroupedEvaluator.getPopChunk(population, from, to);
        assertEquals(expResult, result);
    }

    /** Test of getPopChunk method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetPopChunk3() {
        System.out.println("getPopChunk");
        Population population = getPopulation();
        final int from = 4;
        final int to = 11;
        final List<DoubleVectorIndividual> expResult = new ArrayList<DoubleVectorIndividual>();
        for (int i = 5; i < 12; i++) {
            final DoubleVectorIndividual ind = new DoubleVectorIndividual();
            ind.genome = new double[] { i };
            expResult.add(ind);
        }
        final List<DoubleVectorIndividual> result = MultiPopSimpleGroupedEvaluator.getPopChunk(population, from, to);
        assertEquals(expResult, result);
    }

    /** Test of getPopChunk method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetPopChunk4() {
        System.out.println("getPopChunk");
        Population population = getPopulation();
        final int from = 13;
        final int to = 15;
        final List<DoubleVectorIndividual> expResult = new ArrayList<DoubleVectorIndividual>();
        for (int i = 14; i < 16; i++) {
            final DoubleVectorIndividual ind = new DoubleVectorIndividual();
            ind.genome = new double[] { i };
            expResult.add(ind);
        }
        final List<DoubleVectorIndividual> result = MultiPopSimpleGroupedEvaluator.getPopChunk(population, from, to);
        assertEquals(expResult, result);
    }

    
    /** Test of getChunkSubPopulations method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetChunkSubPopulations1() {
        System.out.println("getChunkSubPopulations");
        Population population = getPopulation();
        int from = 0;
        int to = 15;
        final List<Integer> expResult = new ArrayList<Integer>();
        for (int i = 0; i < 5; i++)
            expResult.add(0);
        for (int i = 0; i < 5; i++)
            expResult.add(1);
        for (int i = 0; i < 5; i++)
            expResult.add(2);
        final List<Integer> result = MultiPopSimpleGroupedEvaluator.getChunkSubPopulations(population, from, to);
        assertEquals(expResult, result);
    }
    
    /** Test of getChunkSubPopulations method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetChunkSubPopulations2() {
        System.out.println("getChunkSubPopulations");
        Population population = getPopulation();
        int from = 0;
        int to = 14;
        final List<Integer> expResult = new ArrayList<Integer>();
        for (int i = 0; i < 5; i++)
            expResult.add(0);
        for (int i = 0; i < 5; i++)
            expResult.add(1);
        for (int i = 0; i < 4; i++)
            expResult.add(2);
        final List<Integer> result = MultiPopSimpleGroupedEvaluator.getChunkSubPopulations(population, from, to);
        assertEquals(expResult, result);
    }
    
    /** Test of getChunkSubPopulations method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetChunkSubPopulations3() {
        System.out.println("getChunkSubPopulations");
        Population population = getPopulation();
        int from = 4;
        int to = 11;
        final List<Integer> expResult = new ArrayList<Integer>();
        expResult.add(0);
        for (int i = 0; i < 5; i++)
            expResult.add(1);
        expResult.add(2);
        final List<Integer> result = MultiPopSimpleGroupedEvaluator.getChunkSubPopulations(population, from, to);
        assertEquals(expResult, result);
    }
    
    /** Test of getChunkSubPopulations method, of class MultiPopSimpleGroupedEvaluator. */
    @Test
    public void testGetChunkSubPopulations4() {
        System.out.println("getChunkSubPopulations");
        Population population = getPopulation();
        int from = 13;
        int to = 15;
        final List<Integer> expResult = new ArrayList<Integer>();
        for (int i = 0; i < 2; i++)
            expResult.add(2);
        final List<Integer> result = MultiPopSimpleGroupedEvaluator.getChunkSubPopulations(population, from, to);
        assertEquals(expResult, result);
    }
    
}
