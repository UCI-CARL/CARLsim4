package ecjapp.statistics;

import ec.EvolutionState;
import ec.Evolve;
import ec.Individual;
import ec.Population;
import ec.Subpopulation;
import ec.simple.SimpleEvolutionState;
import ec.simple.SimpleFitness;
import ec.util.Parameter;
import ec.util.ParameterDatabase;
import ec.vector.BitVectorIndividual;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class FitnessStatisticsTest {
    private final static Parameter BASE = new Parameter("base");
    private final static String OUTPUT_PATH = "fitnessStatisticsTestOutput.csv";
    private EvolutionState state;
    private FitnessStatistics sut;
    
    public FitnessStatisticsTest() { }
    
    @Before
    public void setUp() {
        state = getFreshState();
        sut = new FitnessStatistics();
    }
    
    @After
    public void tearDown() {
        final File output = new File(OUTPUT_PATH);
        output.deleteOnExit();
    }
    
    private static EvolutionState getFreshState() {
        final EvolutionState state = new SimpleEvolutionState();
        state.parameters = getParams();
        state.output = Evolve.buildOutput();
        state.output.setThrowsErrors(true);
        state.job = new Integer[] { 0 };
        state.population = new Population();
        state.population.subpops = new Subpopulation[] { new Subpopulation() };
        state.population.subpops[0].individuals = new Individual[] {
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 3.3, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 1.4, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 6.6, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 2.9, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 7.8, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 6.0, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 4.2, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 5.5, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 0.9, false); }}; }},
            new BitVectorIndividual() {{ fitness = new SimpleFitness() {{ setFitness(state, 2.8, false); }}; }}
        };
        return state;
    }
    
    private static ParameterDatabase getParams() {
        final ParameterDatabase params = new ParameterDatabase();
        params.set(BASE.push(FitnessStatistics.P_INDIVIDUALS), "false");
        params.set(BASE.push(FitnessStatistics.P_STATISTICS_FILE), OUTPUT_PATH);
        return params;
    }
    
    @Test
    public void testPostInitializationStatistics() {
        System.out.println("postInitializationStatistics");
        sut.setup(state, BASE);
        
        sut.postInitializationStatistics(state);
        
        final String expectedResult = "job, generation, time, mean, std, min, max, bsf\n";
        try {
            final String result = new Scanner( new File(OUTPUT_PATH) ).useDelimiter("\\A").next();
            assertEquals(expectedResult, result);
        } catch (final FileNotFoundException ex) {
            fail("Could not find/open output file " + OUTPUT_PATH);
        }
    }

    @Test
    public void testPostEvaluationStatistics1() {
        System.out.println("postEvaluationStatistics");
        sut.setup(state, BASE);
        sut.postEvaluationStatistics(state);
        
        final Integer expectedJob = 0;
        final Integer expectedGeneration = 0;
        final double expectedMean = 4.14;
        final double expectedStd = 2.163423;
        final double expectedMin = 0.9;
        final double expectedMax = 7.8;
        try {
            final String result = new Scanner( new File(OUTPUT_PATH) ).useDelimiter("\\A").next();
            final String[] results = result.split(",");
            assertEquals(results.length, 8);
            assertEquals(expectedJob, Integer.valueOf(results[0].trim()));
            assertEquals(expectedGeneration, Integer.valueOf(results[1].trim()));
            assertEquals(expectedMean, Double.valueOf(results[3]), 0.00001);
            assertEquals(expectedStd, Double.valueOf(results[4]), 0.00001);
            assertEquals(expectedMin, Double.valueOf(results[5]), 0.00001);
            assertEquals(expectedMax, Double.valueOf(results[6]), 0.00001);
        } catch (final FileNotFoundException ex) {
            fail("Could not find/open output file " + OUTPUT_PATH);
        }
    }

    @Test
    public void testPostEvaluationStatistics2() {
        System.out.println("postEvaluationStatistics");
        state.job[0] = 15;
        state.generation = 156;
        sut.setup(state, BASE);
        sut.postEvaluationStatistics(state);
        
        final Integer expectedJob = 15;
        final Integer expectedGeneration = 156;
        final double expectedMean = 4.14;
        final double expectedStd = 2.163423;
        final double expectedMin = 0.9;
        final double expectedMax = 7.8;
        try {
            final String result = new Scanner( new File(OUTPUT_PATH) ).useDelimiter("\\A").next();
            final String[] results = result.split(",");
            assertEquals(results.length, 8);
            assertEquals(expectedJob, Integer.valueOf(results[0].trim()));
            assertEquals(expectedGeneration, Integer.valueOf(results[1].trim()));
            assertEquals(expectedMean, Double.valueOf(results[3]), 0.00001);
            assertEquals(expectedStd, Double.valueOf(results[4]), 0.00001);
            assertEquals(expectedMin, Double.valueOf(results[5]), 0.00001);
            assertEquals(expectedMax, Double.valueOf(results[6]), 0.00001);
        } catch (final FileNotFoundException ex) {
            fail("Could not find/open output file " + OUTPUT_PATH);
        }
    }

    @Test
    public void testPostEvaluationStatisticsIndividuals1() {
        System.out.println("postEvaluationStatistics (individuals)");
        state.parameters.set(BASE.push(FitnessStatistics.P_INDIVIDUALS), "true");
        sut.setup(state, BASE);
        
        sut.postEvaluationStatistics(state);
        
        final Integer expectedJob = 0;
        final Integer expectedGeneration = 0;
        try {
            final String result = new Scanner( new File(OUTPUT_PATH) ).useDelimiter("\\A").next();
            final String[] results = result.split("\n");
            assertEquals(10, results.length);
            for (int i = 0; i < results.length; i++) {
                final String[] line = results[i].split((","));
                assertEquals(line.length, 4);
                assertEquals(expectedJob, Integer.valueOf(line[0].trim()));
                assertEquals(expectedGeneration, Integer.valueOf(line[1].trim()));
                assertEquals(state.population.subpops[0].individuals[i].fitness.fitness(), Double.valueOf(line[3]), 0.00001);
            }
        } catch (final FileNotFoundException ex) {
            fail("Could not find/open output file " + OUTPUT_PATH);
        }
    }

    @Test
    public void testPostEvaluationStatisticsIndividuals2() {
        System.out.println("postEvaluationStatistics (individuals)");
        state.job[0] = 15;
        state.generation = 156;
        state.parameters.set(BASE.push(FitnessStatistics.P_INDIVIDUALS), "true");
        sut.setup(state, BASE);
        
        sut.postEvaluationStatistics(state);
        state.generation = 157;
        sut.postEvaluationStatistics(state);
        
        final Integer expectedJob = 15;
        final Integer[] expectedGeneration = new Integer[] { 156, 157 };
        try {
            final String result = new Scanner( new File(OUTPUT_PATH) ).useDelimiter("\\A").next();
            final String[] results = result.split("\n");
            assertEquals(20, results.length);
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 10; i++) {
                    final String[] line = results[i + j*10].split((","));
                    assertEquals(line.length, 4);
                    assertEquals(expectedJob, Integer.valueOf(line[0].trim()));
                    assertEquals(expectedGeneration[j], Integer.valueOf(line[1].trim()));
                    assertEquals(state.population.subpops[0].individuals[i].fitness.fitness(), Double.valueOf(line[3]), 0.00001);
                }
            }
        } catch (final FileNotFoundException ex) {
            fail("Could not find/open output file " + OUTPUT_PATH);
        }
    }
}