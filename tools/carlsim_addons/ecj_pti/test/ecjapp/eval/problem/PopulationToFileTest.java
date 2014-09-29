package ecjapp.eval.problem;

import ecjapp.util.PopulationToFile;
import ec.vector.DoubleVectorIndividual;
import ecjapp.util.Misc;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class PopulationToFileTest {
    
    public PopulationToFileTest() {
    }
    
    private static List<double[]> getInputData() {
        return new ArrayList<double[]>() {{
            add(new double[] {8.912, 8.591, 4.921, 4.151, 4.753});
            add(new double[] {3.372, 5.962, 2.573, 4.285, 2.218});
            add(new double[] {6.962, 6.534, 3.785, 7.952, 4.399});
            add(new double[] {1.586, 4.200, 8.683, 9.881, 3.581});
            add(new double[] {4.009, 9.646, 8.137, 5.873, 2.643});
        }};
    }
    
    private List<DoubleVectorIndividual> getDoubleVectorPopulation() {
        final List<DoubleVectorIndividual> population = new ArrayList<DoubleVectorIndividual>(){{
            for (int i = 0; i < 5; i++)
                add(new DoubleVectorIndividual());
        }};
        final List<double[]> inputData = getInputData();
        population.get(0).genome = inputData.get(0);
        population.get(1).genome = inputData.get(1);
        population.get(2).genome = inputData.get(2);
        population.get(3).genome = inputData.get(3);
        population.get(4).genome = inputData.get(4);
        return population;
    }

    /** Test of DoubleVectorPopulationToFile method, of class PopulationToFile. */
    @Test
    public void testDoubleVectorPopulationToFile() throws Exception {
        System.out.println("DoubleVectorPopulationToFile");
        final List<DoubleVectorIndividual> population = getDoubleVectorPopulation();
        final StringWriter outWriter = new StringWriter();
        
        PopulationToFile.DoubleVectorPopulationToFile(population, outWriter);
        
        final List<double[]> expected = getInputData();
        final List<double[]> parsedResult = parseCSV(outWriter.toString(), ",");
        
        assertEquals(expected.size(), parsedResult.size());
        for(int i = 0; i < parsedResult.size(); i++)
            assertTrue(Misc.doubleArrayEquals(expected.get(i), parsedResult.get(i), 0.0000000001));
    }

    /** Test of DoubleVectorPopulationToFile method, of class PopulationToFile. */
    @Test(expected = IllegalArgumentException.class)
    public void testDoubleVectorPopulationToFile2() throws Exception {
        System.out.println("DoubleVectorPopulationToFile (null writer IAE)");
        final List<DoubleVectorIndividual> population = getDoubleVectorPopulation();
        final StringWriter outWriter = null;
        
        PopulationToFile.DoubleVectorPopulationToFile(population, outWriter);
    }

    /** Test of DoubleVectorPopulationToFile method, of class PopulationToFile. */
    @Test(expected = IllegalArgumentException.class)
    public void testDoubleVectorPopulationToFile3() throws Exception {
        System.out.println("DoubleVectorPopulationToFile (null population IAE)");
        final List<DoubleVectorIndividual> population = null;
        final StringWriter outWriter = new StringWriter();
        
        PopulationToFile.DoubleVectorPopulationToFile(population, outWriter);
    }
    
    private List<double[]> parseCSV(final String csv, final String delimiter) {
        final String[] lines = csv.split(System.getProperty("line.separator"));
        final List<double[]> outputs = new ArrayList<double[]>() {{
            for (int i = 0; i < lines.length; i++) {
                final String line = lines[i];
                final String[] columnStrings = line.split(delimiter);
                final double[] ind = new double[columnStrings.length];
                for (int j = 0; j < columnStrings.length; j++) {
                    ind[j] = Double.valueOf(columnStrings[j]);
                }
                add(ind);
            }
        }};
        return outputs;
    }
}