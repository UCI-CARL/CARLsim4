package ecjapp.util;

import ec.Individual;
import ec.vector.DoubleVectorIndividual;
import java.io.IOException;
import java.io.Writer;
import java.util.List;

/**
 * Static methods to output populations of individuals in CSV format.
 * 
 * @author Eric 'Siggy' Scott
 */
final public class PopulationToFile {
    
    public final static String DELIMITER = ",";
    
    /** Private constructor throws an error if called (ex. via reflection). */
    private PopulationToFile() throws AssertionError
    {
        throw new AssertionError("PopulationToFile: Cannot create instance of static class.");
    }
    
    /** Take a list of DoubleVectorIndividuals and output them to a tab-delimited file.
     * 
     * @param population A non-empty population of Individuals.  If any element is null an IAE is thrown.
     * @param outWriter A non-null Writer to output the CSV to.  When this method returns it does *not* close the outWriter.
     * @return Nothing.  Side effects: Writes a tab-delimited CSV to outWriter, one row per individual, one column per gene.
     * @throws IOException 
     */
    public static void DoubleVectorPopulationToFile(final List<? extends DoubleVectorIndividual> population, final Writer outWriter) throws IOException{
        if (outWriter == null)
            throw new IllegalArgumentException(PopulationToFile.class.getSimpleName() + ": outWriter is null.");
        if (population == null)
            throw new IllegalArgumentException(PopulationToFile.class.getSimpleName() + ": population is null.");
        if (population.isEmpty())
            throw new IllegalArgumentException(PopulationToFile.class.getSimpleName() + ": population is empty.");
        
        for (final Individual ind : population) {
            final double[] genome = ((DoubleVectorIndividual) ind).genome;
            assert(genome.length > 0);
            outWriter.write(String.valueOf(genome[0]));
            for (int i = 1; i < genome.length; i++)
                outWriter.write(String.format("%s%f", DELIMITER, genome[i]));
            outWriter.write(String.format("%n"));
        }
    }
}
