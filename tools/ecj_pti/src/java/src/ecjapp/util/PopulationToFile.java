package ecjapp.util;

import ec.Individual;
import ec.vector.DoubleVectorIndividual;
import java.io.IOException;
import java.io.StringWriter;
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
     * @param individuals A non-empty population of Individuals.  If any element is null an IAE is thrown.
     * @param outWriter A non-null Writer to output the CSV to.  When this method returns it does *not* close the outWriter.
     * @return Nothing.  Side effects: Writes a tab-delimited CSV to outWriter, one row per individual, one column per gene.
     * @throws IOException 
     */
    public static void DoubleVectorIndividualsToFile(final List<DoubleVectorIndividual> individuals, final Option<List<Integer>> subPopulations, final Writer outWriter) throws IOException{
        assert(outWriter != null);
        assert(individuals != null);
        assert(!individuals.isEmpty());
        assert(!(subPopulations.isDefined() && subPopulations.get().size() != individuals.size()));
        
        for (int i = 0; i < individuals.size(); i++) {
            if (subPopulations.isDefined())
                outWriter.write(String.format("%d%s", subPopulations.get().get(i), DELIMITER));
            final double[] genome = individuals.get(i).genome;
            assert(genome.length > 0);
            outWriter.write(String.valueOf(genome[0]));
            for (int j = 1; j < genome.length; j++)
                outWriter.write(String.format("%s%f", DELIMITER, genome[j]));
            outWriter.write(String.format("%n"));
        }
    }
    
    public static String DoubleVectorIndividualsToString(final List<DoubleVectorIndividual> individuals, final Option<List<Integer>> subPopulations) {
        final Writer stringWriter = new StringWriter();
        try {
            DoubleVectorIndividualsToFile(individuals, subPopulations, stringWriter);
        }
        catch (final IOException e) {
            throw new IllegalStateException(e);
        }
        return stringWriter.toString();
    }
}
