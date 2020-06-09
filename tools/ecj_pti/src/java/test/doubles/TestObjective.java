package ecjapp.doubles;

import ec.EvolutionState;
import ec.simple.SimpleFitness;
import ec.util.Parameter;
import ecjapp.eval.problem.objective.ObjectiveFunction;
import ecjapp.util.PopulationToFile;
import ecjapp.util.Statistics;

/**
 * A simple objective function that returns the maximum element of a
 * real-valued vector.
 * 
 * @author Eric 'Siggy' Scott
 */
public class TestObjective implements ObjectiveFunction<SimpleFitness> {


    @Override
    public void setup(final EvolutionState state, final Parameter base) { }
    
    @Override
    public SimpleFitness evaluate(final EvolutionState state, final String phenotype) {
        assert(phenotype != null);
        assert(!phenotype.isEmpty());
        assert(phenotype.indexOf("\n") == -1);
        
        final SimpleFitness fitness = new SimpleFitness();
        fitness.setFitness(state, (float) Statistics.max(csvRowToDoubleArray(phenotype)), false);
        return fitness;
    }
    
    private double[] csvRowToDoubleArray(final String csvRow) {
        assert(csvRow != null);
        final String[] traitStrings = csvRow.split(PopulationToFile.DELIMITER);
        final double[] traits = new double[traitStrings.length];
        for (int i = 0; i < traits.length; i++)
            traits[i] = Double.valueOf(traitStrings[i]);
        return traits;
    }
    
    public final boolean repOK() {{ return true; }}
    
    @Override
    public boolean equals(final Object o) {
        return (o instanceof TestObjective);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        return hash;
    }
    
    @Override
    public String toString() {
        return "[TestObjective]";
    }
}
