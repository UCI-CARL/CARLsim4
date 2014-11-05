package ecjapp.doubles;

import ec.Individual;
import ec.simple.SimpleFitness;
import ec.util.Parameter;

/**
 * A fake individual.
 * 
 * @author Eric 'Siggy' Scott
 */
public class TestIndividual extends Individual {
    private final int trait;
    
    public int getTrait() {
        return trait;
    }
    
    public TestIndividual(final int trait) {
        this.trait = trait;
        this.fitness = new SimpleFitness();
    };

    @Override
    public Parameter defaultBase() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean equals(Object ind) {
        if (!(ind instanceof TestIndividual))
            return false;
        final TestIndividual ref = (TestIndividual)ind;
        return trait == ref.trait;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 13 * hash + this.trait;
        return hash;
    }
    
}
