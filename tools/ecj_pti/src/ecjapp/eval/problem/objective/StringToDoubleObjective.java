package ecjapp.eval.problem.objective;

import ec.EvolutionState;
import ec.simple.SimpleFitness;
import ec.util.Parameter;
import ecjapp.util.Misc;
import ecjapp.util.Option;

/**
 * A "dumb" objective that interprets the phenotype string as a fitness value.
 * Use this objective if the external simulation does the whole
 * genotype->fitness mapping, instead of just a genotype->phenotype mapping.
 * 
 * @author Eric 'Siggy' Scott
 */
public class StringToDoubleObjective implements ObjectiveFunction<SimpleFitness> {
    public final static String P_IDEAL_FITNESS_VALUE = "idealFitnessValue";
    
    private Option<Double> idealFitnessValue = Option.NONE;

    @Override
    public void setup(EvolutionState state, Parameter base) {
        if (state == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": state is null.");
        if (state.parameters == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": state.parameters is null.");
        if (base == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": base is null.");
        idealFitnessValue = new Option<Double>(state.parameters.getDouble(base.push(P_IDEAL_FITNESS_VALUE), null));
    }
    
    @Override
    public SimpleFitness evaluate(final EvolutionState state, final String phenotype) {
        final double realFitness = Double.valueOf(phenotype);
        final boolean isIdeal = idealFitnessValue.isDefined() ? Misc.doubleEquals(realFitness, idealFitnessValue.get()) || realFitness > idealFitnessValue.get() : false;
        
        final SimpleFitness fitness = new SimpleFitness();
        fitness.setFitness(state, realFitness, isIdeal);
        return fitness;
    }
    
}
