package ecjapp.doubles;

import ec.EvolutionState;
import ec.Individual;
import ec.Problem;
import ec.simple.SimpleProblemForm;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class TestSimpleProblem extends Problem implements SimpleProblemForm {

    @Override
    public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
 
}
