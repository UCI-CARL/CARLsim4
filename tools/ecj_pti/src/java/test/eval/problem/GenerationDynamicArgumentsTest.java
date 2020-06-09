package ecjapp.eval.problem;

import ec.EvolutionState;
import ec.Evolve;
import ec.simple.SimpleEvolutionState;
import ec.util.Parameter;
import ec.util.ParameterDatabase;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class GenerationDynamicArgumentsTest {
    private final static Parameter BASE = new Parameter("base");
    private final static String OPT1 = "-opt1";
    private final static String OPT2 = "-opt2";
    
    private EvolutionState state;
    private GenerationDynamicArguments sut;
    
    public GenerationDynamicArgumentsTest() { }
    
    @Before
    public void setUp() {
        this.state = getFreshState();
        this.sut = new GenerationDynamicArguments();
    }
    
    private static EvolutionState getFreshState() {
        final EvolutionState state = new SimpleEvolutionState();
        state.parameters = getParams();
        state.output = Evolve.buildOutput();
        state.output.setThrowsErrors(true);
        state.generation = 0;
        return state;
    }
    
    private static ParameterDatabase getParams() {
        final ParameterDatabase parameters = new ParameterDatabase();
        parameters.set(BASE.push(GenerationDynamicArguments.P_OPT), OPT1);
        return parameters;
    }
    
    @Test
    public void testGet1() {
        System.out.println("get");
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s", OPT1, state.generation);
        final String result = sut.get(state, 0);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGet2() {
        System.out.println("get");
        state.generation = 15;
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s", OPT1, state.generation);
        final String result = sut.get(state, 0);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGet3() {
        System.out.println("get");
        state.generation = Integer.MAX_VALUE;
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s", OPT1, state.generation);
        final String result = sut.get(state, 0);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped1() {
        System.out.println("get (with wrapped dynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        state.generation = 0;
        final int thread = state.generation;
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s %s %s", OPT2, 0, OPT1, state.generation);
        final String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped2() {
        System.out.println("get (with wrapped dynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        state.generation = 15;
        final int thread = state.generation;
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s %s %s", OPT2, 0, OPT1, state.generation);
        final String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped3() {
        System.out.println("get (with wrapped dynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        state.generation = 156;
        final int thread = state.generation;
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s %s %s", OPT2, 1, OPT1, state.generation);
        final String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped4() {
        System.out.println("get (with wrapped dynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        state.generation = 4;
        final int thread = state.generation;
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s %s %s", OPT2, 4, OPT1, state.generation);
        final String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped5() {
        System.out.println("get (with wrapped dynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        state.generation = 14;
        final int thread = state.generation;
        sut.setup(state, BASE);
        final String expResult = String.format(" %s %s %s %s", OPT2, 4, OPT1, state.generation);
        final String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testEquals() {
       final EvolutionState state1 = getFreshState();
        final EvolutionState state2 = getFreshState();
        state2.parameters.set(BASE.push(GenerationDynamicArguments.P_OPT), OPT2);
        final EvolutionState state3 = getFreshState();
        state3.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state3.parameters.set(BASE.push(GenerationDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        
        sut.setup(state, BASE);
        final GenerationDynamicArguments sut1 = new GenerationDynamicArguments();
        sut1.setup(state1, BASE);
        final GenerationDynamicArguments sut2 = new GenerationDynamicArguments();
        sut2.setup(state2, BASE);
        final GenerationDynamicArguments sut3 = new GenerationDynamicArguments();
        sut3.setup(state3, BASE);
        
        assertTrue(sut.equals(sut));
        assertTrue(sut.equals(sut1));
        assertFalse(sut.equals(sut2));
        assertFalse(sut.equals(sut3));
        assertFalse(sut2.equals(sut));
        assertFalse(sut3.equals(sut));
        
        assertEquals(sut.hashCode(), sut1.hashCode());
        assertFalse(sut.hashCode() == sut2.hashCode());
        assertFalse(sut.hashCode() == sut3.hashCode());
    }
}