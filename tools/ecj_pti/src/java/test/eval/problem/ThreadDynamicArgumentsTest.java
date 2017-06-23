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
public class ThreadDynamicArgumentsTest {
    private final static Parameter BASE = new Parameter("base");
    private final static String OPT1 = "-opt1";
    private final static String OPT2 = "-opt2";
    
    private EvolutionState state;
    private ThreadDynamicArguments sut;
    
    public ThreadDynamicArgumentsTest() { }
    
    @Before
    public void setUp() {
        this.state = getFreshState();
        this.sut = new ThreadDynamicArguments();
    }
    
    private static EvolutionState getFreshState() {
        final EvolutionState state = new SimpleEvolutionState();
        state.parameters = getParams();
        state.output = Evolve.buildOutput();
        state.output.setThrowsErrors(true);
        return state;
    }
    
    private static ParameterDatabase getParams() {
        final ParameterDatabase parameters = new ParameterDatabase();
        parameters.set(BASE.push(ThreadDynamicArguments.P_OPT), OPT1);
        return parameters;
    }

    @Test
    public void testGet1() {
        System.out.println("get");
        sut.setup(state, BASE);
        final int thread = 0;
        final String expResult = String.format(" %s %s", OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGet2() {
        System.out.println("get");
        sut.setup(state, BASE);
        final int thread = 100;
        final String expResult = String.format(" %s %s", OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGet3() {
        System.out.println("get");
        sut.setup(state, BASE);
        final int thread = Integer.MAX_VALUE;
        final String expResult = String.format(" %s %s", OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGetModulo1() {
        System.out.println("get (with modulo)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 0;
        final String expResult = String.format(" %s %s", OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGetModulo2() {
        System.out.println("get (with modulo)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 4;
        final String expResult = String.format(" %s %s", OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGetModulo3() {
        System.out.println("get (with modulo)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 5;
        final String expResult = String.format(" %s %s", OPT1, 0);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGetModulo4() {
        System.out.println("get (with modulo)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 15;
        final String expResult = String.format(" %s %s", OPT1, 0);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGetModulo5() {
        System.out.println("get (with modulo)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 100;
        final String expResult = String.format(" %s %s", OPT1, 0);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }

    @Test
    public void testGetModulo6() {
        System.out.println("get (with modulo)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 156;
        final String expResult = String.format(" %s %s", OPT1, 1);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped1() {
        System.out.println("get (with wrapped dynamicArguments)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 156;
        final String expResult = String.format(" %s %s %s %s", OPT2, 1, OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped2() {
        System.out.println("get (with wrapped dynamicArguments)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 5;
        final String expResult = String.format(" %s %s %s %s", OPT2, 0, OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped3() {
        System.out.println("get (with wrapped dynamicArguments)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 17;
        final String expResult = String.format(" %s %s %s %s", OPT2, 2, OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
    
    @Test
    public void testGetWrapped4() {
        System.out.println("get (with wrapped dynamicArguments)");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        state.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_MODULO), "5");
        sut.setup(state, BASE);
        final int thread = 2;
        final String expResult = String.format(" %s %s %s %s", OPT2, 2, OPT1, thread);
        String result = sut.get(state, thread);
        assertEquals(expResult, result);
    }
   

    @Test
    public void testEquals() {
        System.out.println("equals");
        
        final EvolutionState state1 = getFreshState();
        final EvolutionState state2 = getFreshState();
        state2.parameters.set(BASE.push(ThreadDynamicArguments.P_MODULO), "5");
        final EvolutionState state3 = getFreshState();
        state3.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS), "ecjapp.eval.problem.ThreadDynamicArguments");
        state3.parameters.set(BASE.push(ThreadDynamicArguments.P_DYNAMIC_ARGUMENTS).push(ThreadDynamicArguments.P_OPT), OPT2);
        
        sut.setup(state, BASE);
        final ThreadDynamicArguments sut1 = new ThreadDynamicArguments();
        sut1.setup(state1, BASE);
        final ThreadDynamicArguments sut2 = new ThreadDynamicArguments();
        sut2.setup(state2, BASE);
        final ThreadDynamicArguments sut3 = new ThreadDynamicArguments();
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