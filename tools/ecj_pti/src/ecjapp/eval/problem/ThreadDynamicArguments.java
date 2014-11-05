package ecjapp.eval.problem;

import ec.EvolutionState;
import ec.Individual;
import ec.util.Parameter;
import ecjapp.util.Misc;
import ecjapp.util.Option;

/**
 * Passes information about the parent evaluation thread on to a client
 * command.
 * 
 * "option" specifies the name of the CLI parameter to use.
 * 
 * If "modulo" is specified, then the thread number modulo the specified value
 * is passed as an argument.  Otherwise, the thread number itself is passed.
 * 
 * Example: If the option is "-device", no modulo is specified, and the thread
 * number is 3, then the arguments "-device 3" is returned by get().
 * 
 * @author Eric 'Siggy' Scott
 */
public class ThreadDynamicArguments implements DynamicArguments {
    private final static String P_OPT = "option";
    private final static String P_MODULO = "modulo";
    
    private String option;
    private Option<Integer> modulo;
    
    public void setup(final EvolutionState state, final Parameter base) {
        assert(state != null);
        assert(base != null);
        
        option = Misc.getRequiredParameter(state, base.push(P_OPT));
        if (state.parameters.getString(base.push(P_MODULO), null) != null)
            modulo = new Option<Integer>(state.parameters.getInt(base.push(P_MODULO), null));
        else
            modulo = Option.NONE;
        assert(repOK());
    }
       
    
    @Override
    public String get(final EvolutionState state, final Individual[] individuals, final int from, final int to, final int subpopulation, final int threadnum) {
        assert(threadnum >= 0);
        if (modulo.isDefined())
            return option + " " + Integer.toString(threadnum % modulo.get());
        else
            return option + " " + Integer.toString(threadnum);
    }
    
    // <editor-fold defaultstate="collapsed" desc="Standard Methods">
    final public boolean repOK() {
        return P_OPT != null
                && P_MODULO != null
                && !P_OPT.isEmpty()
                && !P_MODULO.isEmpty()
                && option != null
                && modulo != null;
    }
    
    @Override
    public boolean equals(final Object o) {
        if (!(o instanceof ThreadDynamicArguments))
            return false;
        final ThreadDynamicArguments ref = (ThreadDynamicArguments) o;
        return option.equals(ref.option)
                && modulo.equals(ref.modulo);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 67 * hash + (this.option != null ? this.option.hashCode() : 0);
        hash = 67 * hash + (this.modulo != null ? this.modulo.hashCode() : 0);
        return hash;
    }
    
    @Override
    public String toString() {
        return String.format("[%s: option=%s, modulo=%s]", this.getClass().getSimpleName(), option, modulo);
    }
    // </editor-fold>
}
