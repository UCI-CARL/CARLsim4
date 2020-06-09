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
 * If another DynamicArguments is specified for the "dynamicArguments" option,
 * then the result of its get() method will be prepended.
 * 
 * Example: If the object in the "dynamicArguments" field returns "-gen 20",
 * then the result of the above example will be "-gen 20 -device 3".
 * 
 * @author Eric 'Siggy' Scott
 */
public class ThreadDynamicArguments implements DynamicArguments {
    public final static String P_OPT = "option";
    public final static String P_MODULO = "modulo";
    public final static String P_DYNAMIC_ARGUMENTS = "dynamicArguments";
    
    private String option;
    private Option<Integer> modulo;
    private Option<DynamicArguments> dynamicArguments;
    
    @Override
    public void setup(final EvolutionState state, final Parameter base) {
        assert(state != null);
        assert(base != null);
        
        assert(state.parameters.exists(base.push(P_OPT), null));
        option = Misc.getRequiredParameter(state, base.push(P_OPT));
        if (state.parameters.exists(base.push(P_MODULO), null))
            modulo = new Option<Integer>(state.parameters.getInt(base.push(P_MODULO), null));
        else
            modulo = Option.NONE;
        if (state.parameters.exists(base.push(P_DYNAMIC_ARGUMENTS), null)) {
            dynamicArguments = new Option<DynamicArguments>((DynamicArguments) state.parameters.getInstanceForParameter(base.push(P_DYNAMIC_ARGUMENTS), null, DynamicArguments.class));
            dynamicArguments.get().setup(state, base.push(P_DYNAMIC_ARGUMENTS));
        }
        else
            dynamicArguments = Option.NONE;
        assert(repOK());
    }
    
    @Override
    public String get(final EvolutionState state, final int threadnum) {
        assert(threadnum >= 0);
        final String childArguments = dynamicArguments.isDefined() ? dynamicArguments.get().get(state, threadnum) : "";
        if (modulo.isDefined())
            return String.format("%s %s %s", childArguments, option, Integer.toString(threadnum % modulo.get()));
        else
            return String.format("%s %s %s", childArguments, option, Integer.toString(threadnum));
    }
    
    // <editor-fold defaultstate="collapsed" desc="Standard Methods">
    final public boolean repOK() {
        return P_OPT != null
                && P_MODULO != null
                && P_DYNAMIC_ARGUMENTS != null
                && option != null
                && modulo != null
                && dynamicArguments != null
                && !P_OPT.isEmpty()
                && !P_MODULO.isEmpty()
                && !P_DYNAMIC_ARGUMENTS.isEmpty();
    }
    
    @Override
    public boolean equals(final Object o) {
        if (!(o instanceof ThreadDynamicArguments))
            return false;
        final ThreadDynamicArguments ref = (ThreadDynamicArguments) o;
        return option.equals(ref.option)
                && modulo.equals(ref.modulo)
                && dynamicArguments.equals(ref.dynamicArguments);
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 37 * hash + (this.option != null ? this.option.hashCode() : 0);
        hash = 37 * hash + (this.modulo != null ? this.modulo.hashCode() : 0);
        hash = 37 * hash + (this.dynamicArguments != null ? this.dynamicArguments.hashCode() : 0);
        return hash;
    }
    
    @Override
    public String toString() {
        return String.format("[%s: option=%s, modulo=%d, dynamicArguments=%d]", this.getClass().getSimpleName(), option, modulo);
    }
    // </editor-fold>
}
