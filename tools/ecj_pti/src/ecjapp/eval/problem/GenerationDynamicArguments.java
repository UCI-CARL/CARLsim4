package ecjapp.eval.problem;

import ec.EvolutionState;
import ec.Individual;
import ec.util.Parameter;
import ecjapp.util.Misc;
import ecjapp.util.Option;

/**
 * Passes information about the current generation on to the client command.
 * 
 * "option" specifies the name of the CLI parameter to use.
 * 
 * Example: If the option is "-gen" and the EvolutionState passed to get() is on
 * generation 20, then the arguments "-gen 3" is returned by get().
 * 
 * If another DynamicArguments is specified for the "dynamicArguments" option,
 * then the result of its get() method will be prepended.
 * 
 * Example: If the object in the "dynamicArguments" field returns "-device 3",
 * then the result of the above example will be "-device 3 -gen 3".
 * 
 * @author Eric 'Siggy' Scott
 */
public class GenerationDynamicArguments implements DynamicArguments {
    public final static String P_OPT = "option";
    public final static String P_DYNAMIC_ARGUMENTS = "dynamicArguments";
    
    private String option;
    private Option<DynamicArguments> dynamicArguments;
    
    public void setup(final EvolutionState state, final Parameter base) {
        assert(state != null);
        assert(base != null);
        
        option = Misc.getRequiredParameter(state, base.push(P_OPT));
        if (state.parameters.exists(base.push(P_DYNAMIC_ARGUMENTS), null)) {
            dynamicArguments = new Option<DynamicArguments>((DynamicArguments) state.parameters.getInstanceForParameter(base.push(P_DYNAMIC_ARGUMENTS), null, DynamicArguments.class));
            dynamicArguments.get().setup(state, base.push(P_DYNAMIC_ARGUMENTS));
        }
        else
            dynamicArguments = Option.NONE;
        assert(repOK());
    }
    
    @Override
    public String get(final EvolutionState state, final Individual[] individuals, final int from, final int to, final int subpopulation, final int threadnum) {
        assert(threadnum >= 0);
        final String childArguments = dynamicArguments.isDefined() ? dynamicArguments.get().get(state, individuals, from, to, subpopulation, threadnum) : "";
        return String.format("%s %s %s", childArguments, option, Integer.toString(state.generation));
    }
    
    // <editor-fold defaultstate="collapsed" desc="Standard Methods">
    final public boolean repOK() {
        return P_OPT != null
                && P_DYNAMIC_ARGUMENTS != null
                && !P_OPT.isEmpty()
                && !P_DYNAMIC_ARGUMENTS.isEmpty()
                && option != null
                && dynamicArguments != null;
    }
    
    @Override
    public boolean equals(final Object o) {
        if (!(o instanceof GenerationDynamicArguments))
            return false;
        final GenerationDynamicArguments ref = (GenerationDynamicArguments) o;
        return option.equals(ref.option)
                && dynamicArguments.equals(ref.dynamicArguments);
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 83 * hash + (this.option != null ? this.option.hashCode() : 0);
        hash = 83 * hash + (this.dynamicArguments != null ? this.dynamicArguments.hashCode() : 0);
        return hash;
    }
    
    @Override
    public String toString() {
        return String.format("[%s: option=%s, dynamicArguments=%s]", this.getClass().getSimpleName(), option, dynamicArguments);
    }
    // </editor-fold>
}
