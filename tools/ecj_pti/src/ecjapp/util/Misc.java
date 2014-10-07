package ecjapp.util;

import ec.EvolutionState;
import ec.util.Output.OutputExitException;
import ec.util.Parameter;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class Misc {
    
    /** Private constructor throws an error if called (ex. via reflection). */
    private Misc() throws AssertionError
    {
        throw new AssertionError("Misc: Cannot create instance of static class.");
    }
    
    public static <T> Map<T, Integer> countOccurrences(final T[] array) {
        assert(array != null);
        final Map<T, Integer> counts = new HashMap<T, Integer>();
        for (final T item : array) {
            if (counts.containsKey(item))
                counts.put(item, counts.get(item) + 1);
            else
                counts.put(item, 1);
        }
        return counts;
    }
    
    /**
     * @c A non-null array.
     * @type A non-null class.
     * @return False if c contains an object that null or is not a subtype of type.
     */
    public static <T> boolean containsOnlySubtypesOf(final T[] c, final Class type)
    {
        assert(c != null);
        assert(type != null);
        for (T o : c)
            if (o == null || !type.isInstance(o))
                return false;
        return true;
    }
    
    private static double DEFAULT_PRECISION = 0.000001;
    
    public static boolean doubleEquals(final double a, final double b)
    {
        return doubleEquals(a, b, DEFAULT_PRECISION);
    }
    
    public static boolean doubleEquals(final double a, final double b, final double epsilon)
    {
        return Math.abs(a - b) < epsilon;
    }
    
    public static boolean doubleArrayEquals(final double[] a, final double[] b) {
        return doubleArrayEquals(a, b, DEFAULT_PRECISION);
    }
    
    public static boolean doubleArrayEquals(final double[] a, final double[] b, final double epsilon) {
        if (a.length != b.length)
            return false;
        for (int i = 0; i < a.length; i++)
        {
            if (!doubleEquals(a[i], b[i], epsilon))
                return false;
        }
        return true;
    }
    
    /** Retrieve a parameter from an ECJ instance's ParameterDatabase and throw
     * a fit if it's not found.
     * 
     * @throws ec.util.Output.OutputExitException 
     */
    public static String getRequiredParameter(final EvolutionState state, final Parameter parameter) throws OutputExitException {
        final String param = state.parameters.getString(parameter, null);
        if (param == null || param.isEmpty()) {
            state.output.fatal(String.format("%s: required parameter %s is undefined or empty.", Misc.class.getSimpleName(), parameter.toString()));
        }
        return param;
    }
    
    /** Retrieve a parameter an ECJ instance's ParameterDatabase and create
     * an object from it, or throw a fit if it's not found.
     * 
     * @throws ec.util.Output.OutputExitException 
     */
    public static Object getInstanceOfRequiredParameter(final EvolutionState state, final Parameter parameter, Class mustCastToSuperclass) throws OutputExitException {
        final Object param = state.parameters.getInstanceForParameter(parameter, null, mustCastToSuperclass);
        if (param == null) {
            state.output.fatal(String.format("%s: required parameter %s is undefined or empty.", Misc.class.getSimpleName(), parameter.toString()));
        }
        return param;
    }
}
