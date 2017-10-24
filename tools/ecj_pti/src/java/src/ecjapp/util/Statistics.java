package ecjapp.util;

import java.util.Collection;
import java.util.Comparator;

/**
 * Elementary statistics methods.
 * 
 * @author Eric 'Siggy' Scott
 */
public final class Statistics
{
    private Statistics() throws AssertionError {
        throw new AssertionError(Statistics.class.getSimpleName() + ": Attempted to instantiate static utility class.");
    }
    
    /** Mean. */
    public static double mean(final double[] values) {
        assert(values != null);
        double sum = 0;
        for(int i = 0; i < values.length; i++)
            sum += values[i];
        return sum/values.length;
    }

    /** Standard deviation (without Bessel's correction). */
    public static double std(final double[] values, double mean) {
        assert(values != null);
        double sum = 0;
        for (int i = 0; i < values.length; i++)
            sum += Math.pow(values[i] - mean, 2);
        return Math.sqrt(sum/(values.length));
    }

    /** Maximum value in an array. */
    public static double max(final double[] values) {
        assert(values != null);
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++)
            if (values[i] > max)
                max = values[i];
        return max;
    }

    /** Minimum value in an array. */
    public static double min(final double[] values) {
        assert(values != null);
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < values.length; i++)
            if (values[i] < min)
                min = values[i];
        return min;
    }
    
    /** Mean of arbitrary objects. */
    public static <T> double mean(final Collection<T> values, final DoubleAttribute attribute) {
        assert(values != null);
        double sum = 0;
        for (final T value : values)
            sum += attribute.get(value);
        return sum/values.size();
    }

    /** Standard deviation for arbitrary objects (without Bessel's correction). */
    public static <T> double std(final Collection<T> values, double mean, final DoubleAttribute attribute) {
        assert(values != null);
        double sum = 0;
        for (final T value : values)
            sum += Math.pow(attribute.get(value) - mean, 2);
        return Math.sqrt(sum/(values.size()));
    }

    /** Maximum value in an array of arbitrary objects. 
     * 
     * This is like java.util.Collections.max(), but for arrays.
     */
    public static <T> T max(final T[] values, final Comparator<T> comp) {
        assert(values != null);
        T max = values[0];
        for (int i = 0; i < values.length; i++)
            if (comp.compare(values[i], max) > 0)
                max = values[i];
        return max;
    }

    /** Minimum value in an array of arbitrary objects. 
     * 
     * This is like java.util.Collections.min(), but for arrays.
     */
    public static <T> T min(final T[] values, final Comparator<T> comp) {
        assert(values != null);
        T min = values[0];
        for (int i = 0; i < values.length; i++)
            if (comp.compare(values[i], min) < 0)
                min = values[i];
        return min;
    }
    
    /** Describes a way to represent objects of type T as doubles. */
    public static interface DoubleAttribute<T> {
        double get(final T object);
    }
}
