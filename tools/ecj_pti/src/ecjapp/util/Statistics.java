package ecjapp.util;

/**
 * Elementary statistics methods.
 * 
 * @author Eric 'Siggy' Scott
 */
public final class Statistics
{
    public Statistics() throws AssertionError
    {
        throw new AssertionError("Statistics: Attempted to instantiate static utility class.");
    }
    
    /** Mean. */
    public static double mean(double[] values)
    {
        assert(values != null);
        double sum = 0;
        for(int i = 0; i < values.length; i++)
            sum += values[i];
        return sum/values.length;
    }

    /** Population standard deviation. */
    public static double std(double[] values, double mean)
    {
        assert(values != null);
        double sum = 0;
        for (int i = 0; i < values.length; i++)
            sum += Math.pow(values[i] - mean, 2);
        return Math.sqrt(sum/(values.length));
    }

    /** Maximum value in an array. */
    public static double max(double[] values)
    {
        assert(values != null);
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++)
            if (values[i] > max)
                max = values[i];
        return max;
    }

    /** Minimum value in an array. */
    public static double min(double[] values)
    {
        assert(values != null);
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < values.length; i++)
            if (values[i] < min)
                min = values[i];
        return min;
    }
    
}
