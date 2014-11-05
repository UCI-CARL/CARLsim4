package ecjapp.util;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class StatisticsTest {
    
    public StatisticsTest() {
    }
    
    /** Test of mean method, of class Misc. */
    @Test
    public void testMean() {
        System.out.println("mean");
        double[] values = new double[] { 5, 1, 9, 16, -3, 8, -15, 22, 7.5, 5, 0.1 };
        double expResult = 5.05454545454;
        double result = Statistics.mean(values);
        assertEquals(expResult, result, 0.00001);
    }

    /** Test of std method, of class Misc. */
    @Test
    public void testStd() {
        System.out.println("std");
        double[] values = new double[] { 5, 1, 9, 16, -3, 8, -15, 22, 7.5, 5, 0.1 };
        double mean = 5.05454545454;
        double expResult = 9.2698302069733565;
        double result = Statistics.std(values, mean);
        assertEquals(expResult, result, 0.00001);
    }

    /** Test of max method, of class Misc. */
    @Test
    public void testMax() {
        System.out.println("max");
        double[] values = new double[] { 5, 1, 9, 16, -3, 8, -15, 22, 7.5, 5, 0.1 };
        double expResult = 22;
        double result = Statistics.max(values);
        assertEquals(expResult, result, 0.0);
    }

    /** Test of min method, of class Misc. */
    @Test
    public void testMin() {
        System.out.println("min");
        double[] values = new double[] { 5, 1, 9, 16, -3, 8, -15, 22, 7.5, 5, 0.1 };
        double expResult = -15;
        double result = Statistics.min(values);
        assertEquals(expResult, result, 0.0);
    }
}