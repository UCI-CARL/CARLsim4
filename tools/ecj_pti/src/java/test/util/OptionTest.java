package ecjapp.util;

import java.util.Arrays;
import java.util.Collection;
import org.junit.Before;
import org.junit.Test;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
@RunWith(Parameterized.class)
public class OptionTest {
    private Option sut;
    
    private final Object sutVal;
    private final boolean expectedIsDefined;
    private final String expectedToString;
    private final boolean expectedEqualsRef1;
    private final boolean expectedEqualsRef2;
    private final boolean expectedEqualsRef3;
    
    private final static Object REF1 = new Option<String>("Hello.");
    private final static Object REF2 = new Option<String>("");
    private final static Object REF3 = Option.NONE;
    
    public OptionTest(final Object val, final boolean isDefined, final String toString, final boolean equalsRef1, final boolean equalsRef2, final boolean equalsRef3) {
        this.sutVal = val;
        this.expectedIsDefined = isDefined;
        this.expectedToString = toString;
        this.expectedEqualsRef1 = equalsRef1;
        this.expectedEqualsRef2 = equalsRef2;
        this.expectedEqualsRef3 = equalsRef3;
    }
    
   @Parameters
   public static Collection data() {
      return Arrays.asList(new Object[][] {
         { "Hello.", true, "[Option: val=Hello.]", true, false, false },
         { "Hello world!", true, "[Option: val=Hello world!]", false, false, false },
         { "", true, "[Option: val=]", false, true, false },
         { null, false, "[Option: val=null]", false, false, true },
         { 0, true, "[Option: val=0]", false, false, false }
      });
   }
    
    @Before
    public void setUp() {
        this.sut = new Option(sutVal);
    }

    /** Test of isDefined method, of class Option. */
    @Test
    public void testIsDefined() {
        System.out.println("isDefined");
        assertEquals(expectedIsDefined, sut.isDefined());
    }

    /** Test of toString method, of class Option. */
    @Test
    public void testToString() {
        System.out.println("toString");
        assertEquals(expectedToString, sut.toString());
    }

    /** Test of equals method, of class Option. */
    @Test
    public void testEquals() {
        System.out.println("equals");
        
        // Reflexive
        assertEquals(new Option(sut.get()), sut);
        
        // Symmetric
        assertEquals(expectedEqualsRef1, sut.equals(REF1));
        assertEquals(expectedEqualsRef2, sut.equals(REF2));
        assertEquals(expectedEqualsRef3, sut.equals(REF3));
        assertEquals(expectedEqualsRef1, REF1.equals(sut));
        assertEquals(expectedEqualsRef2, REF2.equals(sut));
        assertEquals(expectedEqualsRef3, REF3.equals(sut));
        
        if (expectedEqualsRef1) assertFalse(sut.equals(REF2));
        if (expectedEqualsRef1) assertFalse(sut.equals(REF3));
        if (expectedEqualsRef2) assertFalse(sut.equals(REF1));
        if (expectedEqualsRef2) assertFalse(sut.equals(REF3));
        if (expectedEqualsRef3) assertFalse(sut.equals(REF1));
        if (expectedEqualsRef3) assertFalse(sut.equals(REF2));
    }

    /** Test of hashCode method, of class Option. */
    @Test
    public void testHashCode() {
        System.out.println("hashCode");
        if (expectedEqualsRef1)
            assertEquals(REF1.hashCode(), sut.hashCode());
        if (expectedEqualsRef2)
            assertEquals(REF2.hashCode(), sut.hashCode());
        if (expectedEqualsRef3)
            assertEquals(REF3.hashCode(), sut.hashCode());
    }
}