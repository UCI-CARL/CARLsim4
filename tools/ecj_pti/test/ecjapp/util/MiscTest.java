package ecjapp.util;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class MiscTest {
    
    public MiscTest() {
    }

    /** Test of countOccurrences method, of class Misc. */
    @Test
    public void testCountOccurrences() {
        System.out.println("countOccurrences");
        final Integer[] array = new Integer[] { 9, 6, 4, 9, 1, 8, 5, 1, 5, 2, 5, 5, 0, 7, 6, 4, 8, 7, 6, 8 };
        final Map<Integer, Integer> expResult = new HashMap<Integer, Integer>() {{
            put(0, 1);
            put(1, 2);
            put(2, 1);
            // There are no 3's, so 3 does not appear as a key
            put(4, 2);
            put(5, 4);
            put(6, 3);
            put(7, 2);
            put(8, 3);
            put(9, 2);
        }};
        final Map<Integer, Integer> result = Misc.countOccurrences(array);
        assertEquals(expResult, result);
    }
        final Collection[] bad1 = new Collection[] { new HashSet(), new TreeSet(), new LinkedList(), new LinkedHashSet() };

    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test
    public void testContainsOnlySubtypesOf1() {
        System.out.println("containsOnlySubtypesOf (empty)");
        final Collection[] c = new Collection[] { };
        assertTrue(Misc.containsOnlySubtypesOf(c, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test
    public void testContainsOnlySubtypesOf2() {
        System.out.println("containsOnlySubtypesOf (3 good elements)");
        final Collection[] c = new Collection[] { new HashSet(), new TreeSet(), new LinkedHashSet() };
        assertTrue(Misc.containsOnlySubtypesOf(c, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test
    public void testContainsOnlySubtypesOf3() {
        System.out.println("containsOnlySubtypesOf (1 bad element");
        final Collection[] c = new Collection[] { new LinkedList() };
        assertFalse(Misc.containsOnlySubtypesOf(c, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test
    public void testContainsOnlySubtypesOf4() {
        System.out.println("containsOnlySubtypesOf (1 bad element in middle");
        final Collection[] c = new Collection[] { new HashSet(), new TreeSet(), new LinkedList(), new LinkedHashSet() };
        assertFalse(Misc.containsOnlySubtypesOf(c, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test
    public void testContainsOnlySubtypesOf5() {
        System.out.println("containsOnlySubtypesOf (1 bad element at end");
        final Collection[] c = new Collection[] { new HashSet(), new TreeSet(), new LinkedHashSet(), new LinkedList() };
        assertFalse(Misc.containsOnlySubtypesOf(c, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test
    public void testContainsOnlySubtypesOf6() {
        System.out.println("containsOnlySubtypesOf (1 bad element at beginning");
        final Collection[] c = new Collection[] { new LinkedList(), new HashSet(), new TreeSet(), new LinkedHashSet() };
        assertFalse(Misc.containsOnlySubtypesOf(c, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test(expected = NullPointerException.class)
    public void testContainsOnlySubtypesOf7() {
        System.out.println("containsOnlySubtypesOf (null array)");
        assertTrue(Misc.containsOnlySubtypesOf(null, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test(expected = NullPointerException.class)
    public void testContainsOnlySubtypesOf8() {
        System.out.println("containsOnlySubtypesOf (array with lone null element)");
        final Collection[] c = new Collection[] { null };
        assertTrue(Misc.containsOnlySubtypesOf(c, Set.class));
    }
    
    /** Test of containsOnlySubtypesOf method, of class Misc. */
    @Test(expected = NullPointerException.class)
    public void testContainsOnlySubtypesOf9() {
        System.out.println("containsOnlySubtypesOf (array with null element in middle)");
        final Collection[] c = new Collection[] { new HashSet(), new TreeSet(), new LinkedList(), new LinkedHashSet() };
        assertTrue(Misc.containsOnlySubtypesOf(c, Set.class));
    }
}