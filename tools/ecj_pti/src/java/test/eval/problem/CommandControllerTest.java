package ecjapp.eval.problem;

import ec.vector.DoubleVectorIndividual;
import ecjapp.util.Option;
import ecjapp.util.PopulationToFile;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.*;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class CommandControllerTest {
    private static final String path = "/bin/cat";
    private CommandController sut;
    private List<DoubleVectorIndividual> testPopulation;
    
    public CommandControllerTest() {
    }
    
    @Before
    public void setUp() {
        sut = new CommandController(path, new Option<String>(""), Option.NONE);
        
        testPopulation = new ArrayList<DoubleVectorIndividual>() {{
            add(new DoubleVectorIndividual() {{ genome = new double[] {9.18,   4.85,    .19,   8.90,   5.04,   7.39,   1.85,   2.27,   7.79,   2.39 }; }});
            add(new DoubleVectorIndividual() {{ genome = new double[] {5.49,   3.60,   3.61,   3.70,   7.78,   3.74,   2.26,   3.90,   6.01,   8.87 }; }});
            add(new DoubleVectorIndividual() {{ genome = new double[] {2.81,   3.56,   9.69,   5.42,   7.31,    .36,   7.87,   2.79,   4.57,   8.13 }; }});
            add(new DoubleVectorIndividual() {{ genome = new double[] {8.97,   2.02,   7.43,   1.14,   9.79,   6.55,   9.41,   6.73,   4.98,   2.89 }; }});
            add(new DoubleVectorIndividual() {{ genome = new double[] {5.08,   6.58,   8.29,   6.83,   5.63,   8.69,   7.48,    .34,   1.39,    .88  }; }});
        }};
    }

    /** Test of constructor, of class CARLsimController. */
    @Test (expected = IllegalArgumentException.class)
    public void testConstructor_IAE1() throws Exception {
        System.out.println("constructor (IAE 1)");
        sut = new CommandController(null, Option.NONE, Option.NONE);
    } 

    /** Test of constructor, of class CARLsimController. */
    @Test (expected = IllegalArgumentException.class)
    public void testConstructor_IAE2() throws Exception {
        System.out.println("constructor (IAE 2)");
        sut = new CommandController("", Option.NONE, Option.NONE);
    } 

    /** Test of constructor, of class CARLsimController. */
    @Test (expected = IllegalArgumentException.class)
    public void testConstructor_IAE3() throws Exception {
        System.out.println("constructor (IAE 3)");
        sut = new CommandController("/bin/cat", Option.NONE, null);
    } 

    /** Test of constructor, of class CARLsimController. */
    @Test (expected = IllegalArgumentException.class)
    public void testConstructor_IAE4() throws Exception {
        System.out.println("constructor (IAE 4)");
        sut = new CommandController("/bin/cat", null, null);
    } 

    /** Test of execute method, of class CARLsimController. */
    @Test (expected = IOException.class)
    public void testExecute_IOE() throws Exception {
        System.out.println("execute (bad command)");
        sut = new CommandController("guhrgwergwerg/wergwer234g", Option.NONE, Option.NONE);
        assertTrue(sut.repOK());
        final String result = sut.execute(testPopulation, Option.NONE, "");
    }    

    /** Test of execute method, of class CARLsimController. */
    @Test
    public void testExecute() throws Exception {
        System.out.println("execute");
        
        final String result = sut.execute(testPopulation, Option.NONE, "");
        
        final String[] lines = result.split("\n");
        testCSVEqualsPopulation(lines, testPopulation);
        assertTrue(sut.repOK());
    }

    /** Test of execute method, of class CARLsimController. */
    @Test
    public void testExecuteWithArgs() throws Exception {
        System.out.println("execute (with arguments)");
        // The -e option tells cat to print '$' at the end of each line.
        sut = new CommandController(path, new Option<String>("-e"), Option.NONE);
        
        final String result = sut.execute(testPopulation, Option.NONE, "");
        
        final String[] lines = result.split("\n");
        for (int i = 0; i < lines.length; i++) {
            assertEquals("$", lines[i].substring(lines[i].length() - 1));
            lines[i] = lines[i].substring(0, lines[i].length() - 1);
        }
        testCSVEqualsPopulation(lines, testPopulation);
        assertTrue(sut.repOK());
    }
    
    private static void testCSVEqualsPopulation(final String[] lines, List<DoubleVectorIndividual> population) {
        assertEquals(population.size(), lines.length);
        // For the ith line in the CSV
        for (int i = 0; i < lines.length; i++) {
            // Make sure it represents the ith individual in the population
            final String[] values = lines[i].split(PopulationToFile.DELIMITER);
            final double[] individual = population.get(i).genome;
            assertEquals(individual.length, values.length);
            for (int j = 0; j < individual.length; j++)
                assertEquals(individual[j], Double.valueOf(values[j]), 0.000001);
        }
    }

    /** Test of equals and hashCode methods, of class CARLsimController. */
    @Test
    public void testEqualsAndHashCode() throws Exception {
        System.out.println("equals and hashCode");
        final CommandController equalController = new CommandController(path, new Option<String>(""), Option.NONE);
        final CommandController notEqualController1 = new CommandController(path, new Option<String>("-e"), Option.NONE);
        final CommandController notEqualController2 = new CommandController("/bin/sh", new Option<String>(""), Option.NONE);
        final CommandController notEqualController3 = new CommandController(path, new Option<String>(""), new Option<RemoteLoginInfo>(new RemoteLoginInfo("Hi", "Go", Option.NONE)));
        
        // Reflexive
        assertEquals(sut, sut);
        
        // Symmetric
        assertEquals(equalController, sut);
        assertEquals(sut, equalController);
        assertThat(sut, not(equalTo(notEqualController1)));
        assertThat(sut, not(equalTo(notEqualController2)));
        assertThat(sut, not(equalTo(notEqualController3)));
        assertThat(notEqualController1, not(equalTo(sut)));
        assertThat(notEqualController2, not(equalTo(sut)));
        assertThat(notEqualController3, not(equalTo(sut)));
        
        // HashCode
        assertEquals(equalController.hashCode(), sut.hashCode());
        assertTrue(sut.repOK());
    }
}