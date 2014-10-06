package ecjapp;

import ec.EvolutionState;
import ec.Evolve;
import ec.eval.Slave;
import ec.util.ParameterDatabase;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class CARLsimEC {

    /** Retrieve the name of the jar we were launched from. */
    final private static String JAR_NAME = new File(CARLsimEC.class.getProtectionDomain()
            .getCodeSource()
            .getLocation()
            .getPath())
            .getName();
    final private static String USAGE = String.format("\nusage: java -jar %s <options> <ECJ params file>\n--slave Run ECJ in slave mode instead of master.", JAR_NAME);
    
    /** If any command-line option besides those in this list is found, the usage
     * message is displayed. */
    final private static List<String> allowedArgs = new ArrayList<String>() { {
            add("--slave");
    }};

    public static void main(final String[] args) {
        assert (args != null);
        
        if (args.length == 0)
            printUsageAndExit();
        
        final String lastArg = args[args.length - 1];
        
        if (args.length == 1) { // Launch a normal (master) run.
            try {
                final File pFile = new File(lastArg);
                if (!pFile.exists())
                    printUsageAndExit(String.format("The file '%s' does not exist.", lastArg));
                else if (!pFile.isFile())
                    printUsageAndExit(String.format("The path '%s' is not a file.", lastArg));
                
                final ParameterDatabase parameters = new ParameterDatabase(pFile);
                final EvolutionState state = Evolve.initialize(parameters, 1);
                state.run(EvolutionState.C_STARTED_FRESH);
            } catch (final FileNotFoundException ex) {
                Logger.getLogger(CARLsimEC.class.getName()).log(Level.SEVERE, null, ex);
            } catch (final IOException ex) {
                Logger.getLogger(CARLsimEC.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else if (args.length == 2) { // Launch a slave node.
            if (!allowedArgs.contains(args[0]))
                printUsageAndExit(String.format("Unknown argument: %s", args[0]));
            assert (args[0].equals("--slave"));
            Slave.main(new String[] { "-file", lastArg });
        }
        else
            printUsageAndExit();
    }
    
    private static void printUsageAndExit(final String message) {
        System.out.println(String.format("%s\n%s", message, USAGE));
        System.exit(0);
    }

    private static void printUsageAndExit() {
        System.out.println(USAGE);
        System.exit(0);
    }
}
