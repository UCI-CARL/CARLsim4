package ecjapp;

import ec.Evolve;
import ec.eval.Slave;
import java.io.File;

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

    public static void main(final String[] args) {
        assert (args != null);
        
        if (args.length > 0 && args[0].equals("--slave"))
            Slave.main(tail(args));
        else
            Evolve.main(args);
    }
    
    private static String[] tail(final String[] array) {
        assert(array != null);
        final String[] result = new String[array.length - 1];
        for (int i = 0; i < result.length; i++)
            result[i] = array[i+1];
        return result;
    }
}
