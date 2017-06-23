package ecjapp;

import ec.EvolutionState;
import ec.Evolve;
import ec.simple.SimpleFitness;
import ec.util.Parameter;
import ec.util.ParameterDatabase;
import ecjapp.eval.problem.objective.ObjectiveFunction;
import ecjapp.util.Option;
import ecjapp.util.Pair;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An main class that accepts phenotype values on standard in and uses an objective
 * function to process them into fitness values.
 * 
 * @author Eric O. Scott
 */
public class ObjectiveEvaluator {
    public final static String P_OBJECTIVE_FUNCTION = "eval.problem.objective";
    
    public static void main(final String[] args) {
        assert (args != null);
        
        try {
            final ParameterDatabase parameters = Evolve.loadParameterDatabase(args);
            
            // Initialize the objective we want to evaluate
            final ObjectiveFunction<SimpleFitness> objective = (ObjectiveFunction) parameters.getInstanceForParameter(new Parameter(P_OBJECTIVE_FUNCTION), null, ObjectiveFunction.class);
            // Initialize an EvolutionState (which may provide other context the objective needs to set itself up)
            final EvolutionState state = Evolve.initialize(parameters, (int) System.currentTimeMillis());
            
            // Load phenotypes from standard in, one per line
            final BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
            String l;
            while ((l = stdin.readLine()) != null && l.length()!= 0) {
                final SimpleFitness result = objective.evaluate(state, l);
                System.out.println(result.fitness());
            }
        }
        catch (final Exception e) {
            e.printStackTrace(System.err);
            Logger.getLogger(ObjectiveEvaluator.class.toString()).log(Level.SEVERE, "", e);
            System.exit(1);
        }
    }
    
    public static Option<String> getOption(final String optionName, final String[] args) {
        assert(optionName != null);
        assert(args != null);
        for (int i = 0; i < args.length - 1; i++) {
            if (!args[i].isEmpty() && args[i].charAt(0) == '-' && args[i].equals(String.format("-%s", optionName)))
                return new Option<String>(args[i+1]);
        }
        return Option.NONE;
    }
    
    public static List<Pair<String>> getAllEqualsOption(final String optionName, final String[] args) {
        assert(optionName != null);
        assert(args != null);
        final List<Pair<String>> result = new ArrayList<Pair<String>>();
        for (int i = 0; i < args.length - 1; i++) {
            if (!args[i].isEmpty() && args[i].charAt(0) == '-' && args[i].equals(String.format("-%s", optionName))) {
                final String combinedArg = args[i+1];
                if (!combinedArg.contains("="))
                    throw new IllegalArgumentException(String.format("%s: Argument to option '-%s' must be of the form 'A=B'.", ObjectiveEvaluator.class.getSimpleName(), optionName));
                final String[] parts = combinedArg.split("=");
                if (parts.length != 2)
                    throw new IllegalArgumentException(String.format("%s: Argument to option '-%s' must be of the form 'A=B'.", ObjectiveEvaluator.class.getSimpleName(), optionName));
                result.add(new Pair<String>(parts[0], parts[1]));
            }
        }
        return result;
    }
}
