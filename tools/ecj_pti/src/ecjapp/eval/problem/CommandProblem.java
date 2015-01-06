package ecjapp.eval.problem;

import ec.EvolutionState;
import ec.Individual;
import ec.Problem;
import ec.util.Output.OutputExitException;
import ec.util.Parameter;
import ec.vector.DoubleVectorIndividual;
import ecjapp.eval.problem.objective.ObjectiveFunction;
import ecjapp.util.Misc;
import ecjapp.util.Option;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * An ECJ Problem that sends a population of DoubleVectorIndividuals to an
 * external command for simulation as an intermediate step in fitness evaluation.
 * 
 * "simulationCommand" and "simulationCommandArguments" are used to specify
 * the external command that launches the simulation.
 * 
 * If the external command requires arguments that depend on the state of the
 * evolutionary run (such as a generation or thread number), they can be
 * provided with a DynamicArguments object via the "dynamicArgument" parameter.
 * The output of dynamicArguments.get() will be appended to the simulation
 * command at runtime.
 * 
 * The external simulation needs to write the evaluation result for each
 * individual to stout, one result per line.
 * 
 * How you specify the ObjectiveFunction for the "objective" parameter depends
 * on the simulation result format.  If results come back as fitness values,
 * then you'll want to set "objective" to ecjapp.eval.problem.objective.StringToDoubleObjective.
 * If the results come back as some kind of complex phenotype (such as a
 * simulation trace), you should specify a custom ObjectiveFunction that
 * converts the phenotype into a fitness value.
 * 
 * @author Eric 'Siggy' Scott
 */
public class CommandProblem extends Problem implements SimpleGroupedProblemForm {
    public final static String P_REMOTE_SERVER = "remoteServer";
    public final static String P_REMOTE_USERNAME = "remoteUsername";
    public final static String P_REMOTE_PATH = "remotePath";
    public final static String P_SIMULATION_COMMAND_PATH = "simulationCommand";
    public final static String P_SIMULATION_COMMAND_ARGUMENTS = "simulationCommandArguments";
    public final static String P_OBJECTIVE_FUNCTION = "objective";
    public final static String P_DYNAMIC_ARGUMENTS = "dynamicArguments";
    
    private ObjectiveFunction objective;
    private Option<DynamicArguments> dynamicArguments;
    private CommandController controller;
    
    public ObjectiveFunction getObjective() { return objective; }
    public CommandController getCommandController() { return controller; }
    
    //<editor-fold defaultstate="collapsed" desc="Setup">
    @Override
    public void setup(final EvolutionState state, final Parameter base) throws OutputExitException {
        if (state == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": state is null.");
        if (state.parameters == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": state.parameters is null.");
        if (base == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": base is null.");
        super.setup(state, base);
        
        final Option<RemoteLoginInfo> remoteInfo = getRemoteLoginInfoFromParams(state, base);
        final String commandPath = getCommandPathFromParams(state, base, remoteInfo.equals(Option.NONE));
        final Option<String> commandArguments = new Option<String>(state.parameters.getString(base.push(P_SIMULATION_COMMAND_ARGUMENTS), null));
        
        this.objective = (ObjectiveFunction) Misc.getInstanceOfRequiredParameter(state, base.push(P_OBJECTIVE_FUNCTION), ObjectiveFunction.class);
        this.objective.setup(state, base.push(P_OBJECTIVE_FUNCTION));
        if (state.parameters.exists(base.push(P_DYNAMIC_ARGUMENTS), null)) {
            this.dynamicArguments = new Option<DynamicArguments>((DynamicArguments) Misc.getInstanceOfRequiredParameter(state, base.push(P_DYNAMIC_ARGUMENTS), DynamicArguments.class));
	    this.dynamicArguments.get().setup(state, base.push(P_DYNAMIC_ARGUMENTS));
	}
        else
            this.dynamicArguments = Option.NONE;
        this.controller = new CommandController(commandPath, commandArguments, remoteInfo);
        
        assert(repOK());
    }

    /** Retrieve the path to the external executable.  
     * @param checkExists If true, this method will check if the file exists
     * on the local machine and throw an OutputExitException if it doesn't. 
     * @throws OutputExitException */
    private String getCommandPathFromParams(final EvolutionState state, final Parameter base, final boolean checkExists) throws OutputExitException {
        assert(state != null);
        assert(base != null);
        String commandPathString = state.parameters.getString(base.push(P_SIMULATION_COMMAND_PATH), null);
        if (commandPathString == null || commandPathString.equals(""))
            state.output.fatal(String.format("%s: required parameter %s is undefined or empty.", this.getClass().getSimpleName(), base.push(P_SIMULATION_COMMAND_PATH)));
        if (checkExists) {
            final File commandPath = state.parameters.getFile(base.push(P_SIMULATION_COMMAND_PATH), null);
            if (!commandPath.exists())
                state.output.fatal(String.format("%s: the path %s specified by parameter %s does not exist.", this.getClass().getSimpleName(), commandPathString, base.push(P_SIMULATION_COMMAND_PATH)));
            if (!commandPath.isFile())
                state.output.fatal(String.format("%s: the path %s specified by parameter %s is not a file.", this.getClass().getSimpleName(), commandPathString, base.push(P_SIMULATION_COMMAND_PATH)));
            commandPathString = commandPath.getAbsolutePath();
        }
        return commandPathString;
    }
    
    /** Retrieve remote login parameters if the command is being invoked on a non-local machine. */
    private static Option<RemoteLoginInfo> getRemoteLoginInfoFromParams(final EvolutionState state, final Parameter base) throws OutputExitException {
        assert(state != null);
        assert(base != null);
        Option<RemoteLoginInfo> remoteInfo;
        final String remoteServer = state.parameters.getString(base.push(P_REMOTE_SERVER), null);
        if (remoteServer == null || remoteServer.isEmpty())
            remoteInfo = Option.NONE;
        else {
            final String remoteUsername = state.parameters.getString(base.push(P_REMOTE_USERNAME), null);
            if (remoteUsername == null || remoteUsername.isEmpty())
                state.output.fatal(String.format("%s: parameter %s is defined, but %s is undefined.", CommandProblem.class.getSimpleName(), base.push(P_REMOTE_SERVER), base.push(P_REMOTE_USERNAME)));
            final Option<String> remotePath = new Option<String>(state.parameters.getString(base.push(P_REMOTE_PATH), null));
            remoteInfo = new Option<RemoteLoginInfo>(new RemoteLoginInfo(remoteUsername, remoteServer, remotePath));
        }
        return remoteInfo;
    }
    // </editor-fold>
    
    @Override
    public void evaluate(final EvolutionState state, final Individual[] individuals, final int from, final int to, final int subpopulation, final int threadnum) throws OutputExitException {
        assert(state != null);
        assert(individuals != null);
        assert(from >= 0);
        assert(to >= from);
        assert(to <= individuals.length);
        assert(subpopulation >= 0);
        assert(threadnum >= 0);
        assert(Misc.containsOnlySubtypesOf(individuals, DoubleVectorIndividual.class));
        
        final List<DoubleVectorIndividual> chunk = new ArrayList<DoubleVectorIndividual>() {{
		for(int i = from; i < to; i++)
		    if (!individuals[i].evaluated)
			add((DoubleVectorIndividual)individuals[i]);
        }};
        
        try {
            final String extraArguments = (dynamicArguments.isDefined()) ? dynamicArguments.get().get(state, individuals, from, to, subpopulation, threadnum) : "";
            final String simulationResult = controller.execute(chunk, extraArguments);
            final String[] lines = simulationResult.split("\n");
            if (lines.length != chunk.size())
                throw new IllegalStateException(String.format("%s: Sent %d individuals to external command %s, but the returned simulation results had %d lines.", this.getClass().getSimpleName(), chunk.size(), controller.getCommandPath(), lines.length));
            for (int i = 0; i < lines.length; i++) {
                final Individual ind = individuals[from + i];
                ind.fitness = objective.evaluate(state, lines[i]);
                ind.evaluated = true;
            }
            
        }
        catch (final Exception e) {
            state.output.fatal(e.toString());
        }
        assert(repOK());
    }
    
    public final boolean repOK() {
        return controller != null;
    }
}
