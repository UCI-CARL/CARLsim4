package ecjapp.eval.problem;

import ecjapp.util.PopulationToFile;
import ec.vector.DoubleVectorIndividual;
import ecjapp.util.Option;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.List;

/**
 * Handles running an external program via a shell command, sending it a List
 * of individuals in CSV format, and retrieving the output of the program.  The
 * program may be executed locally or remotely over SSH.
 * 
 * @author Eric 'Siggy' Scott
 */
public class CommandController {
    private final String commandPath;
    private final Option<String> arguments;
    private final Option<RemoteLoginInfo> remoteInfo;
    
    public String getCommandPath() { return commandPath; }
    public Option<String> getArguments() { return arguments; }
    public Option<RemoteLoginInfo> getRemoteLoginInfo() { return remoteInfo; }
    
    public CommandController(final String commandPath, final Option<String> arguments, final Option<RemoteLoginInfo> remoteLoginInfo) throws IllegalArgumentException {
        if (commandPath == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": binaryPath is null.");
        if (commandPath.isEmpty())
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": binaryPath is empty.");
        if (arguments == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": arguments is null.");
        if (remoteLoginInfo == null)
            throw new IllegalArgumentException(this.getClass().getSimpleName() + ": remoteLoginInfo is null.");
        this.commandPath = commandPath;
        this.arguments = arguments;
        this.remoteInfo = remoteLoginInfo; // Immutable
        assert(repOK());
    }
    
    /** Use CARLsim to execute a population of DoubleVectorIndividuals. 
     * @throws IOException, InterruptedException
     * @return A unique integer identifying the files used for this CARLsim invocation.
     */
    public String execute(final List<DoubleVectorIndividual> individuals, final Option<List<Integer>> subPopulations, final String additionalArguments) throws IOException, InterruptedException {
        assert(individuals != null);
        assert(additionalArguments != null);
        final String allArguments = (arguments.isDefined() ? arguments.get() : "") + (additionalArguments.isEmpty() ? "" : " " + additionalArguments);
        String carlsimShellCommand = String.format("%s %s", commandPath, allArguments);
        if (remoteInfo.isDefined())
            carlsimShellCommand = remoteInfo.get().getSSHCommand(carlsimShellCommand);
        
        //final Process p = Runtime.getRuntime().exec(carlsimShellCommand);
        final Process p = new ProcessBuilder(carlsimShellCommand.split(" ")).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        final Writer carlSimInput = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
        PopulationToFile.DoubleVectorIndividualsToFile(individuals, subPopulations, carlSimInput);
        carlSimInput.close(); // Sends EOF
        p.waitFor();
        
        return streamToString(p.getInputStream());
    }

    private String streamToString(final InputStream s) throws IOException {
        final BufferedReader reader = new BufferedReader(new InputStreamReader(s));
        final StringBuilder sb = new StringBuilder();
        String line = "";			
        while ((line = reader.readLine())!= null) {
            sb.append(line).append("\n");
        }
        return sb.toString();
    }
    
    // <editor-fold defaultstate="collapsed" desc="Standard Methods">
    final public boolean repOK() {
        return commandPath != null
                && !commandPath.isEmpty()
                && remoteInfo != null;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 37 * hash + (this.commandPath != null ? this.commandPath.hashCode() : 0);
        hash = 37 * hash + (this.arguments != null ? this.arguments.hashCode() : 0);
        hash = 37 * hash + (this.remoteInfo != null ? this.remoteInfo.hashCode() : 0);
        return hash;
    }
    
    @Override
    public boolean equals(final Object o) {
        if (!(o instanceof CommandController))
            return false;
        final CommandController ref = (CommandController) o;
        return commandPath.equals(ref.commandPath)
                && arguments.equals(ref.arguments)
                && remoteInfo.equals(ref.remoteInfo);
    }
    
    @Override
    public String toString() {
        return String.format("[%s: commandPath=\"%s\", arguments=\"%s\", remoteInfo=%s]", this.getClass().getSimpleName(), commandPath, arguments.toString(), remoteInfo.toString());
    }
    
    // </editor-fold>
}
