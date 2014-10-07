package ecjapp.eval.problem;

import ecjapp.util.Option;

/**
 * Stores information needed for executing a remote command via SSH.
 * 
 * It is assumed that the user has set up keys such that no password is
 * required.
 * 
 * @author Eric 'Siggy' Scott
 */
public final class RemoteLoginInfo {
    private final String username;
    private final String remoteServer;
    private final Option<String> remotePath;

    public RemoteLoginInfo(final String username, final String remoteServer, final Option<String> remotePath) {
        assert(username != null);
        assert(!username.isEmpty());
        assert(remoteServer != null);
        assert(!remoteServer.isEmpty());
        assert(remotePath != null);
        
        this.username = username;
        this.remoteServer = remoteServer;
        this.remotePath = remotePath;
        assert(repOK());
    }
    
    /** Copy constructor. */
    public RemoteLoginInfo(final RemoteLoginInfo ref) {
        this.username = ref.username;
        this.remoteServer = ref.remoteServer;
        this.remotePath = ref.remotePath;
        assert(repOK());
    }

    public String getSSHCommand(final String command) {
        return String.format("ssh %s@%s \"cd \'%s\'; %s\"", username, remoteServer, (remotePath.isDefined() ? remotePath.get() : ""), command);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Standard Methods">
    @Override
    public String toString() {
        return String.format("[%s: username=%s, remoteServer=%s, remotePath=%s]", this.getClass().getSimpleName(), username, remoteServer, remotePath);
    }
    
    @Override
    public boolean equals(final Object o) {
        if (!(o instanceof RemoteLoginInfo))
            return false;
        final RemoteLoginInfo ref = (RemoteLoginInfo) o;
        return username.equals(ref.username)
                && remoteServer.equals(ref.remoteServer)
                && remotePath.equals(ref.remotePath);
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 89 * hash + (this.username != null ? this.username.hashCode() : 0);
        hash = 89 * hash + (this.remoteServer != null ? this.remoteServer.hashCode() : 0);
        hash = 89 * hash + (this.remotePath != null ? this.remotePath.hashCode() : 0);
        return hash;
    }
    
    public final boolean repOK() {
        return username != null
                && !username.isEmpty()
                && remoteServer != null
                && !remoteServer.isEmpty()
                && remotePath != null;
    }
    //</editor-fold>
}
