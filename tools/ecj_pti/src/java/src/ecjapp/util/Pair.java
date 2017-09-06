package ecjapp.util;

/**
 *
 * @author Eric 'Siggy' Scott
 */
public class Pair<T> {
    private final T x;
    private final T y;
    
    public T getX() { return x; }
    public T getY() { return y; }
    
    public Pair(final T x, final T y) {
        assert(x != null);
        assert(y != null);
        this.x = x;
        this.y = y;
        assert(repOK());
    }
    
    // <editor-fold defaultstate="collapsed" desc="Standard Methods">
    public final boolean repOK() {
        return x != null && y != null;
    }

    @Override
    public boolean equals(final Object o) {
        if (o == this)
            return true;
        if (!(o instanceof Pair))
            return false;
        final Pair<T> ref = (Pair<T>) o;
        return x.equals(ref.x)
                && y.equals(ref.y);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 47 * hash + (this.x != null ? this.x.hashCode() : 0);
        hash = 47 * hash + (this.y != null ? this.y.hashCode() : 0);
        return hash;
    }

    @Override
    public String toString() {
        return String.format("[%s: x=%s, y=%s]", this.getClass().getSimpleName(), x, y);
    }
    // </editor-fold>
}
