package ecjapp.util;

/**
 * An Option wraps an optional value -- that is, an Option may have a value, or
 * it may be equal to Option.NONE.  This is an idiom for handling optional
 * parameters without using null values, borrowed from Scala.
 * 
 * @author Eric 'Siggy' Scott
 */
public class Option<T> {
    public static final Option NONE = new Option();
    private final T val;
    
    private Option() { this.val = null; }
    
    public Option(final T val) { this.val = val; }
    
    public boolean isDefined() { return val != null; }
    
    public T get() { return val; }
    
    @Override
    public String toString() {
        return String.format("[%s: val=%s]", this.getClass().getSimpleName(), val);
    }
    
    @Override
    public boolean equals(final Object o) {
        if (!(o instanceof Option))
            return false;
        final Option ref = (Option) o;
        return (val == null ? ref.val == null : val.equals(ref.val));
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 59 * hash + (this.val != null ? this.val.hashCode() : 0);
        return hash;
    }
}
