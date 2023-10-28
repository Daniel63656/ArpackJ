package net.scoreworks.arpackj;

/**
 * Perform efficient Matrix vector operations where vector is (subset of) a double[] array. This function is intended to
 * return a new array and leave the passed one untouched
 */
@FunctionalInterface
public interface LinearOperation {
    /**
     * @param x vector to apply the function to
     * @param offset specify starting position of x in bigger array. If x spans the entire array, use 0
     * @return the result of OP @ x
     */
    double[] apply(final double[] x, final int offset);
}