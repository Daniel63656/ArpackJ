/*
Copyright 2023 Daniel Maier

This file is part of ArpackJ (https://github.com/Daniel63656/ArpackJ)

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

package net.scoreworks.arpackj;

/**
 * A functional interface for representing a matrix {@code A} as a left-side multiplication operation on a vector {@code x}.
 * Calling {@link #apply(double[], int)} computes the result of {@code A * x} and returns it as a new array.
 * <p>
 * Example implementations can be found in {@link MatrixOperations}.
 */
public interface LinearOperation {
    /**
     * @param x vector array to apply the matrix operation to
     * @param offset the starting position of the vector within the array. Use {@code 0} if {@code x}
     *               contains only the vector itself or the vector starts at the beginning of the array. The vector length
     *               must match the size of the underlying {@link LinearOperation}.
     * @return the result of OP @ x
     */
    double[] apply(final double[] x, final int offset);
}