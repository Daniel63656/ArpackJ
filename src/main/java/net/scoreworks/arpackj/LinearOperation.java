/*
Copyright 2023 Daniel Maier

This file is part of ArpackJ (https://github.com/Daniel63656/ArpackJ)

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

package net.scoreworks.arpackj;

/**
 * Interface for matrix vector operations. This function is ++intended to return a new result array while leaving the
 * passed one untouched. Supports the case that x is saved as part of a bigger array.
 * See {@link MatrixOperations} for example implementations
 */

public interface LinearOperation {
    /**
     * @param x vector to apply the function to
     * @param offset 0 if x spans the entire array, starting position of x in array otherwise. The
     *               length of x is implicitly handled by the implementation
     * @return the result of OP @ x
     */
    double[] apply(final double[] x, final int offset);
}