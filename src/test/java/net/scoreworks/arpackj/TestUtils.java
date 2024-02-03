/*
Copyright 2023 Daniel Maier

This file is part of ArpackJ (https://github.com/Daniel63656/ArpackJ)

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

package net.scoreworks.arpackj;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.Assertions;
import org.la4j.iterator.MatrixIterator;
import org.la4j.matrix.SparseMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import java.util.Random;

public final class TestUtils {
    private static final double epsilon = 0.0001;
    private TestUtils() {}

    public static void checkSolution(double[] d_expected, double[] z_expected, int[] slice, double[] d_actual, double[] z_actual) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<slice.length; i++) {
            Assertions.assertEquals(d_expected[slice[i]], d_actual[i], epsilon);
        }
        System.out.println("All eigenvalues correct");

        //check eigenvectors match up to sign flip
        for (int i=0; i< slice.length; i++) {
            if (i > 0) System.out.println("eigenvector "+i+" correct");
            for (int j=0; j<d_expected.length; j++) {
                Assertions.assertEquals(Math.abs(z_expected[slice[i]*5 + j]), Math.abs(z_actual[i*5 + j]), epsilon);
            }
        }
        System.out.println("eigenvector "+slice.length+" correct");
    }

    public static void checkSolution(Complex[] d_expected, Complex[] z_expected, int[] slice, Complex[] d_actual, Complex[] z_actual) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<slice.length; i++) {
            Assertions.assertEquals(d_expected[slice[i]].getReal(), d_actual[i].getReal(), epsilon);
            Assertions.assertEquals(d_expected[slice[i]].getImaginary(), d_actual[i].getImaginary(), epsilon);
        }
        System.out.println("All eigenvalues correct");

        Complex rotation = null;
        for (int i=0; i<slice.length; i++) {
            if (i > 0) System.out.println("eigenvector "+i+" correct");
            for (int j=0; j<d_expected.length; j++) {
                //each vector can be rotated in the complex plane so calculate rotation factor on first element
                if (j == 0) rotation = z_expected[slice[i]*5].divide(z_actual[i*5]);
                Assertions.assertEquals(z_expected[slice[i]*5 + j].getReal(), z_actual[i*5 + j].multiply(rotation).getReal(), epsilon);
                Assertions.assertEquals(z_expected[slice[i]*5 + j].getImaginary(), z_actual[i*5 + j].multiply(rotation).getImaginary(), epsilon);
            }
        }
        System.out.println("eigenvector "+slice.length+" correct");
    }

    public static CRSMatrix generateRandomSparseMatrix(int rows, int columns, double sparsity) {
        Random random = new Random();
        CRSMatrix matrix = new CRSMatrix(rows, columns);
        //generate random non-zero elements and insert them into the matrix
        for (int i=0; i<rows*columns; i++) {
            if (random.nextDouble() < sparsity)
                matrix.set(i/columns, i%columns, Math.random());
        }
        return matrix;
    }

    public static float checkSparsity(SparseMatrix A) {
        MatrixIterator it = A.nonZeroIterator();
        int count = 0;
        while (it.hasNext()) {
            count++;
            it.next();
        }
        return count/(float)(A.rows()*A.columns());
    }
}
