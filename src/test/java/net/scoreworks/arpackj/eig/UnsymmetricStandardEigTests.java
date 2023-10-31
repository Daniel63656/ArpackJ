package net.scoreworks.arpackj.eig;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

public class UnsymmetricStandardEigTests {
    private static final double epsilon = 0.0001;

    private static final Matrix A, A_sparse;
    private static final Complex[] eigenvalues = new Complex[5];
    static {
        double[][] data = {
                {9, 1, 9, 1, 5},
                {6, 8, 1, 2, 3},
                {3, 1, 7, 5, 1},
                {4, 9, 0, 6, 2},
                {5, 3, 1, 2, 5}};
        A = new Basic2DMatrix(data);
        A_sparse = CRSMatrix.from2DArray(data);

        double[] eigenvalues_real = {20.2520875, 5.86008376, 5.86008376, 0.26825312, 2.75949185};
        double[] eigenvalues_imag = {0.0, 4.5560381, -4.5560381, 0.0, 0.0};
        for (int i = 0; i < eigenvalues_real.length; i++)
            eigenvalues[i] = new Complex(eigenvalues_real[i], eigenvalues_imag[i]);
    }


    private static void checkSolution(Complex[] d, Complex[] z, int[] idx) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<idx.length; i++) {
            Assertions.assertEquals(eigenvalues[idx[i]].getReal(), d[i].getReal(), epsilon);
            Assertions.assertEquals(eigenvalues[idx[i]].getImaginary(), d[i].getImaginary(), epsilon);
        }

        //check eigenvectors match up to sign flip
        /*for (int i=0; i< idx.length; i++) {
            for (int j=0; j<eigenvalues.length; j++) {
                Assertions.assertEquals(Math.abs(eigenvectors[idx[i]*5 + j]), Math.abs(z[i*5 + j]), epsilon);
            }
        }*/
    }

    @Test
    public void testStandardEigenvalueProblemLM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(A, 3, "LM", null, 100, 1e-15);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2});
    }
}