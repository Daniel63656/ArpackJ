package net.scoreworks.arpackj.eig;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import static net.scoreworks.arpackj.MatrixOperations.asLinearOperation;

public class UnsymmetricStandardEigTests {
    private static final double epsilon = 0.0001;

    private static final Matrix A, A_sparse;
    static {
        double[][] data = {
                {9, 1, 9, 1, 5},
                {6, 8, 1, 2, 3},
                {3, 1, 7, 5, 1},
                {4, 9, 0, 6, 2},
                {5, 3, 1, 2, 5}};
        A = new Basic2DMatrix(data);
        A_sparse = CRSMatrix.from2DArray(data);
    }
    private static final Complex[] eigenvalues = {
            new Complex(20.2520875, 0),
            new Complex(5.86008376, 4.5560381),
            new Complex(5.86008376, -4.5560381),
            new Complex(0.26825312, 0),
            new Complex(2.75949185, 0)};
    private static final Complex[] eigenvectors = {
                new Complex(-0.53615599, 0.0),
                new Complex(-0.45944253, 0.0),
                new Complex(-0.36781234, 0.0),
                new Complex(-0.49036391, 0.0),
                new Complex(-0.35455132, 0.0),
                new Complex(0.52948655, -0.04077902),
                new Complex(-0.20530521, -0.26003448),
                new Complex(-0.02074077, 0.4123898),
                new Complex(-0.62946234, 0.0),
                new Complex(-0.09106364, -0.18221402),
                new Complex(0.52948655, 0.04077902),
                new Complex(-0.20530521, 0.26003448),
                new Complex(-0.02074077, -0.4123898),
                new Complex(-0.62946234, 0.0),
                new Complex(-0.09106364, 0.18221402),
                new Complex(0.66575413, 0.0),
                new Complex(-0.33120884, 0.0),
                new Complex(-0.34626462, 0.0),
                new Complex(0.2370903, 0.0),
                new Complex(-0.52053928, 0.0),
                new Complex(-0.3695004, 0.0),
                new Complex(-0.12271274, 0.0),
                new Complex(-0.23191235, 0.0),
                new Complex(0.27322528, 0.0),
                new Complex(0.84851378, 0.0)
        };


    private static void checkSolution(Complex[] d, Complex[] z, int[] idx) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<idx.length; i++) {
            Assertions.assertEquals(eigenvalues[idx[i]].getReal(), d[i].getReal(), epsilon);
            Assertions.assertEquals(eigenvalues[idx[i]].getImaginary(), d[i].getImaginary(), epsilon);
        }

        //check eigenvectors match up to sign flip
        Complex rotation = null;
        for (int i=0; i<idx.length; i++) {
            for (int j=0; j<eigenvalues.length; j++) {
                //each vector can be rotated in the complex plane so calculate rotation factor on first element
                if (j == 0) rotation = eigenvectors[idx[i]*5].divide(z[i*5]);
                Assertions.assertEquals(eigenvectors[idx[i]*5 + j].getReal(), z[i*5 + j].multiply(rotation).getReal(), epsilon);
                Assertions.assertEquals(eigenvectors[idx[i]*5 + j].getImaginary(), z[i*5 + j].multiply(rotation).getImaginary(), epsilon);
            }
        }
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

    @Test
    public void foo() {
        int n = 5;
        int nev = 3;
        for (int i=0; i<15; i++) {
            System.out.println(n*(nev-1-i/n) + i%n);
        }
    }

    @Test
    public void testStandardEigenvalueProblemLM_CRS() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(A_sparse, 3, "LM", null, 100, 1e-15);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2});
    }

    @Test
    public void testStandardEigenvalueProblemSM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(asLinearOperation(A), A.rows(), 3, "SM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{3, 4, 2});
    }
}