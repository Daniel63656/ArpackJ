package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import java.util.Arrays;

import static net.scoreworks.arpackj.MatrixOperations.*;
import static net.scoreworks.arpackj.TestUtils.checkSolution;

public class UnsymmetricStandardEigTests {
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
                new Complex(0.84851378, 0.0)};

    @Test
    public void testStandardEigenvalueProblemLM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(A, 3, "LM", null, 100, 1e-15);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 2}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemLM_CRS() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(A_sparse, 3, "LM", null, 100, 1e-15);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 2}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemSM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(asLinearOperation(A), A.rows(), 3, "SM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{3, 4, 1}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertRealLM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertReal(A, null, 3, "LM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{3, 4, 1}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertRealSM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertReal(A, null, 3, "SM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{2, 1, 0}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertRealLI() {
        double[] Z = new double[2*A.rows()*A.columns()];
        for (int i=0; i<A.rows(); i++) {
            for (int j=0; j<A.columns(); j++) {
                Z[2*(i*A.columns() + j)] = A.get(i, j);
            }
            //subtract sigma on trace
            Z[2*(i*A.columns() + i)] -= 1;
            Z[2*(i*A.columns() + i) + 1] = -1;
        }
        invertComplex(A.columns(), Z);
        LinearOperation OPinv = asLinearOperationReal(A.rows(), A.columns(), Z);
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertReal(asLinearOperation(A), null,
                OPinv, A.rows(), 2, "LI", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{2, 1}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertImagLM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertImag(A, null, 3, "LM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{3, 4, 1}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertImagSM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertImag(A, null, 3, "SM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{2, 1, 0}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertImagLI() {
        double[] Z = new double[2*A.rows()*A.columns()];
        for (int i=0; i<A.rows(); i++) {
            for (int j=0; j<A.columns(); j++) {
                Z[2*(i*A.columns() + j)] = A.get(i, j);
            }
            //subtract sigma on trace
            Z[2*(i*A.columns() + i)] -= 1;
            Z[2*(i*A.columns() + i) + 1] = -1;
        }
        invertComplex(A.columns(), Z);
        LinearOperation OPinv = asLinearOperationImag(A.rows(), A.columns(), Z);
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertImag(asLinearOperation(A), null,
                OPinv, A.rows(), 2, "LI", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{2, 1}, d, z);
    }
}