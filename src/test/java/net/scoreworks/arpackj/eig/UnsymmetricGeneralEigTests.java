/*
Copyright 2023 Daniel Maier

This file is part of ArpackJ (https://github.com/Daniel63656/ArpackJ)

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

package net.scoreworks.arpackj.eig;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import static net.scoreworks.arpackj.MatrixOperations.asLinearOperation;
import static net.scoreworks.arpackj.MatrixOperations.invert;
import static net.scoreworks.arpackj.TestUtils.checkSolution;

public class UnsymmetricGeneralEigTests {

    private static final Matrix A, A_sparse, M, M_sparse;   //M must be symmetric and positive definite
    static {
        double[][] data = {
                {2, 1, 9, 1, 5},
                {0, 0, 1, 0, 3},
                {3, 0, 7, 0, 1},
                {4, 0, 0, 0, 2},
                {1, 3, 0, 0, 5}};
        A = new Basic2DMatrix(data);
        A_sparse = CRSMatrix.from2DArray(data);
        double[][] dataM = {
                {2, 1, 0, 0, 1},
                {1, 5, 1, 0, 1},
                {0, 1, 7, 0, 0},
                {0, 0, 0, 9, 3},
                {1, 1, 0, 3, 4}};
        M = new Basic2DMatrix(dataM);
        M_sparse = CRSMatrix.from2DArray(dataM);
    }
    private static final Complex[] eigenvalues = {
            new Complex(1.94439737, 0.62976039),
            new Complex(1.94439737, -0.62976039),
            new Complex(0.1864016, 0),
            new Complex(-0.62206606, 0),
            new Complex(-0.39720187, 0)};

    private static final Complex[] eigenvectors = {
            new Complex(-0.60260525, -0.21133021),
            new Complex(0.16844472, -0.02691559),
            new Complex(-0.22956689, 0.00859672),
            new Complex(-0.28498002, 0.03814751),
            new Complex(0.59677596, -0.26743808),
            new Complex(-0.60260525, 0.21133021),
            new Complex(0.16844472, 0.02691559),
            new Complex(-0.22956689, -0.00859672),
            new Complex(-0.28498002, -0.03814751),
            new Complex(0.59677596, 0.26743808),
            new Complex(-0.35563866, 0.0),
            new Complex(0.03035096, 0.0),
            new Complex(0.20090202, 0.0),
            new Complex(-0.90945674, 0.0),
            new Complex(-0.07160149, 0.0),
            new Complex(-0.80254223, 0.0),
            new Complex(0.0255325, 0.0),
            new Complex(0.20866545, 0.0),
            new Complex(0.55787844, 0.0),
            new Complex(0.02245973, 0.0),
            new Complex(0.4452752, 0.0),
            new Complex(-0.42795898, 0.0),
            new Complex(-0.14558052, 0.0),
            new Complex(-0.72857695, 0.0),
            new Complex(0.25799814, 0.0)
    };

    @Test
    public void testGeneralEigenvalueProblemLM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(A, M, 3, "LM", null, 100, 1e-15);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 3}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemLM_CRS() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(A_sparse, M_sparse, 3, "LM", null, 100, 1e-15);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 3}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemSM() {
        Matrix M_inv = invert(M);
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs(asLinearOperation(A), asLinearOperation(M), asLinearOperation(M_inv), A.rows(), 3, "SM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{2, 3, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertRealLM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertReal(A, M, 3, "LM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{1, 0, 2}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertRealLM2() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertReal(A, M, 3, "LM", new Complex(0, 1), null, 100, 1e-15);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{1, 0, 3}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertRealSM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertReal(A, M, 3, "SM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{3, 2, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertRealSM2() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertReal(A, M, 3, "SM", new Complex(0, 1), null, 100, 1e-15);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{1, 0, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertImagLM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertImag(A, M, 3, "LM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{1, 0, 2}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertImagLM2() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertImag(A_sparse, M_sparse, 3, "LM", new Complex(0, 1), null, 100, 1e-15);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{3, 2, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertImagSM() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertImag(A, M, 3, "SM", new Complex(1, 1), null, 100, 1e-15);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{3, 1, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertImagSM2() {
        UnsymmetricArpackSolver solver = MatrixDecomposition.eigs_shiftInvertImag(A, M, 3, "SM", new Complex(0, 1), null, 100, 1e-15);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        Complex[] d = solver.getEigenvalues();
        Complex[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{1, 0, 3}, d, z);
    }
}