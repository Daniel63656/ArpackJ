package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import static net.scoreworks.arpackj.MatrixOperations.asLinearOperation;
import static net.scoreworks.arpackj.MatrixOperations.invert;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.*;

public class SymmetricGeneralEigTests {
    private static final double epsilon = 0.0001;
    private static final Matrix A, A_sparse, M, M_sparse;
    static {
        double[][] data = {
                {0, 9, 0, 1, 0},
                {9, 0, 4, 0, 0},
                {0, 4, 2, 8, 5},
                {1, 0, 8, 1, 0},
                {0, 0, 5, 0, 7}};
        A = new Basic2DMatrix(data);
        A_sparse = CRSMatrix.from2DArray(data);
        //M is positive semi-definite
        double[][] dataM = {
                {9, 2, 3, 4, 5},
                {2, 8, 1, 2, 3},
                {3, 1, 7, 0, 1},
                {4, 2, 0, 6, 2},
                {5, 3, 1, 2, 5}};
        M = new Basic2DMatrix(dataM);
        M_sparse = CRSMatrix.from2DArray(dataM);
    }
    private static final double[] eigenvalues = {-1.7666, -0.81, 0.7156, 1.5177, 9.1618};
    private static final double[] eigenvectors = {  //each row is one eigenvector
            -0.2709963, 0.21255562, -0.13496624, 0.18759745, 0.09585369,
            0.15097938, -0.16383273, -0.21687804, 0.21463843, 0.06325637,
            0.1432534, 0.18610294, 0.05259292, -0.10265756, 0.15772811,
            0.03555523, -0.04953955, -0.24544191, -0.35011594, 0.27828486,
            0.58091305, 0.28752617, -0.26688309, -0.27229615, -0.7321415};

    private static void checkSolution(double[] d, double[] z, int[] idx) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<idx.length; i++) {
            Assertions.assertEquals(eigenvalues[idx[i]], d[i], epsilon);
        }

        //check eigenvectors match up to sign flip
        for (int i=0; i< idx.length; i++) {
            for (int j=0; j<eigenvalues.length; j++) {
                Assertions.assertEquals(Math.abs(eigenvectors[idx[i]*5 + j]), Math.abs(z[i*5 + j]), epsilon);
            }
        }
    }

    @Test
    public void testGeneralEigenvalueProblemLM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A, 4, M, "LM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 3, 4});
    }

    @Test
    public void testGeneralEigenvalueProblemLM_CRS() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A_sparse, 4, M_sparse, "LM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 3, 4});
    }

    @Test
    public void testGeneralEigenvalueProblemSM() {
        LinearOperation M_inv = asLinearOperation(invert(M));
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(asLinearOperation(A), A.rows(), 4, asLinearOperation(M), M_inv, "SM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertLM() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A, 4, M, "LM", 1, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertLM_CRS() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A_sparse, 4, M_sparse, "LM", 1, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertSM() {
        // (A - sigma*M)^-1 = A^-1 because sigma=0
        LinearOperation OP_inv = asLinearOperation(invert(A));
        SymmetricArpackSolver solver = eigsh_shiftInvert(A.rows(), 4, asLinearOperation(M), OP_inv, "SM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 3, 4});
    }
}