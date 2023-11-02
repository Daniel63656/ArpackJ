package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import static net.scoreworks.arpackj.MatrixOperations.asLinearOperation;
import static net.scoreworks.arpackj.MatrixOperations.invert;
import static net.scoreworks.arpackj.TestUtils.checkSolution;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.*;

public class SymmetricGeneralEigTests {
    private static final Matrix A, A_sparse, M, M_sparse, B, B_sparse, Mb, Mb_sparse;
    static {
        double[][] data = {
                {0, 9, 0, 1, 0},
                {9, 0, 4, 0, 0},
                {0, 4, 2, 8, 5},
                {1, 0, 8, 1, 0},
                {0, 0, 5, 0, 7}};
        A = new Basic2DMatrix(data);
        A_sparse = CRSMatrix.from2DArray(data);
        //M is positive definite
        double[][] dataM = {
                {9, 2, 3, 4, 5},
                {2, 8, 1, 2, 3},
                {3, 1, 7, 0, 1},
                {4, 2, 0, 6, 2},
                {5, 3, 1, 2, 5}};
        M = new Basic2DMatrix(dataM);
        M_sparse = CRSMatrix.from2DArray(dataM);
        //B is symmetric, positive-definite
        double[][] dataB = {
                {1.5, 0.2,-0.1,-0.3,-0.4},
                {0.2, 1.6,-0.2,-0.4,-0.5},
                {-0.1,-0.2, 1.7,-0.5,-0.6},
                {-0.3,-0.4,-0.5, 1.8,-0.7},
                {-0.4,-0.5,-0.6,-0.7, 1.9}};
        B = new Basic2DMatrix(dataB);
        B_sparse = CRSMatrix.from2DArray(dataB);
        //Mb is symmetric indefinite
        double[][] dataMb = {
                {2.3,-1.2,-1.4,-1.6,-1.8},
                {-1.2,-2.4,-1.6,-1.8,-2.0},
                {-1.4,-1.6, 2.5,-1.9,-2.2},
                {-1.6,-1.8,-1.9,-2.6,-2.4},
                {-1.8,-2.0,-2.2,-2.4, 2.7}};
        Mb = new Basic2DMatrix(dataMb);
        Mb_sparse = CRSMatrix.from2DArray(dataMb);
    }
    private static final double[] eigenvalues_A = {-1.7666, -0.81, 0.7156, 1.5177, 9.1618};
    private static final double[] eigenvectors_A = {  //each row is one eigenvector
            -0.2709963, 0.21255562, -0.13496624, 0.18759745, 0.09585369,
            0.15097938, -0.16383273, -0.21687804, 0.21463843, 0.06325637,
            0.1432534, 0.18610294, 0.05259292, -0.10265756, 0.15772811,
            0.03555523, -0.04953955, -0.24544191, -0.35011594, 0.27828486,
            0.58091305, 0.28752617, -0.26688309, -0.27229615, -0.7321415};
    private static final double[] eigenvalues_B = {-2.75737884, 0.43884819, 0.50518823, 0.92988871, -0.02900193};
    private static final double[] eigenvectors_B = {  //each row is one eigenvector
            -0.06975583, 0.56816724, -0.03562472, -0.42445665, -0.03213135,
            0.65900697, -0.0841094, -0.4292224, -0.01995139, -0.08569716,
            0.198436, -0.0462395, 0.36077647, 0.00276442, -0.49141503,
            0.35982939, -0.29633667, 0.38990511, -0.2601493, 0.36126889,
            -0.61085061, -0.89190993, -1.00227489, -1.21469481, -1.20005095};

    @Test
    public void testGeneralEigenvalueProblemLM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A, M, 4, "LM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{0, 1, 3, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemLM_CRS() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A_sparse, M_sparse, 4, "LM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{0, 1, 3, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemSM() {
        LinearOperation M_inv = asLinearOperation(invert(M));
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(asLinearOperation(A), asLinearOperation(M), M_inv, A.rows(), 4, "SM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{0, 1, 2, 3}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertLM() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A, M, 4, "LM", 1, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{0, 1, 2, 3}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertLM_CRS() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A_sparse, M_sparse, 4, "LM", 1, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{0, 1, 2, 3}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertSM() {
        // (A - sigma*M)^-1 = A^-1 because sigma=0
        LinearOperation OP_inv = asLinearOperation(invert(A));
        SymmetricArpackSolver solver = eigsh_shiftInvert(asLinearOperation(M), OP_inv, A.rows(), 4, "SM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{0, 1, 3, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemBucklingLM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh_buckling(B, Mb, 3, "LM", 1, null, 100, 1e-5);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_B, eigenvectors_B, new int[]{1, 2, 3}, d, z);
    }
    @Test
    public void testGeneralEigenvalueProblemBucklingSM() {
        LinearOperation OPinv = asLinearOperation(invert(B.subtract(Mb.multiply(1))));
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh_buckling(asLinearOperation(B_sparse), OPinv, B.rows(), 3, "SM", 1, null, 100, 1e-5);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_B, eigenvectors_B, new int[]{0, 4, 1}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemBucklingSM2() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh_buckling(B_sparse, Mb_sparse, 3, "SM", 0.1, null, 100, 1e-5);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_B, eigenvectors_B, new int[]{0, 4, 3}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemCayleyLM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh_cayley(A, M, 3, "LM", 1, null, 100, 1e-5);
        Assertions.assertSame(5, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{2, 3, 4}, d, z);
    }

    @Test
    public void testGeneralEigenvalueProblemCayleySM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh_cayley(A, M, 3, "SM", 1, null, 100, 1e-5);
        Assertions.assertSame(5, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues_A, eigenvectors_A, new int[]{0, 1, 4}, d, z);
    }
}