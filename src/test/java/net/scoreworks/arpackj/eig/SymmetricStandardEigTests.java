package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import static net.scoreworks.arpackj.MatrixOperations.*;
import static net.scoreworks.arpackj.TestUtils.checkSolution;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.*;

public class SymmetricStandardEigTests {
    private static final Matrix A, A_sparse;
    static {
        double[][] data = {
                {0, 9, 0, 1, 0},
                {9, 0, 4, 0, 0},
                {0, 4, 2, 8, 5},
                {1, 0, 8, 1, 0},
                {0, 0, 5, 0, 7}};
        A = new Basic2DMatrix(data);
        A_sparse = CRSMatrix.from2DArray(data);
    }
    private static final double[] eigenvalues = {-10.9023, -5.73301, 4.66164, 8.42532, 13.5483};
    private static final double[] eigenvectors = {  //each row is one eigenvector
            0.53365, -0.60749, 0.455019, -0.350674, -0.12708,
            -0.4676, 0.36081, 0.53506, -0.56629, -0.21011,
            0.23612, 0.190267, -0.30952, -0.61175, 0.66182,
            0.60465, 0.57489, -0.14954, -0.07968, -0.52459,
            0.27418, 0.36616, 0.62330, 0.41923, 0.47592};

    @Test
    public void testStandardEigenvalueProblemLM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A, 4, "LM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 3, 4}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemLM_CRS() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A_sparse, 4, "LM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 3, 4}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemSM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(asLinearOperation(A), A.rows(), 4, "SM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 2, 3}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertLM() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A, null, 4, "LM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 2, 3}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertLM_CRS() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A_sparse, null, 4, "LM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 2, 3}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertSM() {
        // (A - sigma*I)^-1 = A^-1 because sigma=0
        LinearOperation OP_inv = asLinearOperation(invert(A));
        SymmetricArpackSolver solver = eigsh_shiftInvert(null, OP_inv, A.rows(), 4, "SM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 3, 4}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemCayleyLM() {
        SymmetricArpackSolver solver = eigsh_cayley(A, null, 3, "LM", 1, null, 100, 1e-5);
        Assertions.assertSame(5, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{2, 3, 4}, d, z);
    }

    @Test
    public void testStandardEigenvalueProblemCayleySM() {
        double[] res = flattenRowMajor(A);
        //subtract sigma on the trace
        for(int i=0; i<A.rows(); i++) {
            res[i*A.columns()+i] -= 1;
        }
        invert(A.rows(), res);
        LinearOperation OPinv = asLinearOperation(A.rows(), A.columns(), res);
        SymmetricArpackSolver solver = eigsh_cayley(asLinearOperation(A), null, OPinv, A.rows(), 3, "SM", 1, null, 100, 1e-5);
        Assertions.assertSame(5, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(eigenvalues, eigenvectors, new int[]{0, 1, 4}, d, z);
    }
}