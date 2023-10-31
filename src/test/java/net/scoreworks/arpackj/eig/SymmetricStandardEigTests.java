package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import static net.scoreworks.arpackj.MatrixOperations.*;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.*;

public class SymmetricStandardEigTests {
    private static final double epsilon = 0.0001;

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
    public void testStandardEigenvalueProblemLM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A, 4, "LM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 3, 4});
    }

    @Test
    public void testStandardEigenvalueProblemLM_CRS() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(A_sparse, 4, "LM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 3, 4});
    }

    @Test
    public void testStandardEigenvalueProblemSM() {
        SymmetricArpackSolver solver = MatrixDecomposition.eigsh(asLinearOperation(A), A.rows(), 4, "SM", null, 100, 1e-5);
        Assertions.assertSame(1, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertLM() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A, 4, null, "LM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertLM_CRS() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A_sparse, 4, null, "LM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvertSM() {
        // (A - sigma*I)^-1 = A^-1 because sigma=0
        LinearOperation OP_inv = asLinearOperation(invert(A));
        SymmetricArpackSolver solver = eigsh_shiftInvert(A.rows(), 4, null, OP_inv, "SM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] z = solver.getEigenvectors();
        checkSolution(d, z, new int[]{0, 1, 3, 4});
    }
}