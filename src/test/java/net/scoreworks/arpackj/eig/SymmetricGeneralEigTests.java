package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import net.scoreworks.arpackj.eig.SymmetricArpackSolver;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import static net.scoreworks.arpackj.MatrixOperations.asLinearOperation;
import static net.scoreworks.arpackj.MatrixOperations.invert;
import static net.scoreworks.arpackj.eig.EigenvalueDecomposition.*;

public class SymmetricGeneralEigTests {
    private static final double epsilon = 0.0001;

    /** Matrix to test on */
    private static final Matrix A, M;
    static {
        double[][] dataA = {
                {0, 9, 0, 1, 0},
                {9, 0, 4, 0, 0},
                {0, 4, 2, 8, 5},
                {1, 0, 8, 1, 0},
                {0, 0, 5, 0, 7}};
        A = new Basic2DMatrix(dataA);
        //M is positive semi-definite
        double[][] dataM = {
                {9, 2, 3, 4, 5},
                {2, 8, 1, 2, 3},
                {3, 1, 7, 0, 1},
                {4, 2, 0, 6, 2},
                {5, 3, 1, 2, 5}};
        M = new Basic2DMatrix(dataM);
    }
    private static final double[] eigenvalues = new double[]{-1.7666, -0.81, 0.7156, 1.5177, 9.1618};
    private static final double[] eigenvectors = {
            0.2710, 0.1510, -0.1433, -0.0356, -0.5809,
            -0.2126, -0.1638, -0.1861, 0.0495, -0.2875,
            0.1350, -0.2169, -0.0526, 0.2454, 0.2669,
            -0.1876, 0.2146, 0.1027, 0.3501, 0.2723,
            -0.0959, 0.0633, -0.1577, -0.2783, 0.7321};

    private static void checkSolution(double[] d, double[] v,int[] idx) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<idx.length; i++) {
            Assertions.assertEquals(eigenvalues[idx[i]], d[i], epsilon);
        }

        //check eigenvectors match up to sign flip
        for (int i=0; i< idx.length; i++) {
            for (int j=0; j<eigenvalues.length; j++) {
                Assertions.assertEquals(Math.abs(eigenvectors[idx[i]*5 + j]), Math.abs(v[i*5 + j]), epsilon);
            }
        }
    }

    @Test
    public void testGeneralEigenvalueProblemLargestMagnitude() {
        SymmetricArpackSolver solver = eigsh_general(A, 4, M, "LM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();
        checkSolution(d, v, new int[]{0, 1, 3, 4});
    }

    @Test
    public void testGeneralEigenvalueProblemSmallestMagnitude() {
        LinearOperation M_inv = asLinearOperation(invert(M));
        SymmetricArpackSolver solver = eigsh_general(asLinearOperation(A), A.rows(), 4, asLinearOperation(M), M_inv, "SM", null, 100, 1e-5);
        Assertions.assertSame(2, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();
        checkSolution(d, v, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertLM() {
        SymmetricArpackSolver solver = eigsh_shiftInvert(A, 4, M, "LM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();
        checkSolution(d, v, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testGeneralEigenvalueProblemShiftInvertSM() {
        // (A - sigma*M)^-1 = A^-1 because sigma=0
        LinearOperation OP_inv = asLinearOperation(invert(A));
        SymmetricArpackSolver solver = eigsh_shiftInvert(asLinearOperation(A), A.rows(), 4, OP_inv, "SM", 0, null, 100, 1e-5);
        Assertions.assertSame(3, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();
        checkSolution(d, v, new int[]{0, 1, 3, 4});
    }

    //TODO buckling mode

    @Test
    public void testGeneralEigenvalueProblemCayleyLM() {
        SymmetricArpackSolver solver = eigsh_cayley(A, 4, M, "LM", 0, null, 100, 1e-5);
        Assertions.assertSame(5, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();
        checkSolution(d, v, new int[]{0, 1, 2, 3});
    }

    @Test
    public void testGeneralEigenvalueProblemCayleySM() {
        // (A - sigma*I)^-1 = A^-1 because sigma=0
        LinearOperation OP_inv = asLinearOperation(invert(A));
        SymmetricArpackSolver solver = eigsh_cayley(asLinearOperation(A), A.rows(), 4, OP_inv, "SM", 0, null, 100, 1e-5);
        Assertions.assertSame(5, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();
        checkSolution(d, v, new int[]{0, 1, 3, 4});
    }
}