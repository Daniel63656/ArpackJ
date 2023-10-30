package net.scoreworks.arpackj.eig;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import static net.scoreworks.arpackj.MatrixOperations.asLinearOperation;
import static net.scoreworks.arpackj.MatrixOperations.invert;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.eigsh_buckling;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.eigsh_shiftInvert;

public class BucklingModeTests {
    private static final double epsilon = 0.0001;
    private static final Matrix A, A_sparse, M, M_sparse;
    static {
        double[][] dataA = {
                {10, 1, 0, 1, 0},
                {1, 10, 1, 0, 1},
                {0, 1, 10, 1, 0},
                {1, 0, 1, 10, 1},
                {0, 1, 0, 1, 10}};
        A = new Basic2DMatrix(dataA);   //A must be positive-definite
        A_sparse = CRSMatrix.from2DArray(dataA);
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

    //solutions for eigenvectors are unstable so a solution per test must be provided
    private static void checkSolution(double[] d_sol, double[] d, double[] v, double[] v_sol) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<d.length; i++) {
            Assertions.assertEquals(d_sol[i], d[i], epsilon);
        }

        //check eigenvectors match up to sign flip
        for (int i=0; i< d.length; i++) {
            for (int j=0; j<d.length; j++) {
                Assertions.assertEquals(Math.abs(v_sol[i*5 + j]), Math.abs(v[i*5 + j]), epsilon);
            }
        }
    }
    @Test
    public void testGeneralEigenvalueProblemBucklingLM() {
        SymmetricArpackSolver solver = eigsh_buckling(A, 4, M, "LM", 10, null, 100, 1e-5);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();

        checkSolution(new double[]{1.15866927, 1.61125964, 2.73369412, 11.41708084}, d, new double[]{
                -0.08597825, 0.22406474, -0.21323371, 0.13672978, -0.00243987,
                -0.09578178, 0.20635749, 0.16740696, -0.13887269, -0.00999573,
                0.0552968, -0.01050284, -0.12802326, -0.22646812, 0.18577403,
                -0.18492006, -0.07327136, 0.06716367, 0.08952588, 0.21813691}, v);
    }

    @Test
    public void testGeneralEigenvalueProblemBucklingLM_CRS() {
        SymmetricArpackSolver solver = eigsh_buckling(A_sparse, 4, M_sparse, "LM", 10, null, 100, 1e-5);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();

        checkSolution(new double[]{1.15866927, 1.61125964, 2.73369412, 11.41708084}, d, new double[]{
                -0.08597825, 0.22406474, -0.21323371, 0.13672978, -0.00243987,
                -0.09578178, 0.20635749, 0.16740696, -0.13887269, -0.00999573,
                0.0552968, -0.01050284, -0.12802326, -0.22646812, 0.18577403,
                -0.18492006, -0.07327136, 0.06716367, 0.08952588, 0.21813691}, v);
    }

    @Test
    public void testGeneralEigenvalueProblemBucklingSM() {
        SymmetricArpackSolver solver = eigsh_buckling(A, 4, M, "SM", 10, null, 100, 1e-5);
        Assertions.assertSame(4, solver.mode);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();

        checkSolution(new double[]{0.6905975, 1.15866927, 1.61125964, 2.73369412}, d, new double[]{
                -0.21979101, -0.07017172, -0.08793773, -0.07686443, -0.14115981,
                -0.08597825, 0.22406474, -0.21323371, 0.13672978, -0.00243987,
                -0.09578178, 0.20635749, 0.16740696, -0.13887269, -0.00999573,
                0.0552968, -0.01050284, -0.12802326, -0.22646812, 0.18577403}, v);
    }
}