package net.scoreworks.arpackj;

import net.scoreworks.arpackj.eig.SymmetricArpackSolver;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import static net.scoreworks.arpackj.MatrixOperations.invert;
import static net.scoreworks.arpackj.eig.EigenvalueDecomposition.eigsh_shiftInvert;
import static net.scoreworks.arpackj.eig.EigenvalueDecomposition.eigsh_standard;

public class EigenvalueDecompositionTests {
    private static final double epsilon = 0.0001;

    /** Matrix to test on */
    private static final Matrix A;
    static {
        A = new Basic2DMatrix(5, 5);
        A.set(0, 1, 9);
        A.set(1, 0, 9);
        A.set(0, 3, 1);
        A.set(3, 0, 1);
        A.set(1, 2, 4);
        A.set(2, 1, 4);
        A.set(2, 2, 2);
        A.set(2, 3, 8);
        A.set(3, 2, 8);
        A.set(2, 4, 5);
        A.set(4, 2, 5);
        A.set(3, 3, 1);
        A.set(4, 4, 7);
    }
    private static final double[] eigenvalues = new double[]{-10.9023, -5.73301, 4.66164, 8.42532, 13.5483};
    private static final double[] eigenvectors = new double[]{
            0.53365, -0.60749, 0.455019, -0.350674, -0.12708,
            -0.4676, 0.36081, 0.53506, -0.56629, -0.21011,
            0.23612, 0.190267, -0.30952, -0.61175, 0.66182,
            0.60465, 0.57489, -0.14954, -0.07968, -0.52459,
            0.27418, 0.36616, 0.62330, 0.41923, 0.47592};

    @Test
    public void testStandardEigenvalueProblemLM() {
        int[] idx = new int[]{0, 1, 3, 4};
        SymmetricArpackSolver solver = eigsh_standard(A, 4, "LM", null, 100, 1e-5);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();

        //check eigenvectors match (they are returned in ascending order)
        Assertions.assertEquals(d[0], eigenvalues[idx[0]], epsilon);
        Assertions.assertEquals(d[1], eigenvalues[idx[1]], epsilon);
        Assertions.assertEquals(d[2], eigenvalues[idx[2]], epsilon);
        Assertions.assertEquals(d[3], eigenvalues[idx[3]], epsilon);

        //check eigenvectors match
        for (int i=0; i<4; i++) {
            for (int j=0; j<5; j++) {
                Assertions.assertEquals(v[i*5 + j], eigenvectors[idx[i]*5 + j], epsilon);
            }
        }
    }

    @Test
    public void testStandardEigenvalueProblemSM() {
        int[] idx = new int[]{0, 1, 2, 3};
        SymmetricArpackSolver solver = eigsh_standard(A, 4, "SM", null, 100, 1e-5);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();

        //check eigenvectors match (they are returned in ascending order)
        Assertions.assertEquals(d[0], eigenvalues[idx[0]], epsilon);
        Assertions.assertEquals(d[1], eigenvalues[idx[1]], epsilon);
        Assertions.assertEquals(d[2], eigenvalues[idx[2]], epsilon);
        Assertions.assertEquals(d[3], eigenvalues[idx[3]], epsilon);

        //check eigenvectors match
        for (int i=0; i<4; i++) {
            for (int j=0; j<5; j++) {
                Assertions.assertEquals(v[i*5 + j], eigenvectors[idx[i]*5 + j], epsilon);
            }
        }
    }

    @Test
    public void testStandardEigenvalueProblemShiftInvert() {
        int[] idx = new int[]{0, 1, 2, 3};
        SymmetricArpackSolver solver = eigsh_shiftInvert(A, 4, "LM", 0, null, 100, 1e-5);
        solver.solve();
        double[] d = solver.getEigenvalues();
        double[] v = solver.getEigenvectors();

        //check eigenvectors match (they are returned in ascending order)
        Assertions.assertEquals(d[0], eigenvalues[idx[0]], epsilon);
        Assertions.assertEquals(d[1], eigenvalues[idx[1]], epsilon);
        Assertions.assertEquals(d[2], eigenvalues[idx[2]], epsilon);
        Assertions.assertEquals(d[3], eigenvalues[idx[3]], epsilon);

        //check eigenvectors match
        for (int i=0; i<4; i++) {
            for (int j=0; j<5; j++) {
                Assertions.assertEquals(v[i*5 + j], eigenvectors[idx[i]*5 + j], epsilon);
            }
        }
    }

    @Test
    public void testInverse() {
        invert(A);
    }
}