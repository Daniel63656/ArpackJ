package net.scoreworks.arpackj;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import static net.scoreworks.arpackj.MatrixOperations.asLinearOperation;
import static net.scoreworks.arpackj.MatrixOperations.invert;

public class LinalgTests {
    private static final double epsilon = 0.0001;

    private static final Matrix M, M_inv;
    static {
        double[][] data = {
                {9, 2, 3, 4, 5},
                {2, 8, 1, 2, 3},
                {3, 1, 7, 0, 1},
                {4, 2, 0, 6, 2},
                {5, 3, 1, 2, 5}};
        M = new Basic2DMatrix(data);
        double[][] data_inv = {
                {0.4555, 0.1096, -0.1524, -0.2038, -0.4092},
                {0.1096, 0.1918, -0.0479, -0.0753, -0.1849},
                {-0.1524, -0.0479, 0.1995, 0.0813, 0.1087},
                {-0.2038, -0.0753, 0.0813, 0.2885, 0.1173},
                {-0.4092, -0.1849, 0.1087, 0.1173, 0.6515}};
        M_inv = new Basic2DMatrix(data_inv);
    }

    @Test
    public void testMatmul() {
        double[] x = new double[]{3, 2, 5, 3, 2};
        double[] y = new double[]{68, 39, 48, 38, 42};
        LinearOperation M_OP = asLinearOperation(M);
        double[] res = M_OP.apply(x, 0);

        for (int i=0; i<y.length; i++) {
            Assertions.assertEquals(y[i], res[i], epsilon);
        }
    }

    @Test
    public void testChainedMatmul() {
        double[] x = new double[]{-1, -1, -1, 3, 2, 5, 3, 2};
        double[] y = new double[]{1196, 698, 621, 662, 791};
        LinearOperation M_OP = asLinearOperation(M);
        double[] res = M_OP.apply(M_OP.apply(x, 3), 0);

        for (int i=0; i<y.length; i++) {
            Assertions.assertEquals(y[i], res[i], epsilon);
        }
    }

    @Test
    public void testInverse() {
        Matrix M_calc = invert(M);

        //check result
        for (int i=0; i<M.rows(); i++) {
            for (int j=0; j<M.columns(); j++) {
                Assertions.assertEquals(M_inv.get(i, j), M_calc.get(i, j), epsilon);
            }
        }
    }
}