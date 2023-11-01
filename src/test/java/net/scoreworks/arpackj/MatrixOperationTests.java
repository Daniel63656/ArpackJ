package net.scoreworks.arpackj;

import net.scoreworks.arpackj.eig.MatrixDecomposition;
import org.bytedeco.openblas.global.openblas;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import java.util.Arrays;

import static net.scoreworks.arpackj.MatrixOperations.*;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.LU;

public class MatrixOperationTests {
    private static final double epsilon = 0.0001;
    private static final Matrix B, B_inv, B_LU;
    static {
        double[][] dataB = {    //non-symmetric
                {9, 1, 9, 1, 5},
                {6, 8, 1, 2, 3},
                {3, 1, 7, 5, 1},
                {4, 9, 0, 6, 2},
                {5, 3, 1, 2, 5}};
        B = new Basic2DMatrix(dataB);
        double[][] data_inv = {
                {1.22033898, -1.50847458, -1.25423729, 1.57627119, -0.69491525},
                {-0.60290557, 0.92978208, 0.60774818, -0.79661017, 0.24213075},
                {-0.58716707, 0.89346247, 0.73244552, -0.89830508, 0.26392252},
                {0.38983051, -0.74576271, -0.37288136, 0.71186441, -0.15254237},
                {-0.89709443, 1.07021792, 0.89225182, -1.20338983, 0.75786925}};
        B_inv = new Basic2DMatrix(data_inv);
        double[][] data_LU = {
                {9.0, 1.0, 9.0, 1.0, 5.0},
                {0.44444444, 8.55555556, -4.0, 5.55555556, -0.22222222},
                {0.33333333, 0.07792208, 4.31168831, 4.23376623, -0.64935065},
                {0.55555556, 0.28571429, -0.6626506, 2.6626506, 1.85542169},
                {0.66666667, 0.85714286, -0.36445783, -0.7081448, 0.93438914}};
        B_LU = new Basic2DMatrix(data_LU);
    }

    @Test
    public void testFlattenRowMajor() {
        double[] res = flattenRowMajor(B);
        for (int i = 0; i< B.rows(); i++) {
            for (int j = 0; j< B.columns(); j++) {
                Assertions.assertEquals(B.get(i, j), res[i* B.columns() + j], epsilon);
            }
        }
    }

    @Test
    public void testFlattenColumnMajor() {
        double[] res = flattenColumnMajor(B);
        for (int i = 0; i< B.rows(); i++) {
            for (int j = 0; j< B.columns(); j++) {
                Assertions.assertEquals(B.get(i, j), res[j* B.rows() + i], epsilon);
            }
        }
    }

    @Test
    public void testMatmul() {
        double[] x = new double[]{3, 2, 5, 3, 2};
        double[] y = new double[]{87, 51, 63, 52, 42};
        LinearOperation M_OP = asLinearOperation(B);
        double[] res = M_OP.apply(x, 0);

        for (int i=0; i<y.length; i++) {
            Assertions.assertEquals(y[i], res[i], epsilon);
        }
    }

    @Test
    public void testChainedMatmul() {
        double[] x = new double[]{-1, -1, -1, 3, 2, 5, 3, 2};
        double[] y = new double[]{1663, 1223, 1055, 1203, 965};
        LinearOperation M_OP = asLinearOperation(B);
        double[] res = M_OP.apply(M_OP.apply(x, 3), 0);

        for (int i=0; i<y.length; i++) {
            Assertions.assertEquals(y[i], res[i], epsilon);
        }
    }

    @Test
    public void testLU_Decomposition() {
        Matrix res = MatrixDecomposition.LU(B);
        //check result
        for (int i = 0; i< B.rows(); i++) {
            for (int j = 0; j< B.columns(); j++) {
                Assertions.assertEquals(B_LU.get(i, j), res.get(i, j), epsilon);
            }
        }
    }

    @Test
    public void testInverse() {
        Matrix inverse = invert(B);
        //check result
        for (int i = 0; i< B.rows(); i++) {
            for (int j = 0; j< B.columns(); j++) {
                Assertions.assertEquals(B_inv.get(i, j), inverse.get(i, j), epsilon);
            }
        }
    }

    @Test
    public void testComplexInverse() {
        //3x3 complex matrix (real, imaginary as 2 consecutive double values)
        double[] Z = new double[]{
                1,2, 3,4, 5,6,
                1,0, 4,3, 5,6,
                5,6, 1,1, 1,1};
        double[] Z_inv = {
                0.03161398, 0.04825291, -0.04459235, -0.02595674, 0.08252912, -0.11613977,
                -0.57986689, -0.51663894, 0.57054908, 0.48136439, 0.03361065, 0.19866889,
                0.43011647, 0.23544093, -0.33826955, -0.31630616, -0.04559068, -0.10116473
        };
        invertComplex(3, Z);
        for (int i=0; i<9; i++) {
            Assertions.assertEquals(Z_inv[i], Z[i], epsilon);
        }
    }
}