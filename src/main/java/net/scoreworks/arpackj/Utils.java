package net.scoreworks.arpackj;

import org.bytedeco.openblas.global.openblas;
import org.la4j.Matrix;
import org.la4j.iterator.MatrixIterator;
import org.la4j.matrix.SparseMatrix;
import org.la4j.matrix.dense.Basic2DMatrix;

/**
 * Helper class for various general functions and operations
 */
public final class Utils {
    private Utils() {}  //make instantiation impossible


    //========== Linear Operations ==========//
    public static final LinearOperation IDENTITY = (x, off) -> x;

    /**
     * @return a given matrix as a {@link LinearOperation}. Takes sparsity into account.
     */
    public static LinearOperation asLinearOperation(Matrix A) {
        double[] result = new double[A.columns()];

        if (A instanceof SparseMatrix a) {
            return (b, off) -> {
                MatrixIterator it = a.nonZeroIterator();
                double x;
                int i, j;
                while (it.hasNext()) {
                    x = it.next();
                    i = it.rowIndex();
                    j = it.columnIndex();
                    result[i] = result[i] + (x * b[j + off]);
                }
                return result;
            };
        }
        else return (b, off) -> {
            double acc;
            for (int i=0; i<A.rows(); i++) {
                acc = 0.0;
                for (int j=0; j<A.columns(); j++) {
                    acc += A.get(i, j) * b[j + off];
                }
                result[i] = acc;
            }
            return result;
        };
    }

    /**
     * @return a flattened matrix as a {@link LinearOperation}. Dense by definition.
     */
    public static LinearOperation asLinearOperation(int rows, int cols, double[] a) {
        double[] result = new double[cols];
        return (b, off) -> {
            double acc;
            for (int i=0; i<rows; i++) {
                acc = 0.0;
                for (int j = 0; j < cols; j++) {
                    acc += a[j*cols + i] * b[j + off];
                }
                result[i] = acc;
            }
            return result;
        };
    }


    //========== Matrix inversions ==========//

    /**
     * @param A the Matrix to be inverted
     * @return the inverse of A using openBLAS routines. Dense because inverses of sparse matrices are dense in general
     */
    public static Matrix invert(Matrix A) {
        return invert(A.rows(), A.columns(), flattenMatrix(A));
    }

    /**
     *
     * @param rows rows of the Matrix
     * @param cols columns of the Matrix
     * @param a the Matrix flattened to a 1d array
     * @return the inverse of A using openBLAS routines. Dense because inverses of sparse matrices are dense in general.
     * The result is directly written to the provided array
     */
    public static Matrix invert(int rows, int cols, double[] a) {
        int[] m = new int[]{rows};
        int[] n = new int[]{cols};
        int[] lda = new int[]{cols};
        int[] ipiv = new int[lda[0]];
        int[] info = new int[1];
        int[] lwork = new int[]{n[0]*n[0]};
        double[] work = new double[lwork[0]];

        //LU decomposition
        openblas.LAPACK_dgetrf(m, n, a, lda, ipiv, info);
        //calculate inverse of A: A^-1 = L^-1*U^-1
        openblas.LAPACK_dgetri(n, a, lda, ipiv, work, lwork, info);

        return Basic2DMatrix.from1DArray(rows, cols, a);
    }


    //========== Other ==========//

    /**
     * @return a given matrix A as a flattened double array. This array is a deep copy, any changes made to it do not
     * affect the original matrix.
     */
    public static double[] flattenMatrix(Matrix A) {
        double[] result = new double[A.rows() * A.columns()];

        if (A instanceof SparseMatrix a) {
            MatrixIterator it = a.nonZeroIterator();
            double x;
            int i, j;
            while (it.hasNext()) {
                x = it.next();
                i = it.rowIndex();
                j = it.columnIndex();
                result[j*A.columns() + i] = x;
            }

        }
        else {
            for (int i=0; i<A.rows(); i++) {
                for (int j = 0; j < A.columns(); j++) {
                    result[j*A.columns() + i] = A.get(i, j);
                }
            }
        }
        return result;
    }
}