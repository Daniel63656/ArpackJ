package net.scoreworks.arpackj;

import org.apache.commons.math3.complex.Complex;
import org.bytedeco.openblas.global.openblas;
import org.la4j.Matrix;
import org.la4j.iterator.MatrixIterator;
import org.la4j.matrix.SparseMatrix;
import org.la4j.matrix.dense.Basic2DMatrix;

/**
 * Helper class for various general functions and operations regarding matrices
 */
public final class MatrixOperations {
    private MatrixOperations() {}  //make instantiation impossible


    //========== Linear Operations ==========//
    /**
     * Implementation of the identity operation
     */
    public static final LinearOperation IDENTITY = (x, off) -> x;

    //TODO replace by openBLAS
    /**
     * @return a given {@link Matrix} as a {@link LinearOperation}. Takes sparsity into account in case that matrix is sparse
     */
    public static LinearOperation asLinearOperation(Matrix A) {
        if (A instanceof SparseMatrix a) {
            return (b, off) -> {
                double[] result = new double[A.columns()];
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
            double[] result = new double[A.columns()];
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
     * @return a flattened matrix in row-major ordering as a {@link LinearOperation}. Dense by definition.
     */
    public static LinearOperation asLinearOperation(int rows, int cols, double[] a) {
        return (b, off) -> {
            double[] result = new double[cols];
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
     * @param A square matrix to calculate inverse of
     * @return the inverse of A using openBLAS routines. Result is dense because inverses of sparse matrices are
     * dense in general
     */
    public static Matrix invert(Matrix A) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A must be square in order to be invertible");
        double[] res = flattenRowMajor(A);
        invert(A.rows(), res);
        return Basic2DMatrix.from1DArray(A.rows(), A.rows(), res);
    }

    /**
     * Calculate the inverse of A using openBLAS routines. Result is dense because inverses of sparse matrices are
     * dense in general. The result is written to the provided array directly
     * @param rows dimensions of the square matrix
     * @param a the Matrix flattened to a 1d array. Indifferent to ordering (row-major, column-major)
     */
    public static void invert(int rows, double[] a) {
        int[] m = new int[]{rows};
        int[] n = new int[]{rows};
        int[] lda = new int[]{rows};
        int[] ipiv = new int[lda[0]];
        int[] info = new int[1];
        int[] lwork = new int[]{n[0]*n[0]};
        double[] work = new double[lwork[0]];

        //LU decomposition
        openblas.LAPACK_dgetrf(m, n, a, lda, ipiv, info);
        //calculate inverse of A: A^-1 = L^-1*U^-1
        openblas.LAPACK_dgetri(n, a, lda, ipiv, work, lwork, info);
    }

    public static void invertComplex(int rows, double[] a) {
        int[] m = new int[]{rows};
        int[] n = new int[]{rows};
        int[] lda = new int[]{rows};
        int[] ipiv = new int[lda[0]];
        int[] info = new int[1];
        int[] lwork = new int[]{n[0]*n[0]};
        double[] work = new double[lwork[0]];

        // LU decomposition
        openblas.LAPACK_zgetrf(m, n, a, lda, ipiv, info);
        // Calculate the inverse of A: A^-1 = L^-1 * U^-1
        openblas.LAPACK_zgetri(n, a, lda, ipiv, work, lwork, info);
    }


    //========== Other ==========//

    /**
     * @return a given {@link Matrix} A as a flattened, row-major double array. This array is a deep copy, any changes made to it do not
     * affect the original matrix.
     */
    public static double[] flattenRowMajor(Matrix A) {
        double[] result = new double[A.rows() * A.columns()];

        if (A instanceof SparseMatrix a) {
            MatrixIterator it = a.nonZeroIterator();
            double x;
            int i, j;
            while (it.hasNext()) {
                x = it.next();
                i = it.rowIndex();
                j = it.columnIndex();
                result[i*A.columns() + j] = x;
            }
        }
        else {
            for (int i=0; i<A.rows(); i++) {
                for (int j = 0; j<A.columns(); j++) {
                    result[i*A.columns() + j] = A.get(i, j);
                }
            }
        }
        return result;
    }

    /**
     * @return a given {@link Matrix} A as a flattened, column-major double array. This array is a deep copy, any changes made to it do not
     * affect the original matrix.
     */
    public static double[] flattenColumnMajor(Matrix A) {
        double[] result = new double[A.columns() * A.rows()];

        if (A instanceof SparseMatrix a) {
            MatrixIterator it = a.nonZeroIterator();
            double x;
            int i, j;
            while (it.hasNext()) {
                x = it.next();
                i = it.rowIndex();
                j = it.columnIndex();
                result[j*A.rows() + i] = x;
            }
        }
        else {
            for (int i=0; i<A.rows(); i++) {
                for (int j = 0; j < A.columns(); j++) {
                    result[j*A.rows() + i] = A.get(i, j);
                }
            }
        }
        return result;
    }
}