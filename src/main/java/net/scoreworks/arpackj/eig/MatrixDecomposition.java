package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.apache.commons.math3.complex.Complex;
import org.bytedeco.openblas.global.openblas;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import static net.scoreworks.arpackj.MatrixOperations.*;

public final class MatrixDecomposition {
    private MatrixDecomposition() {}  //make instantiation impossible


    /**
     * Performs LU decomposition on a given matrix
     * @param A Matrix to perform the decomposition on
     * @return matrix containing the lower triangular matrix (L) and the upper triangular matrix (U)
     */
    public static Matrix LU(Matrix A) {
        return Basic2DMatrix.from1DArray(A.rows(), A.columns(), LU(A.rows(), A.columns(), flattenColumnMajor(A))).transpose();
    }

    /**
     * Performs LU decomposition on a flattened matrix in column-major ordering
     * @param rows of the matrix
     * @param cols of the matrix
     * @param a flattened matrix. Expects column-major ordering
     * @return a matrix that contains the lower triangular matrix (L) and the upper triangular matrix (U) in
     * column major ordering
     */
    public static double[] LU(int rows, int cols, double[] a) {
        int[] m = new int[]{rows};
        int[] n = new int[]{cols};
        int[] lda = new int[]{cols};
        int[] ipiv = new int[lda[0]];
        int[] info = new int[1];
        openblas.LAPACK_dgetrf(m, n, a, lda, ipiv, info);
        return a;
    }

    /**
     * Solve the standard eigenvalue problem A*x = lambda*x for a square, symmetric matrix A
     * @param A square, symmetric {@link Matrix}
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh(Matrix A, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        return new SymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 1, which, ncv, 0, maxIter, tolerance, null, null);
    }

    /**
     * Solve the standard eigenvalue problem A*x = lambda*x for a square, symmetric matrix A
     * @param A Left multiplication by square, symmetric matrix A
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh(LinearOperation A, int n, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(A, n, nev, 1, which, ncv, 0, maxIter, tolerance, null, null);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square, symmetric matrix A
     * @param A square, symmetric {@link Matrix}
     * @param M square, symmetric and positive-definite {@link Matrix}. Must have same dimensions as A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh(Matrix A, Matrix M, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        if (M.rows() != A.rows() || M.columns() != A.columns())
            throw new IllegalArgumentException("M must have same dimensions as A");
        //calculate inverse of M
        Matrix M_inv = invert(M);
        return new SymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 2, which, ncv, 0, maxIter, tolerance, asLinearOperation(M), asLinearOperation(M_inv));
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square, symmetric matrix A
     * @param A Left multiplication by square, symmetric matrix A
     * @param M Left multiplication by square, symmetric and positive-definite matrix M with same dimensions as A
     * @param M_inv Left multiplication by inverse of M
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh(LinearOperation A, LinearOperation M, LinearOperation M_inv, int n, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(A, n, nev, 2, which, ncv, 0, maxIter, tolerance, M, M_inv);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square, symmetric matrix A in shift-invert mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A square, symmetric {@link Matrix}
     * @param M square, symmetric and positive semi-definite {@link Matrix}. Must have same dimensions as A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_shiftInvert(Matrix A, Matrix M, int nev, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        if (M == null)
            return new SymmetricArpackSolver(null, A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, null, getOP_inv(A, null, sigma, 0));
        else
            return new SymmetricArpackSolver(null, A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, asLinearOperation(M), getOP_inv(A, M, sigma, 0));
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square, symmetric matrix A in shift-invert mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param M Left multiplication by square, symmetric and positive semi-definite matrix M with same dimensions as A
     * @param OP_inv Linear operation representing left multiplication by (A - sigma*M)^-1
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_shiftInvert(LinearOperation M, LinearOperation OP_inv, int n, int nev, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(null, n, nev, 3, which, ncv, sigma, maxIter, tolerance, M, OP_inv);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square, symmetric and positive-definite matrix A in buckling mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A square, symmetric and positive-definite {@link Matrix}
     * @param M square, symmetric indefinite {@link Matrix}. Must have same dimensions as A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_buckling(Matrix A, Matrix M, int nev, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 4, which, ncv, sigma, maxIter, tolerance, null, getOP_inv(A, M, sigma, 0));
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square, symmetric and positive-definite matrix A in buckling mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A Left multiplication by square, symmetric and positive-definite matrix A
     * @param OP_inv Left multiplication by (A -sigma*M)^-1, where M is square and symmetric indefinite
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_buckling(LinearOperation A, LinearOperation OP_inv, int n, int nev, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(A, n, nev, 4, which, ncv, sigma, maxIter, tolerance, null, OP_inv);
    }



    /**
     * Solve the standard eigenvalue problem A*x = lambda*x for a square matrix A
     * @param A square {@link Matrix}
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigs(Matrix A, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        return new UnsymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 1, which, ncv, null, maxIter, tolerance, null, null);
    }

    /**
     * Solve the standard eigenvalue problem A*x = lambda*x for a square matrix A
     * @param A Left multiplication by square matrix A
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigs(LinearOperation A, int n, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        return new UnsymmetricArpackSolver(A, n, nev, 1, which, ncv, null, maxIter, tolerance, null, null);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square matrix A
     * @param A square {@link Matrix}
     * @param M square, symmetric and positive-definite {@link Matrix}. Must have same dimensions as A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigs(Matrix A, Matrix M, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        if (M.rows() != A.rows() || M.columns() != A.columns())
            throw new IllegalArgumentException("M must have same dimensions as A");
        //calculate inverse of M
        Matrix M_inv = invert(M);
        return new UnsymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 2, which, ncv, null, maxIter, tolerance, asLinearOperation(M), asLinearOperation(M_inv));
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square matrix A
     * @param A Left multiplication by square matrix A
     * @param M Left multiplication by square, symmetric and positive-definite matrix M with same dimensions as A
     * @param M_inv Left multiplication by inverse of M
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigs(LinearOperation A, LinearOperation M, LinearOperation M_inv, int n, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        return new UnsymmetricArpackSolver(A, n, nev, 2, which, ncv, null, maxIter, tolerance, M, M_inv);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square matrix A in shift-invert xxx mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A square {@link Matrix}
     * @param M square, symmetric and positive semi-definite {@link Matrix}. Must have same dimensions as A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma complex shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigs_shiftInvertReal(Matrix A, Matrix M, int nev, String which, Complex sigma, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        return new UnsymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, null, getOP_inv(A, M, sigma.getReal(), sigma.getImaginary()));
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square matrix A in shift-invert xxx mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A Left multiplication by square A
     * @param M Left multiplication by square and positive semi-definite matrix M with same dimensions as A
     * @param OP_inv Linear operation representing left multiplication by real((A - sigma*M)^-1)
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma complex shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigsh_shiftInvertReal(LinearOperation A, LinearOperation M, LinearOperation OP_inv, int n, int nev, String which, Complex sigma, Integer ncv, int maxIter, double tolerance) {
        return new UnsymmetricArpackSolver(A, n, nev, 3, which, ncv, sigma, maxIter, tolerance, M, OP_inv);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square matrix A in shift-invert xxx mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A square {@link Matrix}
     * @param M square, symmetric and positive semi-definite {@link Matrix}. Must have same dimensions as A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma complex shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigs_shiftInvertImag(Matrix A, Matrix M, int nev, String which, Complex sigma, Integer ncv, int maxIter, double tolerance) {
        if (sigma.getImaginary() == 0)
            throw new IllegalArgumentException("sigma must be complex in this mode");
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        if (sigma.getReal() == 0 && sigma.getImaginary() == 0) {
            return new UnsymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 4, which, ncv, sigma, maxIter, tolerance, null, asLinearOperation(invert(A)));
        }

        Complex z;
        //deepcopy and convert to dense for later invert
        double[] res = flattenRowMajor(A);
        if (M == null) {    //calculate real((A - sigma*I)^-1)
            //subtract sigma on the trace and take real part
            for(int i=0; i<A.rows(); i++) {
                z = new Complex(res[i*A.columns()+i], 0).subtract(sigma);
                res[i*A.columns()+i] = z.getImaginary();
            }
        }
        else {      //calculate real((A - sigma*M)^-1)
            //can't use la4j's operations as it does not support complex type
            for(int i=0; i<A.rows(); i++) {
                for(int j=0; i<A.columns(); j++) {
                    z = sigma.multiply(M.get(i, j));
                    z = new Complex(res[i*A.columns()+i], 0).subtract(z);
                    res[i*A.columns()+i] = z.getImaginary();
                }
            }
        }
        invert(A.rows(), res);
        LinearOperation OPinv = asLinearOperation(A.rows(), A.columns(), res);
        return new UnsymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 4, which, ncv, sigma, maxIter, tolerance, null, OPinv);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x for a square matrix A in shift-invert xxx mode to find eigenvalues near sigma.
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A Left multiplication by square A
     * @param M Left multiplication by square and positive semi-definite matrix M with same dimensions as A
     * @param OP_inv Linear operation representing left multiplication by imag((A - sigma*M)^-1)
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma complex shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static UnsymmetricArpackSolver eigsh_shiftInvertImag(LinearOperation A, LinearOperation M, LinearOperation OP_inv, int n, int nev, String which, Complex sigma, Integer ncv, int maxIter, double tolerance) {
        if (sigma.getImaginary() == 0)
            throw new IllegalArgumentException("sigma must be complex in this mode");
        return new UnsymmetricArpackSolver(A, n, nev, 4, which, ncv, sigma, maxIter, tolerance, M, OP_inv);
    }


    /**
     * @return (A -sigma*M)^-1
     */
    private static LinearOperation getOP_inv(Matrix A, Matrix M, double sigma_r, double sigma_i) {
        if (sigma_r == 0 && sigma_i == 0) {
            return asLinearOperation(invert(A));
        }
        //sigma is real number
        if (sigma_i == 0) {
            if (M == null) {     //calculate (A - sigma*I)^-1
                //deepcopy and convert to dense for later invert
                double[] res = flattenRowMajor(A);
                //subtract sigma on the trace
                for(int i=0; i<A.rows(); i++) {
                    res[i*A.columns()+i] -= sigma_r;
                }
                invert(A.rows(), res);
                return asLinearOperation(A.rows(), A.columns(), res);
            }
            return asLinearOperation(invert(A.subtract(M.multiply(sigma_r))));
        }

        //sigma is complex
        if (M == null) {    //calculate real((A - sigma*I)^-1)
            //Option 1 TODO why does this work?
            /*Complex z;
            //deepcopy and convert to dense for later invert
            double[] res = flattenRowMajor(A);
            //subtract sigma on the trace and take real part
            for(int i=0; i<A.rows(); i++) {
                z = new Complex(res[i*A.columns()+i], 0).subtract(sigma);
                res[i*A.columns()+i] = z.getReal();
            }
            invert(A.rows(), res);
            return asLinearOperation(A.rows(), A.columns(), res);*/



            //create complex matrix Z=A-sigma*I
            double[] Z = new double[2*A.rows()*A.columns()];
            int cols = A.columns();
            for (int i=0; i<A.rows(); i++) {
                for (int j=0; j<A.columns(); j++) {
                    Z[2*(i*cols + j)] = A.get(i, j);
                }
                //subtract sigma on trace
                Z[2*(i*cols + i)] -= sigma_r;
                Z[2*(i*cols + i) + 1] = -sigma_i;
            }
            invertComplex(cols, Z);
            return asLinearOperationReal(A.rows(), A.columns(), Z);
        }




        return null;
    }
}