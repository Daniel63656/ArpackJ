package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.la4j.LinearAlgebra;
import org.la4j.Matrix;
import org.la4j.inversion.MatrixInverter;

import java.util.HashSet;
import java.util.Set;

import static net.scoreworks.arpackj.MatrixOperations.*;

public class EigenvalueDecomposition {
    private static final Set<String> SEUPD_WHICH = new HashSet<>();
    private static final Set<String> NEUPD_WHICH = new HashSet<>();
    static {
        SEUPD_WHICH.add("LM");
        SEUPD_WHICH.add("SM");
        SEUPD_WHICH.add("LA");
        SEUPD_WHICH.add("SA");
        SEUPD_WHICH.add("BE");

        NEUPD_WHICH.add("LM");
        NEUPD_WHICH.add("SM");
        NEUPD_WHICH.add("LR");
        NEUPD_WHICH.add("SR");
        NEUPD_WHICH.add("LI");
        NEUPD_WHICH.add("SI");
    }

    /**
     * Solve the standard eigenvalue problem A*x = lambda*x for a square, symmetric matrix A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_standard(Matrix A, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        return eigsh(asLinearOperation(A), A.rows(), nev, 1, which, ncv, 0, maxIter, tolerance, null, null);
    }

    /**
     * Solve the standard eigenvalue problem A*x = lambda*x for a square, symmetric matrix A
     * @param A Linear operation representing left multiplication by A
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_standard(LinearOperation A, int n, int nev, String which, Integer ncv, int maxIter, double tolerance) {
        return eigsh(A, n, nev, 1, which, ncv, 0, maxIter, tolerance, null, null);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x. A and M must be square, symmetric and match in dimensions
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_general(Matrix A, int nev, Matrix M, String which, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        if (M.rows() != A.rows() || M.columns() != A.columns())
            throw new IllegalArgumentException("M must have same dimensions as A");

        //calculate inverse of M
        Matrix M_inv = invert(M);
        return eigsh(asLinearOperation(A), A.rows(), nev, 2, which, ncv, 0, maxIter, tolerance, asLinearOperation(M), asLinearOperation(M_inv));
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x. A and M must be square, symmetric and match in dimensions
     * @param A Linear operation representing left multiplication by A
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param M Linear operation representing left multiplication by M
     * @param M_inv Linear operation representing left multiplication by inverse of M
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_general(LinearOperation A, int n, int nev, LinearOperation M, LinearOperation M_inv, String which, Integer ncv, int maxIter, double tolerance) {
        return eigsh(A, n, nev, 2, which, ncv, 0, maxIter, tolerance, M, M_inv);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x in shift-invert mode to find eigenvalues near sigma.
     * A and M must be square, symmetric and match in dimensions. M is positive semi-definite
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_shiftInvert(Matrix A, int nev, Matrix M, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");

        if (M == null) {
            //calculate (A - sigma*I)^-1
            double[] res = flattenMatrix(A);
            for(int i=0; i<A.rows(); i++) {
                res[i*A.columns()+i] -= sigma;
            }
            Matrix res_inv = invert(A.rows(), A.columns(), res);
            return eigsh(asLinearOperation(A), A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, null, asLinearOperation(res_inv));
        }
        else {
            if (M.rows() != A.rows() || M.columns() != A.columns())
                throw new IllegalArgumentException("M must have same dimensions as A");

            //calculate (A - sigma*M)^-1
            Matrix res = A.subtract(M.multiply(sigma));
            Matrix res_inv = invert(res);
            return eigsh(asLinearOperation(A), A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, asLinearOperation(M), asLinearOperation(res_inv));
        }
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x in shift-invert mode to find eigenvalues near sigma.
     * A and M must be square, symmetric and match in dimensions. M is positive semi-definite
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A Linear operation representing left multiplication by A
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param OP_inv Linear operation representing left multiplication by (A -sigma*M)^-1
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_shiftInvert(LinearOperation A, int n, int nev, LinearOperation OP_inv, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return eigsh(A, n, nev, 3, which, ncv, sigma, maxIter, tolerance, null, OP_inv);
    }



    /*public static SymmetricArpackSolver eigsh_buckling(Matrix A, int nev, Matrix M, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        //calculate (A -sigma*M)^-1
        Matrix res = A.subtract(M.multiply(sigma));
        Matrix res_inv = invert(res);
        return eigsh(asLinearOperation(A), A.rows(), nev, 4, which, ncv, sigma, maxIter, tolerance, null, asLinearOperation(res_inv));
    }

    public static SymmetricArpackSolver eigsh_buckling(Matrix A, int nev, LinearOperation OP_inv, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return eigsh(asLinearOperation(A), A.rows(), nev, 4, which, ncv, sigma, maxIter, tolerance, null, OP_inv);
    }*/

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x in Cayley-transformed mode to find eigenvalues near sigma.
     * A and M must be square, symmetric and match in dimensions. M is positive semi-definite
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_cayley(Matrix A, int nev, Matrix M, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");

        if (M == null) {
            //calculate (A - sigma*I)^-1
            double[] res = flattenMatrix(A);
            for(int i=0; i<A.rows(); i++) {
                res[i*A.columns()+i] -= sigma;
            }
            Matrix res_inv = invert(A.rows(), A.columns(), res);
            return eigsh(asLinearOperation(A), A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, null, asLinearOperation(res_inv));
        }
        else {
            if (M.rows() != A.rows() || M.columns() != A.columns())
                throw new IllegalArgumentException("M must have same dimensions as A");

            //calculate (A - sigma*M)^-1
            Matrix res = A.subtract(M.multiply(sigma));
            Matrix res_inv = invert(res);
            return eigsh(asLinearOperation(A), A.rows(), nev, 5, which, ncv, sigma, maxIter, tolerance, asLinearOperation(M), asLinearOperation(res_inv));
        }
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x in Cayley-transformed mode to find eigenvalues near sigma.
     * A and M must be square, symmetric and match in dimensions. M is positive semi-definite
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A Linear operation representing left multiplication by A
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param OP_inv Linear operation representing left multiplication by (A -sigma*M)^-1
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_cayley(LinearOperation A, int n, int nev, LinearOperation OP_inv, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return eigsh(A, n, nev, 5, which, ncv, sigma, maxIter, tolerance, null, OP_inv);
    }





    /**
     * trickle-down method used internally by publicly exposed methods
     */
    private static SymmetricArpackSolver eigsh(LinearOperation A, int n, int nev, int mode, String which, Integer ncv, double sigma,
                                         int maxIter, double tolerance, LinearOperation M, LinearOperation Minv) {
        if (nev <= 0)
            throw new IllegalArgumentException("nev must be positive, nev="+nev);
        if (nev >= n)
            throw new IllegalArgumentException("nev must be smaller than n="+n);
        if (!SEUPD_WHICH.contains(which))
            throw new IllegalArgumentException("which must be one of 'LM', 'SM', 'LA', 'SA' or 'BE");
        if (maxIter <= 0)
            throw new IllegalArgumentException("max number of iterations must be positive");
        if (ncv == null)
            ncv = Math.min(n, Math.max(2 * nev + 1, 20));
        else if (ncv > n || ncv <= nev)
            throw new IllegalArgumentException("ncv must be nev<ncv<=n but is "+ncv);

        if (mode == 3)
            return new SymmetricArpackSolver(n, nev, mode, which.getBytes(), ncv, sigma, maxIter, tolerance, null, M, Minv);
        return new SymmetricArpackSolver(n, nev, mode, which.getBytes(), ncv, sigma, maxIter, tolerance, A, M, Minv);
    }
}