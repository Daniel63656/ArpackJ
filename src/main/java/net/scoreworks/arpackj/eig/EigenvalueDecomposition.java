package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.la4j.Matrix;

import java.util.HashSet;
import java.util.Set;

import static net.scoreworks.arpackj.MatrixOperations.*;

public final class EigenvalueDecomposition {
    private EigenvalueDecomposition() {}  //make instantiation impossible

    //TODO move to UNSYMMETRIC ARPACK SOLVER
    /*static final Set<String> NEUPD_WHICH = new HashSet<>();
    static {
        NEUPD_WHICH.add("LM");
        NEUPD_WHICH.add("SM");
        NEUPD_WHICH.add("LR");
        NEUPD_WHICH.add("SR");
        NEUPD_WHICH.add("LI");
        NEUPD_WHICH.add("SI");
    }*/

    /**
     * Solve the standard eigenvalue problem A*x = lambda*x for a square, symmetric matrix A
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
     * @param A Linear operation representing left multiplication by A
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
     * Solve the general eigenvalue problem A*x = lambda*M*x. A and M must be square, symmetric and match in dimensions
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh(Matrix A, int nev, Matrix M, String which, Integer ncv, int maxIter, double tolerance) {
        if (A.rows() != A.columns())
            throw new IllegalArgumentException("A is not a square matrix");
        if (M.rows() != A.rows() || M.columns() != A.columns())
            throw new IllegalArgumentException("M must have same dimensions as A");

        //calculate inverse of M
        Matrix M_inv = invert(M);
        return new SymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 2, which, ncv, 0, maxIter, tolerance, asLinearOperation(M), asLinearOperation(M_inv));
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
    public static SymmetricArpackSolver eigsh(LinearOperation A, int n, int nev, LinearOperation M, LinearOperation M_inv, String which, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(A, n, nev, 2, which, ncv, 0, maxIter, tolerance, M, M_inv);
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
            return new SymmetricArpackSolver(null, A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, null, asLinearOperation(res_inv));
        }
        else {
            if (M.rows() != A.rows() || M.columns() != A.columns())
                throw new IllegalArgumentException("M must have same dimensions as A");

            //calculate (A - sigma*M)^-1
            Matrix res = A.subtract(M.multiply(sigma));
            Matrix res_inv = invert(res);
            return new SymmetricArpackSolver(null, A.rows(), nev, 3, which, ncv, sigma, maxIter, tolerance, asLinearOperation(M), asLinearOperation(res_inv));
        }
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x in shift-invert mode to find eigenvalues near sigma.
     * A and M must be square, symmetric and match in dimensions. M is positive semi-definite
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param A Linear operation representing left multiplication by A
     * @param n shape (rows, or columns) of A
     * @param nev number of eigenvalues to compute
     * @param OP_inv Linear operation representing left multiplication by (A - sigma*M)^-1
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_shiftInvert(int n, int nev, LinearOperation M, LinearOperation OP_inv, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(null, n, nev, 3, which, ncv, sigma, maxIter, tolerance, M, OP_inv);
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x in buckling mode to find eigenvalues near sigma.
     * A and M must be square, symmetric and match in dimensions. M is positive semi-definite
     * If M is null, the standard eigenvalue problem will be solved instead
     * @param nev number of eigenvalues to compute
     * @param which select which eigenvalues to compute
     * @param sigma shift applied to A
     * @param ncv number of Arnoldi vectors. Use null to let them be chosen automatically
     * @param maxIter maximal number of iterations
     * @param tolerance iteration is terminated when this relative tolerance is reached
     */
    public static SymmetricArpackSolver eigsh_buckling(Matrix A, int nev, Matrix M, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        //calculate (A -sigma*M)^-1
        //TODO check if sigma is 0
        Matrix res = A.subtract(M.multiply(sigma));
        Matrix res_inv = invert(res);
        return new SymmetricArpackSolver(asLinearOperation(A), A.rows(), nev, 4, which, ncv, sigma, maxIter, tolerance, null, asLinearOperation(res_inv));
    }

    /**
     * Solve the general eigenvalue problem A*x = lambda*M*x in buckling mode to find eigenvalues near sigma.
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
    public static SymmetricArpackSolver eigsh_buckling(LinearOperation A, int n, int nev, LinearOperation OP_inv, String which, double sigma, Integer ncv, int maxIter, double tolerance) {
        return new SymmetricArpackSolver(A, n, nev, 4, which, ncv, sigma, maxIter, tolerance, null, OP_inv);
    }
}