package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.bytedeco.arpackng.global.arpack;

import java.util.HashSet;
import java.util.Set;

import static net.scoreworks.arpackj.MatrixOperations.IDENTITY;

public class SymmetricArpackSolver extends ArpackSolver {
    private static final Set<String> SEUPD_WHICH = new HashSet<>();
    static {
        SEUPD_WHICH.add("LM");
        SEUPD_WHICH.add("SM");
        SEUPD_WHICH.add("LA");
        SEUPD_WHICH.add("SA");
        SEUPD_WHICH.add("BE");
    }
    private final LinearOperation OP, B;
    private LinearOperation OPa, OPb, A_matvec;

    SymmetricArpackSolver(LinearOperation A_matvec, int n, int nev, int mode, String which, Integer ncv, double sigma,
                          int maxIter, double tol, LinearOperation M_matvec, LinearOperation Minv_matvec) {
        super(n, nev, mode, which.getBytes(), ncv, sigma, maxIter, tol);
        if (!SEUPD_WHICH.contains(which))
            throw new IllegalArgumentException("which must be one of 'LM', 'SM', 'LA', 'SA' or 'BE");

        if (mode == 1) {
            if (A_matvec == null)
                throw new IllegalArgumentException("matvec must be specified for mode=1");
            if (M_matvec != null)
                throw new IllegalArgumentException("M_matvec cannot be specified for mode=1");
            if (Minv_matvec != null)
                throw new IllegalArgumentException("M_matvec cannot be specified for mode=1");

            this.OP = A_matvec;
            this.B = IDENTITY;
            this.bmat = "I".getBytes();
        }
        else if (mode == 2) {
            if (A_matvec == null)
                throw new IllegalArgumentException("matvec must be specified for mode=2");
            if (M_matvec == null)
                throw new IllegalArgumentException("M_matvec must be specified for mode=2");
            if (Minv_matvec == null)
                throw new IllegalArgumentException("Minv_matvec must be specified for mode=2");

            this.OP = (x, off) -> Minv_matvec.apply(A_matvec.apply(x, off), 0);
            this.OPa = Minv_matvec;
            this.OPb = A_matvec;
            this.B = M_matvec;
            this.bmat = "G".getBytes();
        }
        else if (mode == 3) {
            if (A_matvec != null)
                throw new IllegalArgumentException("matvec must not be specified for mode=3");
            if (Minv_matvec == null)
                throw new IllegalArgumentException("Minv_matvec must be specified for mode=3");

            if (M_matvec == null) {
                this.OP = Minv_matvec;
                this.OPa = Minv_matvec;
                this.B = IDENTITY;
                this.bmat = "I".getBytes();
            }
            else {
                this.OP = (x, off) -> Minv_matvec.apply(M_matvec.apply(x, off), 0);
                this.OPa = Minv_matvec;
                this.B = M_matvec;
                this.bmat = "G".getBytes();
            }
        }
        else if (mode == 4) {
            if (A_matvec == null)
                throw new IllegalArgumentException("matvec must be specified for mode=4");
            if (M_matvec != null)
                throw new IllegalArgumentException("M_matvec must not be specified for mode=4");
            if (Minv_matvec == null)
                throw new IllegalArgumentException("Minv_matvec must be specified for mode=4");

            this.OP = (x, off) -> Minv_matvec.apply(A_matvec.apply(x, off), 0);
            this.OPa = Minv_matvec;
            this.B = A_matvec;
            this.bmat = "G".getBytes();
        }
        else if (mode == 5) {
            if (A_matvec == null)
                throw new IllegalArgumentException("matvec must be specified for mode=5");
            if (Minv_matvec == null)
                throw new IllegalArgumentException("Minv_matvec must be specified for mode=5");

            this.OPa = Minv_matvec;
            this.A_matvec = A_matvec;
            if (M_matvec == null) {
                //this.OP = Minv_matvec(A_matvec(x) + sigma*x))
                this.OP = (x, off) -> {
                    double[] res = A_matvec.apply(x, off);
                    for (int i=0; i<res.length; i++) {
                        res[i] += sigma*x[i + off];
                    }
                    return Minv_matvec.apply(res, 0);
                };
                this.B = IDENTITY;
                this.bmat = "I".getBytes();
            }
            else {
                this.OP = (x, off) -> {
                    //this.OP = Minv_matvec(A_matvec(x) + sigma*M_matvec(x)))
                    double[] res1 = A_matvec.apply(x, off);
                    double[] res2 = M_matvec.apply(x, off);
                    for (int i=0; i<res1.length; i++) {
                        res1[i] += sigma*res2[i];
                    }
                    return Minv_matvec.apply(res1, 0);
                };
                this.B = M_matvec;
                this.bmat = "G".getBytes();
            }
        }
        else {
            throw new IllegalArgumentException("mode=" + mode + " not implemented");
        }
    }

    protected void iterate() {
        arpack.dsaupd_c(ido, bmat, n, which, nev, tol, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info);

        if (ido[0] == -1) {
            //initialization: compute y = Op*x
            System.arraycopy(OP.apply(workd, ipntr[0] - 1), 0, workd, ipntr[1] - 1, n);
        }
        else if (ido[0] == 1) {
            //compute y = Op*x
            if (mode == 1) {
                System.arraycopy(OP.apply(workd, ipntr[0] - 1), 0, workd, ipntr[1] - 1, n);
            }
            else if (mode == 2) {
                System.arraycopy(OPb.apply(workd, ipntr[0] - 1), 0, workd, ipntr[0] - 1, n);
                System.arraycopy(OPa.apply(workd, ipntr[0] - 1), 0, workd, ipntr[1] - 1, n);
            }
            else if (mode == 5) {
                double[] Ax = new double[n];
                System.arraycopy(A_matvec.apply(workd, ipntr[0] - 1), 0, Ax, 0, n);
                for (int i=0; i<n; i++) {
                    Ax[i] += sigma*workd[ipntr[2]-1+i];
                }
                System.arraycopy(OPa.apply(Ax, 0), 0, workd, ipntr[1] - 1, n);
            }
            else {
                //compute OPa*(B*x)
                System.arraycopy(OPa.apply(workd, ipntr[2] - 1), 0, workd, ipntr[1] - 1, n);
            }
        }
        else if (ido[0] == 2) {
            System.arraycopy(B.apply(workd, ipntr[0] - 1), 0, workd, ipntr[1] - 1, n);
        }
        else if (ido[0] == 3) {
            throw new IllegalArgumentException("ARPACK requested user shifts. Assure iparam(1) is set to 0");
        }
    }

    protected void extract() {
        //There is negligible additional cost to obtain eigenvectors so always get them
        int rvec = 1;                   //0 would mean no eigenvectors
        byte[] howmy = "A".getBytes();  //get all nev eigenvalues/eigenvectors
        int[] select = new int[ncv];    //unused
        d = new double[nev];            //eigenvalues in ascending order
        z = new double[n * nev];        //eigenvectors

        arpack.dseupd_c(rvec, howmy, select, d, z, ncv, sigma, bmat, n, which, nev, tol, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info);
        if (info[0] != 0)
            throw new ArpackException(getExtractionErrorCode(info[0]));
    }

    protected void noConvergence() {
        extract();
        int nconv = iparam[4];  //number of converged ritz vectors
    }

    @Override
    protected String getErrorCode(int errorCode) {
        switch (errorCode) {
            case 1 -> {
                return "Maximum number of iterations taken. All possible eigenvalues of OP has been found. IPARAM(5) returns the number of wanted converged Ritz values.";
            }
            case 3 -> {
                return "No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV.";
            }
            case -1 -> {
                return "N must be positive.";
            }
            case -2 -> {
                return "NEV must be positive.";
            }
            case -3 -> {
                return "NCV-NEV >= 2 and less than or equal to N.";
            }
            case -4 -> {
                return "The maximum number of Arnoldi update iterations allowed must be greater than zero.";
            }
            case -5 -> {
                return "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'.";
            }
            case -6 -> {
                return "BMAT must be one of 'I' or 'G'.";
            }
            case -7 -> {
                return "Length of private work array WORKL is not sufficient.";
            }
            case -8 -> {
                return "Error return from LAPACK eigenvalue calculation.";
            }
            case -9 -> {
                return "Starting vector is zero.";
            }
            case -10 -> {
                return "IPARAM(7) must be 1, 2, 3, or 4.";
            }
            case -11 -> {
                return "IPARAM(7) = 1 and BMAT = 'G' are incompatable";
            }
            case -12 -> {
                return "IPARAM(1) must be equal to 0 or 1";
            }
            case -13 -> {
                return "NEV and WHICH = 'BE' are incompatable";
            }
            case -9999 -> {
                return "Could not build an Arnoldi factorization";
            }
        }
        return "unknown ARPACK error";
    }

    @Override
    protected String getExtractionErrorCode(int errorCode) {
        switch (errorCode) {
            case 1 -> {
                return "The Schur form computed by LAPACK routine dlahqr could not be reordered by LAPACK routine dtrsen. Re-enter subroutine dneupd  with IPARAM(5)NCV and increase the size of the arrays DR and DI to have dimension at least dimension NCV and allocate at least NCV columns for Z. NOTE: Not necessary if Z and V share the same space. Please notify the authors if this error occurs.";
            }
            case -1 -> {
                return "N must be positive.";
            }
            case -2 -> {
                return "NEV must be positive.";
            }
            case -3 -> {
                return "NCV-NEV >= 2 and less than or equal to N.";
            }
            case -5 -> {
                return "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'.";
            }
            case -6 -> {
                return "BMAT must be one of 'I' or 'G'.";
            }
            case -7 -> {
                return "Length of private work array WORKL is not sufficient.";
            }
            case -8 -> {
                return "Error return from calculation of a real Schur form. Informational error from LAPACK routine dlahqr.";
            }
            case -9 -> {
                return "Error return from calculation of eigenvectors.";
            }
            case -10 -> {
                return "IPARAM(7) must be 1, 2, 3, or 4.";
            }
            case -11 -> {
                return "IPARAM(7) = 1 and BMAT = 'G' are incompatable";
            }
            case -12 -> {
                return "HOWMNY = 'S' not yet implemented";
            }
            case -13 -> {
                return "HOWMNY must be one of 'A' or 'P' if RVEC = true";
            }
            case -14 -> {
                return "DNAUPD  did not find any eigenvalues to sufficient accuracy.";
            }
            case -15 -> {
                return "DNEUPD got a different count of the number of converged Ritz values than DNAUPD got. This indicates the user probably made an error in passing data from DNAUPD to DNEUPD or that the data was modified before entering DNEUPD";

            }
        }
        return "unknown ARPACK error";
    }
}