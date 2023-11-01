package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.apache.commons.math3.complex.Complex;
import org.bytedeco.arpackng.global.arpack;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;

import static net.scoreworks.arpackj.MatrixOperations.IDENTITY;

public class UnsymmetricArpackSolver extends ArpackSolver {
    private static final Set<String> NEUPD_WHICH = new HashSet<>();
    static {
        NEUPD_WHICH.add("LM");
        NEUPD_WHICH.add("SM");
        NEUPD_WHICH.add("LR");
        NEUPD_WHICH.add("SR");
        NEUPD_WHICH.add("LI");
        NEUPD_WHICH.add("SI");
    }

    /** store computed eigenvalues */
    private Complex[] d;

    /** store computed eigenvectors */
    private Complex[] z;

    /** shift parameter when used in shift-invert mode (3,4,5) */
    private final Complex sigma;
    private final LinearOperation OP, B;
    private LinearOperation OPa, OPb;

    UnsymmetricArpackSolver(LinearOperation A_matvec, int n, int nev, int mode, String which, Integer ncv, Complex sigma,
                            int maxIter, double tol, LinearOperation M_matvec, LinearOperation Minv_matvec) {
        super(n, nev, mode, which.getBytes(), ncv, maxIter, tol);
        ipntr = new int[14];
        lworkl = 3*this.ncv * (this.ncv + 2);
        workl = new double[lworkl];
        this.sigma = sigma;

        if (!NEUPD_WHICH.contains(which))
            throw new IllegalArgumentException("which must be one of 'LM', 'SM', 'LR', 'SR', 'LI' or 'SI");
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
        else if (mode == 3 || mode == 4) {
            if (A_matvec != null)
                throw new IllegalArgumentException("matvec must not be specified for mode=3,4");
            if (Minv_matvec == null)
                throw new IllegalArgumentException("Minv_matvec must be specified for mode=3,4");

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
        else {
            throw new IllegalArgumentException("mode=" + mode + " not implemented");
        }
    }

    protected void iterate() {
        arpack.dnaupd_c(ido, bmat, n, which, nev, tol, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info);
        //System.out.println("ido:"+ido[0]);

        if (ido[0] == -1) {
            //initialization: compute y = Op*x
            System.arraycopy(OP.apply(workd, ipntr[0] - 1), 0, workd, ipntr[1] - 1, n);
        }
        else if (ido[0] == 1) {
            //compute y = Op*x
            if (mode == 1 || mode == 2) {
                System.arraycopy(OP.apply(workd, ipntr[0] - 1), 0, workd, ipntr[1] - 1, n);
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
        double[] workev = new double[3 * ncv];
        //ATTENTION: the +1 is needed for a potential complex conjugate that needs to be sorted out later in the routine
        double[] d_r = new double[nev+1];           //eigenvalues in ascending order
        double[] d_i = new double[nev+1];
        double[] z_r = new double[n * (nev+1)];     //eigenvectors, stored consecutively as real part (, imaginary part)

        if (sigma == null)
            arpack.dneupd_c(rvec, howmy, select, d_r, d_i, z_r, ncv, 0, 0, workev, bmat, n, which, nev, tol, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info);
        else
            arpack.dneupd_c(rvec, howmy, select, d_r, d_i, z_r, ncv, sigma.getReal(), sigma.getImaginary(), workev, bmat, n, which, nev, tol, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info);
        if (info[0] != 0)
            throw new ArpackException(getExtractionErrorCode(info[0]));

        //create complex eigenvectors
        d = new Complex[nev];
        z = new Complex[n * nev];
        Complex[] z_raw = new Complex[n * (nev+1)];
        //TODO if sigma_i == 0

        int i = 0;
        while (i <= nev) {
            if (Math.abs(d_i[i]) != 0) {
                // this is a complex conjugate pair (2 columns in z_r)
                if (i < nev) {
                    for (int j=0; j<n; j++) {
                        z_raw[i*n + j] = new Complex(z_r[i*n + j], z_r[(i + 1)*n + j]);
                        z_raw[(i + 1)*n + j] = z_raw[i*n + j].conjugate();
                    }
                    i++;
                }
                else {

                }
            }
            else {
                //this is a real eigenvector (1 column  in z_r)
                for (int j=0; j<n; j++) {
                    z_raw[i*n + j] = new Complex(z_r[i*n + j], 0);
                }
            }
            i++;
        }
        //TODO handle sigma != 0


        //now there are nev+1 possible eigenvector/eigenvalue pairs. Filter one out based on "which"
        //TODO check if this is true with nreturned

        if (which[1] == 'R')
            Arrays.sort(z_raw, Comparator.comparing(n -> Math.abs(n.getReal())));
        else if (which[1] == 'I')
            Arrays.sort(z_raw, Comparator.comparing(n -> Math.abs(n.getImaginary())));


        if (which[0] == 'L') {
            //copy first nev values to solution arrays
            for (i=0; i<nev; i++)
                d[i] = new Complex(d_r[i], d_i[i]);
            System.arraycopy(z_raw, 0, z, 0, n*nev);
        }
        else if (which[0] == 'S') {
            //copy last nev values in reverse to solution arrays
            for (i=0; i<nev; i++)
                d[i] = new Complex(d_r[nev - i], d_i[nev - i]);
            int idx;
            for (i=0; i<n*nev; i++) {
                idx = n*(nev-i/n) + i%n;
                z[i] = z_raw[idx];
            }
        }
    }

    public Complex[] getEigenvalues() {
        return d;
    }
    public Complex[] getEigenvectors() {
        return z;
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