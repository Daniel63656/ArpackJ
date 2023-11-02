package net.scoreworks.arpackj.eig;

import net.scoreworks.arpackj.LinearOperation;
import org.apache.commons.math3.complex.Complex;
import org.bytedeco.arpackng.global.arpack;

import java.util.*;

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
    private LinearOperation A;  //needed for modes 3,4 to transform eigenvalues/eigenvectors back into original system

    UnsymmetricArpackSolver(LinearOperation A_matvec, int n, int nev, int mode, String which, Integer ncv, Complex sigma,
                            int maxIter, double tol, LinearOperation M_matvec, LinearOperation Minv_matvec) {
        super(n, nev, mode, which.getBytes(), ncv, maxIter, tol);
        ipntr = new int[14];
        lworkl = 3*this.ncv * (this.ncv + 2);
        workl = new double[lworkl];
        this.sigma = Objects.requireNonNullElseGet(sigma, () -> new Complex(0, 0));

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
            if (A_matvec == null)
                throw new IllegalArgumentException("matvec must not be specified for mode=3,4");
            if (Minv_matvec == null)
                throw new IllegalArgumentException("Minv_matvec must be specified for mode=3,4");
            this.A = A_matvec;

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
        int rvec = 1;                               //0 would mean no eigenvectors
        byte[] howmy = "A".getBytes();              //get all nev eigenvalues/eigenvectors
        int[] select = new int[ncv];                //unused
        double[] workev = new double[3 * ncv];
        //ATTENTION: the +1 is needed for a potential complex conjugate that needs to be sorted out later in the routine
        double[] d_r = new double[nev+1];           //eigenvalues in ascending order
        double[] d_i = new double[nev+1];
        double[] z_r = new double[n * (nev+1)];     //eigenvectors, stored consecutively as real part (, imaginary part)

        arpack.dneupd_c(rvec, howmy, select, d_r, d_i, z_r, ncv, sigma.getReal(), sigma.getImaginary(), workev, bmat, n, which, nev, tol, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info);
        if (info[0] != 0)
            throw new ArpackException(getExtractionErrorCode(info[0]));
        int nReturned = iparam[4];  // number of good eigenvalues returned (might be less than nev+1)

        //calculate eigenvalues and eigenvalues and store them as pairs to allow for easy sorting afterwards
        EigenPair[] eigenPair = new EigenPair[nev+1];
        if (sigma.getImaginary() == 0) {
            //in this case eigenvalues are correct and only eigenvectors need to be computed
            int i = 0;
            while (i <= nev) {
                if (Math.abs(d_i[i]) != 0) {
                    // this is a complex conjugate pair (2 columns in z_r)
                    if (i < nev) {
                        eigenPair[i] = new EigenPair(new Complex(d_r[i], d_i[i]), n);
                        eigenPair[i+1] = new EigenPair(eigenPair[i].d.conjugate(), n);
                        for (int j=0; j<n; j++) {
                            eigenPair[i].z[j] = new Complex(z_r[i*n + j], z_r[(i + 1)*n + j]);
                            eigenPair[i+1].z[j] = eigenPair[i].z[j].conjugate();
                        }
                        i++;
                    }
                    else {
                        //last eigenvalue is complex but only one slot is free. Ignore this case and return all nev
                        //previously discovered eigenPairs by setting nReturn to nev
                        nReturned = nev;
                    }
                }
                else {
                    //this is a real eigenvector (1 column in z_r)
                    eigenPair[i] = new EigenPair(new Complex(d_r[i], d_i[i]), n);
                    for (int j=0; j<n; j++) {
                        eigenPair[i].z[j] = new Complex(z_r[i*n + j], 0);
                    }
                }
                i++;
            }
        }
        else {
            // d_r contains the real part of the Ritz values of OP computed by DNAUPD. eigenvectors AND eigenvalues
            // must therefore be computed
            int i = 0;
            while (i <= nev) {
                if (Math.abs(d_i[i]) == 0) {
                    eigenPair[i] = new EigenPair(n);
                    // d = z_r * A(z_r)
                    double[] Az = A.apply(z_r, i*n);
                    double real = 0;
                    for (int j=0; j<n; j++) {
                        real += z_r[i*n + j] * Az[j];
                        eigenPair[i].z[j] = new Complex(z_r[i*n + j], 0);
                    }
                    eigenPair[i].d = new Complex(real, 0);
                }
                else {
                    if (i < nev) {
                        eigenPair[i] = new EigenPair(n);
                        eigenPair[i+1] = new EigenPair(n);
                        double[] Az    = A.apply(z_r, i*n);
                        double[] Az_cc = A.apply(z_r, (i+1)*n);
                        double real = 0;
                        double imag = 0;
                        for (int j=0; j<n; j++) {
                            eigenPair[i].z[j] = new Complex(z_r[i*n + j], z_r[(i + 1)*n + j]);
                            eigenPair[i+1].z[j] = eigenPair[i].z[j].conjugate();
                            real += z_r[i*n + j] * Az[j]    + z_r[(i+1)*n + j] * Az_cc[j];
                            imag += z_r[i*n + j] * Az_cc[j] - z_r[(i+1)*n + j] * Az[j];
                        }
                        eigenPair[i].d = new Complex(real, imag);
                        eigenPair[i+1].d = eigenPair[i].d.conjugate();
                        i++;
                    }
                    else {
                        //last eigenvalue is complex but only one slot is free. Ignore this case and return all nev
                        //previously discovered eigenPairs by setting nReturn to nev
                        nReturned = nev;
                    }
                }
                i++;
            }
        }

        //got less than or equal eigenPairs to specified number nev
        if (nReturned <= nev) {
            this.d = new Complex[nReturned];
            this.z = new Complex[n * nReturned];
            //copy first nReturned values to solution arrays and return
            for (int i=0; i<nReturned; i++) {
                d[i] = eigenPair[i].d;
                System.arraycopy(eigenPair[i].z, 0, this.z, i*n, n);
            }
            return;
        }
        //at this point there are nev+1 possible eigenvector/eigenvalue pairs. Filter one out based on "which"
        this.d = new Complex[nev];
        this.z = new Complex[n * nev];

        if (mode == 1 || mode == 2) {   //no shift
            if (which[1] == 'R')
                Arrays.sort(eigenPair, Comparator.comparing(eig -> eig.d.getReal()));
            else if (which[1] == 'I')
                Arrays.sort(eigenPair, Comparator.comparing(eig -> eig.d.getImaginary()));
            else    //sort by magnitude
                Arrays.sort(eigenPair, Comparator.comparing(eig -> eig.d.abs()));
        }
        else {   //sort by 1 / (d-sigma)
            Complex one = new Complex(1, 0);
            if (which[1] == 'R')
                Arrays.sort(eigenPair, Comparator.comparing(eig -> one.divide(eig.d.subtract(sigma)).getReal()));
            else if (which[1] == 'I')
                Arrays.sort(eigenPair, Comparator.comparing(eig -> one.divide(eig.d.subtract(sigma)).getImaginary()));
            else    //sort by magnitude
                Arrays.sort(eigenPair, Comparator.comparing(eig -> one.divide(eig.d.subtract(sigma)).abs()));
        }


        if (which[0] == 'S') {
            //copy first nev values to solution arrays
            for (int i=0; i<nev; i++) {
                d[i] = eigenPair[i].d;
                System.arraycopy(eigenPair[i].z, 0, this.z, i*n, n);
            }
        }
        else if (which[0] == 'L') {
            //copy last nev values in reverse to solution arrays
            for (int i=0; i<nev; i++) {
                d[i] = eigenPair[nev-i].d;
                System.arraycopy(eigenPair[nev-i].z, 0, this.z, i*n, n);
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

    /**
     * Simple POJO to help during calculation and sorting of eigenvalues/-vectors
     */
    private static class EigenPair {
        Complex d;
        Complex[] z;
        public EigenPair(int nev) {
            this.z = new Complex[nev];
        }
        public EigenPair(Complex d, int nev) {
            this.d = d;
            this.z = new Complex[nev];
        }
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