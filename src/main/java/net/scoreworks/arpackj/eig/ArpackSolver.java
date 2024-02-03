/*
Copyright 2023 Daniel Maier

This file is part of ArpackJ (https://github.com/Daniel63656/ArpackJ)

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

package net.scoreworks.arpackj.eig;

public abstract class ArpackSolver {
    /** dimension of the eigenvalue problem */
    protected int n;

    /** number of eigenvalues to compute */
    protected int nev;

    /** specify solving mode */
    protected int mode;

    /** ARPACK internal parameter */
    protected byte[] which;

    /** storage for the Lanczos vectors*/
    protected double[] v;

    /** maximal number of iterations */
    protected int maxIter;

    /** relative tolerance */
    protected double tol;

    /** store instruction code during the reversed communication*/
    protected int[] ido = new int[1];

    /** specify nature of the eigenvalue problem, 'I': standard, 'G': general */
    protected byte[] bmat;

    /** residual vectors*/
    protected double[] resid;

    /** Number of Lanczos vectors. ncv <= n */
    protected int ncv;

    /** ARPACK internal parameters */
    protected int[] iparam = new int[11];

    /** ARPACK internal parameters */
    protected int[] ipntr;

    /** arpack workspace array*/
    protected double[] workd;
    protected int lworkl;           // Size of the work array

    /** arpack work array */
    protected double[] workl;

    /** returns the status of the computation upon completion */
    protected int[] info = new int[1];


    ArpackSolver(int n, int nev, int mode, byte[] which, Integer ncv, int maxIter, double tol) {
        if (nev <= 0)
            throw new IllegalArgumentException("nev must be positive, nev="+nev);
        if (nev >= n)
            throw new IllegalArgumentException("nev must be smaller than n="+n);
        if (maxIter <= 0)
            throw new IllegalArgumentException("max number of iterations must be positive");
        if (ncv == null)
            ncv = Math.min(n, Math.max(2 * nev + 1, 20));
        else if (ncv > n || ncv <= nev)
            throw new IllegalArgumentException("ncv must be nev<ncv<=n but is "+ncv);

        this.n = n;
        this.nev = nev;
        this.mode = mode;
        this.ncv = ncv;
        v = new double[n * ncv];
        resid = new double[n];  //if no v0 is provided via setInitialV(), ARPACK will choose them at random
        this.maxIter = maxIter;
        this.which = which;
        this.tol = tol;
        workd = new double[3 * n];
        
        // set solver mode and parameters
        iparam[0] = 1;     //shifts not provided by user
        iparam[2] = maxIter;
        iparam[3] = 1;
        iparam[6] = mode;
    }

    public void setInitialV(double[] v0) {
        System.arraycopy(v0, 0, resid, 0, n);
        info[0] = 1;
    }

    /**
     * solve the given eigenvalue problem
     */
    public void solve() {
        while (true) {
            iterate();
            if (ido[0] == 99) {     //done
                if (info[0] == 0) {
                    extract();
                    break;
                }
                if (info[0] == 1) {
                    noConvergence();
                    break;
                }
                else
                    throw new ArpackException(getErrorCode(info[0]));
            }
        }
    }

    /**
     * do one iteration step
     */
    protected abstract void iterate();

    /**
     * extract eigenvalues and eigenvectors upon convergence
     */
    protected abstract void extract();

    /**
     * extract subset of eigenvalues and eigenvectors that have converged
     */
    protected abstract void noConvergence();
    protected abstract String getErrorCode(int errorCode);
    protected abstract String getExtractionErrorCode(int errorCode);
}