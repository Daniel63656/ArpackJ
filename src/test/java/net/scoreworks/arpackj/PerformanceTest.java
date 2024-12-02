/*
Copyright 2023 Daniel Maier

This file is part of ArpackJ (https://github.com/Daniel63656/ArpackJ)

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

package net.scoreworks.arpackj;

import net.scoreworks.arpackj.eig.SymmetricArpackSolver;
import net.scoreworks.arpackj.eig.UnsymmetricArpackSolver;
import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.Test;
import org.la4j.iterator.MatrixIterator;
import org.la4j.matrix.SparseMatrix;
import org.la4j.matrix.sparse.CRSMatrix;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.Random;

import static net.scoreworks.arpackj.eig.MatrixDecomposition.eigs_shiftInvertReal;
import static net.scoreworks.arpackj.eig.MatrixDecomposition.eigsh_shiftInvert;

public class PerformanceTest {
    static {
        // Load the OpenCV native library
        System.load(System.getProperty("user.dir") + "/openCV/opencv_java460.dll");
    }

    public static CRSMatrix generateRandomSparseMatrix(int rows, int columns, double sparsity) {
        Random random = new Random();
        CRSMatrix matrix = new CRSMatrix(rows, columns);
        //generate random non-zero elements and insert them into the matrix
        for (int i=0; i<rows*columns; i++) {
            if (random.nextDouble() < sparsity)
                matrix.set(i/columns, i%columns, Math.random());
        }
        return matrix;
    }

    public static float checkSparsity(SparseMatrix A) {
        MatrixIterator it = A.nonZeroIterator();
        int count = 0;
        while (it.hasNext()) {
            count++;
            it.next();
        }
        return count/(float)(A.rows()*A.columns());
    }

    //@Test
    public void performanceComparisonTest() {
        int rows = 2000;
        SparseMatrix A = generateRandomSparseMatrix(rows, rows, 0.1f);

        //copy as openCV matrix
        Mat cvMat = Mat.zeros(rows, rows, CvType.CV_32F);
        MatrixIterator it = A.nonZeroIterator();
        double value;
        while (it.hasNext()) {
            value = it.next();
            cvMat.put(it.rowIndex(), it.columnIndex(), value);
        }
        System.out.println("starting...");

        long startTime = System.currentTimeMillis();
        UnsymmetricArpackSolver solver = eigs_shiftInvertReal(A, null,  2, "LM", new Complex(0, 0), null, 500, 1e-12);
        solver.solve();
        long endTime = System.currentTimeMillis();
        System.out.println("ArpackJ Time: " + (endTime - startTime) + " milliseconds");

        Mat eigenvalues = new Mat();
        Mat eigenvectors = new Mat();
        startTime = System.currentTimeMillis();
        Core.eigen(cvMat, eigenvalues, eigenvectors);
        endTime = System.currentTimeMillis();
        System.out.println("openCV Time: " + (endTime - startTime) + " milliseconds");

        /*startTime = System.currentTimeMillis();
        EigenDecompositor decompositor = new EigenDecompositor(A);
        decompositor.decompose();
        endTime = System.currentTimeMillis();
        System.out.println("la4j Time: " + (endTime - startTime) + " milliseconds");*/
    }

    //@Test
    public void sparsityPerformanceTest() {
        int rows = 2000;
        SparseMatrix A = generateRandomSparseMatrix(rows, rows, 0.1f);
        System.out.println("starting...");

        long startTime = System.currentTimeMillis();
        SymmetricArpackSolver solver = eigsh_shiftInvert(A, null, 2, "LM", 0, null, 500, 1e-12);
        solver.solve();
        long endTime = System.currentTimeMillis();
        System.out.println("ArpackJ Time: " + (endTime - startTime) + " milliseconds");
    }
}
