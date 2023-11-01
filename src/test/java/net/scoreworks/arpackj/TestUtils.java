package net.scoreworks.arpackj;

import org.apache.commons.math3.complex.Complex;
import org.junit.jupiter.api.Assertions;

public final class TestUtils {
    private static final double epsilon = 0.0001;
    private TestUtils() {}

    public static void checkSolution(double[] d_expected, double[] z_expected, int[] slice, double[] d_actual, double[] z_actual) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<slice.length; i++) {
            Assertions.assertEquals(d_expected[slice[i]], d_actual[i], epsilon);
        }
        System.out.println("All eigenvalues correct");

        //check eigenvectors match up to sign flip
        for (int i=0; i< slice.length; i++) {
            if (i > 0) System.out.println("eigenvector "+i+" correct");
            for (int j=0; j<d_expected.length; j++) {
                Assertions.assertEquals(Math.abs(z_expected[slice[i]*5 + j]), Math.abs(z_actual[i*5 + j]), epsilon);
            }
        }
        System.out.println("eigenvector "+slice.length+" correct");
    }

    public static void checkSolution(Complex[] d_expected, Complex[] z_expected, int[] slice, Complex[] d_actual, Complex[] z_actual) {
        //idx is a list of entries that should be returned, considering they are returned in ascending order
        for (int i=0; i<slice.length; i++) {
            Assertions.assertEquals(d_expected[slice[i]].getReal(), d_actual[i].getReal(), epsilon);
            Assertions.assertEquals(d_expected[slice[i]].getImaginary(), d_actual[i].getImaginary(), epsilon);
        }
        System.out.println("All eigenvalues correct");

        //check eigenvectors match up to sign flip
        Complex rotation = null;
        for (int i=0; i<slice.length; i++) {
            if (i > 0) System.out.println("eigenvector "+i+" correct");
            for (int j=0; j<d_expected.length; j++) {
                //each vector can be rotated in the complex plane so calculate rotation factor on first element
                if (j == 0) rotation = z_expected[slice[i]*5].divide(z_actual[i*5]);
                Assertions.assertEquals(z_expected[slice[i]*5 + j].getReal(), z_actual[i*5 + j].multiply(rotation).getReal(), epsilon);
                Assertions.assertEquals(z_expected[slice[i]*5 + j].getImaginary(), z_actual[i*5 + j].multiply(rotation).getImaginary(), epsilon);
            }
        }
        System.out.println("eigenvector "+slice.length+" correct");
    }
}
