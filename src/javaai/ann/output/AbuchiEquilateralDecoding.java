/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Lab 6
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.ann.output;

import javaai.util.Helper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * This class tests equilateral decoding tolerance for the iris data set.
 * @author Abuchi Obiegbu
 */
public class AbuchiEquilateralDecoding {
    /** Number of tests to run */
    public final static int NUM_TESTS = 100;

    /** Tolerance as a percent, e.g., 1.0 == 1% */
    public final static double PERTURBANCE = 30.0;

    /** EquilateralEncoding values */
    static double ideals[][] = {
            {-0.8660, -0.5000},   // Viginica
            {0.8660, -0.5000},   // Setosa
            {0.0000, 1.0000}    // Versicolor
    };

    /** Specie names -- order MUST correspond to measures */
    static final List<String> species =
            new ArrayList<>(Arrays.asList("virginica", "setosa", "versicolor"));

    /**
     * Launch point for program.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        EquilateralEncoding.load();

        // Random ran = new Random(0);
        Random ran = new Random();

        int success = 0;

        // reporting variables
        String report = "";

        // populating header display format
        String header = String.format("%3s %10s %18s | %11s %18s","#","ideal","Encoding","actual", "Encoding");

        // display report header
        System.out.println(header);

        for(int n=0; n < NUM_TESTS; n++) {
            // Pick a species randomly
            int idealIndex = ran.nextInt(ideals.length);

            // Get a random encoding from ideals using idealIndex.
            double[] encodings = ideals[idealIndex];

            // Create a new array of activations perturbed by the tolerance divided by 100.
            double[] activations = new double[encodings.length];

            // perturb the activations
            for(int k=0; k < encodings.length; k++) {
                double epsilon = 1 + ran.nextGaussian() * PERTURBANCE / 100.0;
                activations[k] = encodings[k] * epsilon;
            }
            
            // Decode these perturbed activations.
            int actualIndex = EquilateralEncoding.eq.decode(activations);

            // populate report body
            report = String.format("%3d %10s %18s | %11s %18s ", n+1, species.get(idealIndex),
                        Helper.asString(encodings), species.get(actualIndex), Helper.asString(activations));

            // If the predicted index equals the actual index, update success count.
            if (idealIndex == actualIndex) {
                // increase success count
                success++;
            } else {
                // add missed to the report body
                report += "MISSED!";
            }
            // display report content
            System.out.println(report);
        }

        double rate = (double)success / NUM_TESTS;

        System.out.printf("accuracy = %d of %d or %4.2f%% perturbance = %5.2f%%\n",success, NUM_TESTS,
                rate, PERTURBANCE);
    }
}