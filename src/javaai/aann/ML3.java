/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Lab 3
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.aann;

import java.util.HashMap;
import java.util.Map;

/**
 * This class reads the iris csv file, populate the ideals and NNS
 * then test and compare the classification of both for accuracy
 * @author Abuchi Obiegbu
 * @version 1.0
 */
public class ML3 extends ML2 {

    public final static double TRAINING_PERCENT = 0.80;

    public static void main(String[] args) {
        // ideals hashmap
        Map<Measure, Species> ideals = new HashMap<>();

        // load ideals
        load(ideals);

        // add random measure that has no mapping to get a null
        measures.add(new Measure());

        // calculate the test start index
        int testingStart = (int) (ideals.size() * TRAINING_PERCENT);

        // NNS hashmap
        NNSMap nns = new NNSMap();

        // train NNS
        for(int k=0; k < testingStart; k++)
            nns.put(measures.get(k), flowers.get(k));

        // performance instruments
        int tried = 0;
        int missed = 0;

        // reporting variables
        String report = "";
        // populating header display format
        String header = String.format("%2s  %-25s %10s %11s","#","Measure","Ideal","Actual");
        // display report header
        System.out.println(header);

        // testing driver
        for (int k = testingStart; k < measures.size(); k++) {
            // get a test measure
            Measure test = measures.get(k);

            // increment number of tries
            tried++;

            // tests ideals
            Species ideal = ideals.get(test);

            // test NNS
            Species actual = nns.get(test);

            // populate report string for accurate tests
            report = String.format("%2d. %9s %11s %11s", tried, test, ideal, actual);

            // compare actual and ideal
            if (!actual.equals(ideal)) {
                // populate report string for missed tests
                report += String.format("%8s", "MISSED!");

                // increment number of misses
                missed++;
            }
            // display report content
            System.out.println(report);
        }

        // store number of accurate tests
        int accurateTests = tried-missed;

        // store percent of accuracy
        double accuracyPercent = (accurateTests/(double)tried) * 100;

        // populate the summary string
        String summary = String.format("accuracy: %s of %s or %.0f%s", accurateTests, tried, accuracyPercent, "%");

        // display accuracy summary
        System.out.println(summary);
    }
}
