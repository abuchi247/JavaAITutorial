/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Lab 2
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.aann;

import java.util.HashMap;
import java.util.Set;

/**
 * This class implements the NNS hashmap for classification.
 * @author Abuchi Obiegbu
 * @version 1
 */
public class NNSMap extends HashMap<Measure, Species> {


    /**
     * Gets the NNS species prediction.
     * @param dest destination measure
     * @return A nearest neighbor species
     */
    @Override
    public Species get(Object dest) {
        // Ensure dest is instance of Measure
        if (!(dest instanceof Measure))
            return null;
        // Starting minimum distance -- the maximum possible value
        double minDist = Double.MAX_VALUE;
        // Arbitrarily choose a nearest measure
        Set<Measure> keys = this.keySet();

        Measure nearest = (Measure) keys.toArray()[0];
        // Search each measure in the hashmap
        for (Measure src : keys) {
            // get the distance from this src to dest measure
            double dist = getDistance(src, (Measure) dest);
            // If we’re closer than before, update the nearest
            if (dist < minDist) {
                minDist = dist;
                nearest = src;
            }
        }
        // get the species prediction for the nearest one
        Species prediction = super.get(nearest);
        return prediction;
    }

    /**
     * Calculate and get the Euclidean distance metric
     * @param src a source measure from the iris data
     * @param dest destination measure
     * @return Square distance
     */
    protected double getDistance(Measure src, Measure dest) { // This is the accumulator
        double dist2 = 0;
        // The sepal & petal values are in a 4D array.
        for (int k = 0; k < src.values.length; k++) {
            // Get the difference or delta
            double delta = src.values[k] - dest.values[k];
            // Sum the square differences
            dist2 += (delta * delta);
        }
        // The metric is the sum of square differences.
        // The square root of dist2, is the Euclidean distance.
        return dist2;
    }
}
