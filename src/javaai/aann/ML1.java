/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Lab 1
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.aann;

import java.util.*;

/**
 * This class reads the iris csv file, populates a hashmap for
 * classification purposes and tests classification.
 * @author Abuchi Obiegbu
 * @version 1.0
 */
public class ML1 extends BaseML{
    /**
     * Loads data and populate a hashmap for classification purposes
     * @param target species hashmap
     */
    public static void load(Map<Measure, Species> target) {
        // Load iris data from csv
        BaseML.load();

        // Store data in a hashmap for classification purposes
        for(int k=0; k < measures.size(); k++) {
            target.put(measures.get(k), flowers.get(k));
        }
    }

    public static void main(String[] args) {
        // species hashmap
        Map<Measure, Species> map = new HashMap<>();    // HashMap is a Map -> Polymorphism

        // populate species hashmap
        load(map);

        // list of test cases
        List<Measure> tests =
                new ArrayList<>(Arrays.asList(
                        measures.get(93),   // get the measure found at index 93 instead of using the string
                        measures.get(1),
                        measures.get(2),
                        measures.get(3),
                        measures.get(4)));

        // testing classification
        for(Measure measure: tests) {
            System.out.println(map.get(measure) + " " + measure);
        }
    }
}

